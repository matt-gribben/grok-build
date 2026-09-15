//! Cursor's bidirectional `AgentService.Run` Connect transport.
//!
//! This module pins authorization to the validated Cursor regional origin and
//! owns the open request-body channel needed for same-stream client messages.
//! It does not interpret Grok tools or read local credentials.

use std::convert::Infallible;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use futures_util::{StreamExt, stream};
use prost::Message;
use reqwest::header::{CONTENT_TYPE, HeaderValue};
use reqwest::{Client, Response, StatusCode, Url};
use thiserror::Error;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

use crate::cursor_catalog::{CursorCatalogError, resolve_agent_origin};
use crate::cursor_connect::{
    ConnectFrame, ConnectFrameDecoder, ConnectFrameError, encode_connect_frame,
};
use crate::cursor_proto::agent::v1::{
    AgentClientMessage, AgentServerMessage, agent_client_message, agent_server_message,
};

const RUN_RPC: &str = "/agent.v1.AgentService/Run";
const DEFAULT_CURSOR_CLIENT_VERSION: &str = "cli-2026.07.23-e383d2b";
const CONNECT_DATA_FRAME: u8 = 0;
const CONNECT_END_STREAM_FLAG: u8 = 0b0000_0010;
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const REQUEST_CHANNEL_CAPACITY: usize = 8;
const RESPONSE_CHANNEL_CAPACITY: usize = 256;
const MAX_RUN_FRAME_BYTES: usize = 17 * 1024 * 1024;
const RESPONSE_QUEUE_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const RESPONSE_QUEUE_BUDGET_UNIT_BYTES: usize = 4 * 1024;
const MAX_RUN_PROTO_FIELDS: usize = 65_536;
const MAX_RUN_PROTO_DEPTH: usize = 32;
const MAX_RUN_PROTO_SCAN_BYTES: usize = 64 * 1024 * 1024;
const PROTO_FIELD_HEAP_ESTIMATE_BYTES: usize = 128;
const RAW_FRAME_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const RAW_FRAME_BUDGET_UNIT_BYTES: usize = 4 * 1024;
const HTTP_CHUNK_BUFFER_BUDGET_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_RESPONSE_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_ACTIVE_RUNS: usize = 16;
const MAX_FRAME_DECODE_BATCH_BYTES: usize = 64 * 1024;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

static RESPONSE_QUEUE_BUDGET: OnceLock<Arc<Semaphore>> = OnceLock::new();
static RAW_FRAME_BUDGET: OnceLock<Arc<Semaphore>> = OnceLock::new();
static HTTP_CHUNK_BUFFER_BUDGET: OnceLock<Arc<Semaphore>> = OnceLock::new();
static ACTIVE_RUN_SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn response_queue_budget() -> Arc<Semaphore> {
    RESPONSE_QUEUE_BUDGET
        .get_or_init(|| {
            Arc::new(Semaphore::new(
                (RESPONSE_QUEUE_BUDGET_BYTES / RESPONSE_QUEUE_BUDGET_UNIT_BYTES) as u32 as usize,
            ))
        })
        .clone()
}

fn raw_frame_budget() -> Arc<Semaphore> {
    RAW_FRAME_BUDGET
        .get_or_init(|| {
            Arc::new(Semaphore::new(
                (RAW_FRAME_BUDGET_BYTES / RAW_FRAME_BUDGET_UNIT_BYTES) as u32 as usize,
            ))
        })
        .clone()
}

fn http_chunk_buffer_budget() -> Arc<Semaphore> {
    HTTP_CHUNK_BUFFER_BUDGET
        .get_or_init(|| {
            Arc::new(Semaphore::new(
                (HTTP_CHUNK_BUFFER_BUDGET_BYTES / RAW_FRAME_BUDGET_UNIT_BYTES) as u32 as usize,
            ))
        })
        .clone()
}

fn active_run_slots() -> Arc<Semaphore> {
    ACTIVE_RUN_SLOTS
        .get_or_init(|| Arc::new(Semaphore::new(MAX_ACTIVE_RUNS)))
        .clone()
}

/// A decoded message retains a budget permit until the sampler finishes handling it.
pub(crate) struct QueuedRunMessage {
    pub(crate) message: AgentServerMessage,
    _budget: OwnedSemaphorePermit,
}

fn response_budget_units(wire_bytes: usize, protobuf_fields: usize) -> Option<u32> {
    // The wire preflight bounds every nested field occurrence before Prost
    // allocates. Charge payload copies plus conservative per-field collection,
    // map, and message overhead.
    let estimated = wire_bytes
        .saturating_mul(2)
        .saturating_add(protobuf_fields.saturating_mul(PROTO_FIELD_HEAP_ESTIMATE_BYTES));
    let units = estimated.saturating_add(RESPONSE_QUEUE_BUDGET_UNIT_BYTES - 1)
        / RESPONSE_QUEUE_BUDGET_UNIT_BYTES;
    u32::try_from(units.max(1)).ok()
}

#[derive(Clone, Copy, Debug, Default)]
struct ProtoWireStats {
    fields: usize,
    scanned_bytes: usize,
}

fn restore_proto_stats(stats: &mut ProtoWireStats, snapshot: ProtoWireStats) {
    let scanned_bytes = stats.scanned_bytes;
    *stats = snapshot;
    stats.scanned_bytes = scanned_bytes;
}

/// Validate the protobuf wire shape and estimate allocations before Prost
/// decodes it. Length-delimited values are recursively inspected only when
/// they also form a complete protobuf message; ordinary text/blob payloads
/// remain opaque. Total field count, nesting, and scan work are all capped.
fn preflight_server_message(payload: &[u8]) -> Result<usize, CursorTransportError> {
    let mut stats = ProtoWireStats::default();
    scan_wire_message(payload, 0, true, &mut stats)?;
    Ok(stats.fields)
}

fn scan_wire_message(
    bytes: &[u8],
    depth: usize,
    top_level: bool,
    stats: &mut ProtoWireStats,
) -> Result<bool, CursorTransportError> {
    if depth > MAX_RUN_PROTO_DEPTH {
        return Err(CursorTransportError::ResponseBufferLimit);
    }
    stats.scanned_bytes = stats.scanned_bytes.saturating_add(bytes.len());
    if stats.scanned_bytes > MAX_RUN_PROTO_SCAN_BYTES {
        return Err(CursorTransportError::ResponseBufferLimit);
    }
    let original = *stats;
    let mut offset = 0;
    while offset < bytes.len() {
        let key = match take_wire_varint(bytes, &mut offset) {
            Some(value) if value >> 3 != 0 => value,
            _ => {
                restore_proto_stats(stats, original);
                return if top_level {
                    Err(CursorTransportError::InvalidResponse)
                } else {
                    Ok(false)
                };
            }
        };
        stats.fields = stats.fields.saturating_add(1);
        if stats.fields > MAX_RUN_PROTO_FIELDS {
            return Err(CursorTransportError::ResponseBufferLimit);
        }
        match key & 7 {
            0 => {
                if take_wire_varint(bytes, &mut offset).is_none() {
                    restore_proto_stats(stats, original);
                    return if top_level {
                        Err(CursorTransportError::InvalidResponse)
                    } else {
                        Ok(false)
                    };
                }
            }
            1 => {
                offset = match offset.checked_add(8).filter(|end| *end <= bytes.len()) {
                    Some(end) => end,
                    None => {
                        restore_proto_stats(stats, original);
                        return if top_level {
                            Err(CursorTransportError::InvalidResponse)
                        } else {
                            Ok(false)
                        };
                    }
                };
            }
            2 => {
                let length = match take_wire_varint(bytes, &mut offset)
                    .and_then(|value| usize::try_from(value).ok())
                {
                    Some(length) => length,
                    None => {
                        restore_proto_stats(stats, original);
                        return if top_level {
                            Err(CursorTransportError::InvalidResponse)
                        } else {
                            Ok(false)
                        };
                    }
                };
                let end = match offset.checked_add(length).filter(|end| *end <= bytes.len()) {
                    Some(end) => end,
                    None => {
                        restore_proto_stats(stats, original);
                        return if top_level {
                            Err(CursorTransportError::InvalidResponse)
                        } else {
                            Ok(false)
                        };
                    }
                };
                if length > 0 {
                    let snapshot = *stats;
                    if !scan_wire_message(&bytes[offset..end], depth + 1, false, stats)? {
                        restore_proto_stats(stats, snapshot);
                    }
                }
                offset = end;
            }
            3 => {
                let field_number = key >> 3;
                match scan_wire_group(bytes, offset, field_number, depth + 1, stats)? {
                    Some(end) => offset = end,
                    None => {
                        restore_proto_stats(stats, original);
                        return if top_level {
                            Err(CursorTransportError::InvalidResponse)
                        } else {
                            Ok(false)
                        };
                    }
                }
            }
            5 => {
                offset = match offset.checked_add(4).filter(|end| *end <= bytes.len()) {
                    Some(end) => end,
                    None => {
                        restore_proto_stats(stats, original);
                        return if top_level {
                            Err(CursorTransportError::InvalidResponse)
                        } else {
                            Ok(false)
                        };
                    }
                };
            }
            // End-group tags are valid only inside scan_wire_group.
            _ => {
                restore_proto_stats(stats, original);
                return if top_level {
                    Err(CursorTransportError::InvalidResponse)
                } else {
                    Ok(false)
                };
            }
        }
    }
    Ok(true)
}

fn scan_wire_group(
    bytes: &[u8],
    mut offset: usize,
    expected_end_field: u64,
    depth: usize,
    stats: &mut ProtoWireStats,
) -> Result<Option<usize>, CursorTransportError> {
    if depth > MAX_RUN_PROTO_DEPTH {
        return Err(CursorTransportError::ResponseBufferLimit);
    }
    while offset < bytes.len() {
        let Some(key) = take_wire_varint(bytes, &mut offset) else {
            return Ok(None);
        };
        let field_number = key >> 3;
        if field_number == 0 {
            return Ok(None);
        }
        stats.fields = stats.fields.saturating_add(1);
        if stats.fields > MAX_RUN_PROTO_FIELDS {
            return Err(CursorTransportError::ResponseBufferLimit);
        }
        match key & 7 {
            0 => {
                if take_wire_varint(bytes, &mut offset).is_none() {
                    return Ok(None);
                }
            }
            1 => {
                let Some(end) = offset.checked_add(8).filter(|end| *end <= bytes.len()) else {
                    return Ok(None);
                };
                offset = end;
            }
            2 => {
                let Some(length) = take_wire_varint(bytes, &mut offset)
                    .and_then(|value| usize::try_from(value).ok())
                else {
                    return Ok(None);
                };
                let Some(end) = offset.checked_add(length).filter(|end| *end <= bytes.len()) else {
                    return Ok(None);
                };
                if length > 0 {
                    let snapshot = *stats;
                    if !scan_wire_message(&bytes[offset..end], depth + 1, false, stats)? {
                        restore_proto_stats(stats, snapshot);
                    }
                }
                offset = end;
            }
            3 => {
                let Some(end) = scan_wire_group(bytes, offset, field_number, depth + 1, stats)?
                else {
                    return Ok(None);
                };
                offset = end;
            }
            4 => {
                return Ok((field_number == expected_end_field).then_some(offset));
            }
            5 => {
                let Some(end) = offset.checked_add(4).filter(|end| *end <= bytes.len()) else {
                    return Ok(None);
                };
                offset = end;
            }
            _ => return Ok(None),
        }
    }
    Ok(None)
}

fn take_wire_varint(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*offset)?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return None;
        }
        value |= u64::from(byte & 0x7f).checked_shl(shift)?;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CursorTransportError {
    #[error("Cursor inference requires the signed-in Cursor Desktop session.")]
    MissingCredential,
    #[error("Cursor rejected the local session. Sign in to Cursor Desktop and retry.")]
    Unauthorized,
    #[error("Cursor inference failed with HTTP status {0}.")]
    HttpStatus(u16),
    #[error("Cursor inference transport could not be initialized.")]
    TransportUnavailable,
    #[error("Cursor inference was cancelled.")]
    Cancelled,
    #[error("Cursor returned an invalid Connect response.")]
    InvalidResponse,
    #[error("Cursor returned a compressed Connect frame that this adapter does not support.")]
    UnsupportedCompression,
    #[error("Cursor returned a malformed Connect frame.")]
    InvalidFrame,
    #[error("Cursor returned an error frame.")]
    RemoteError,
    #[error("Cursor response buffering reached its process-wide memory limit.")]
    ResponseBufferLimit,
}

impl From<ConnectFrameError> for CursorTransportError {
    fn from(_: ConnectFrameError) -> Self {
        Self::InvalidFrame
    }
}

impl From<CursorCatalogError> for CursorTransportError {
    fn from(error: CursorCatalogError) -> Self {
        match error {
            CursorCatalogError::UnsafeEndpoint => Self::TransportUnavailable,
            _ => Self::TransportUnavailable,
        }
    }
}

/// A cloneable sender for additional client messages on one parked Run stream.
#[derive(Clone)]
pub(crate) struct CursorRunSender {
    tx: mpsc::Sender<Bytes>,
    cancellation: CancellationToken,
}

impl std::fmt::Debug for CursorRunSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorRunSender")
            .field("closed", &self.tx.is_closed())
            .field("cancelled", &self.cancellation.is_cancelled())
            .finish()
    }
}

impl CursorRunSender {
    pub(crate) async fn send_message(
        &self,
        message: &AgentClientMessage,
    ) -> Result<(), CursorTransportError> {
        self.send_message_cancellable(message, &CancellationToken::new())
            .await
    }

    pub(crate) async fn send_message_cancellable(
        &self,
        message: &AgentClientMessage,
        request_cancellation: &CancellationToken,
    ) -> Result<(), CursorTransportError> {
        let payload = message.encode_to_vec();
        let frame = encode_connect_frame(CONNECT_DATA_FRAME, &payload)?;
        tokio::select! {
            biased;
            _ = self.cancellation.cancelled() => Err(CursorTransportError::Cancelled),
            _ = request_cancellation.cancelled() => Err(CursorTransportError::Cancelled),
            result = self.tx.send(Bytes::from(frame)) => {
                result.map_err(|_| CursorTransportError::TransportUnavailable)
            }
        }
    }
}

/// Open Cursor Run stream and expose decoded server messages incrementally.
pub(crate) struct CursorRunConnection {
    incoming: mpsc::Receiver<Result<QueuedRunMessage, CursorTransportError>>,
    sender: CursorRunSender,
    cancellation: CancellationToken,
    _active_slot: OwnedSemaphorePermit,
}

impl std::fmt::Debug for CursorRunConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorRunConnection")
            .field("incoming_queue", &self.incoming.len())
            .field("request_sender", &self.sender)
            .field("cancelled", &self.cancellation.is_cancelled())
            .finish()
    }
}

impl Drop for CursorRunConnection {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Dedicated Cursor `AgentService.Run` client. It ignores Grok model URLs,
/// API keys, extra headers, redirects, and proxy settings by construction.
#[derive(Clone)]
pub(crate) struct CursorRunTransport {
    http: Client,
    agent_origin: Url,
}

impl std::fmt::Debug for CursorRunTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorRunTransport")
            .field("agent_origin", &self.agent_origin)
            .finish_non_exhaustive()
    }
}

impl CursorRunTransport {
    pub(crate) fn new() -> Result<Self, CursorTransportError> {
        let agent_origin = resolve_agent_origin()?;
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .no_proxy()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .build()
            .map_err(|_| CursorTransportError::TransportUnavailable)?;
        Ok(Self { http, agent_origin })
    }

    #[cfg(test)]
    pub(crate) fn from_test_origin(origin: &str, certificate: reqwest::Certificate) -> Self {
        let http = Client::builder()
            .add_root_certificate(certificate)
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .no_proxy()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .build()
            .expect("test HTTP/2 client");
        Self {
            http,
            agent_origin: Url::parse(origin).expect("test origin"),
        }
    }

    pub(crate) async fn open(
        &self,
        initial_frame: Vec<u8>,
        access_token: &str,
        request_cancellation: CancellationToken,
    ) -> Result<CursorRunConnection, CursorTransportError> {
        if access_token.is_empty() {
            return Err(CursorTransportError::MissingCredential);
        }
        if request_cancellation.is_cancelled() {
            return Err(CursorTransportError::Cancelled);
        }
        let active_slot = tokio::select! {
            biased;
            _ = request_cancellation.cancelled() => return Err(CursorTransportError::Cancelled),
            result = tokio::time::timeout(HTTP_CONNECT_TIMEOUT, active_run_slots().acquire_owned()) => {
                match result {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) | Err(_) => return Err(CursorTransportError::TransportUnavailable),
                }
            }
        };
        let url = self
            .agent_origin
            .join(RUN_RPC)
            .map_err(|_| CursorTransportError::TransportUnavailable)?;
        let (tx, rx) = mpsc::channel::<Bytes>(REQUEST_CHANNEL_CAPACITY);
        let bridge_cancellation = CancellationToken::new();
        let sender = CursorRunSender {
            tx,
            cancellation: bridge_cancellation.clone(),
        };
        // Buffer the first Connect frame before starting HTTP/2 so the server
        // sees a complete Run request immediately while the body stays open.
        sender
            .tx
            .send(Bytes::from(initial_frame))
            .await
            .map_err(|_| CursorTransportError::TransportUnavailable)?;
        let request_body = stream::unfold(rx, |mut receiver| async move {
            receiver
                .recv()
                .await
                .map(|bytes| (Ok::<Bytes, Infallible>(bytes), receiver))
        });
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = self
            .http
            .post(url)
            .header(
                CONTENT_TYPE,
                HeaderValue::from_static("application/connect+proto"),
            )
            .header("connect-protocol-version", "1")
            .header("te", "trailers")
            .header("x-ghost-mode", "true")
            .header("x-cursor-client-version", DEFAULT_CURSOR_CLIENT_VERSION)
            .header("x-cursor-client-type", "cli")
            .header("x-request-id", request_id)
            .bearer_auth(access_token)
            .body(reqwest::Body::wrap_stream(request_body))
            .build()
            .map_err(|_| CursorTransportError::TransportUnavailable)?;
        let response = tokio::select! {
            biased;
            _ = request_cancellation.cancelled() => return Err(CursorTransportError::Cancelled),
            response = tokio::time::timeout(HTTP_CONNECT_TIMEOUT, self.http.execute(request)) => {
                match response {
                    Ok(Ok(response)) => response,
                    Ok(Err(_)) | Err(_) => return Err(CursorTransportError::TransportUnavailable),
                }
            }
        };
        validate_run_response(&response)?;
        let (incoming_tx, incoming) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        tokio::spawn(drive_run_response(
            response.bytes_stream().boxed(),
            sender.clone(),
            incoming_tx,
            bridge_cancellation.clone(),
        ));
        Ok(CursorRunConnection {
            incoming,
            sender,
            cancellation: bridge_cancellation,
            _active_slot: active_slot,
        })
    }
}

fn validate_run_response(response: &Response) -> Result<(), CursorTransportError> {
    let status = response.status();
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return Err(CursorTransportError::Unauthorized);
    }
    if !status.is_success() {
        return Err(CursorTransportError::HttpStatus(status.as_u16()));
    }
    if !response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(';').next().is_some_and(|media_type| {
                media_type
                    .trim()
                    .eq_ignore_ascii_case("application/connect+proto")
            })
        })
    {
        return Err(CursorTransportError::InvalidResponse);
    }
    Ok(())
}

impl CursorRunConnection {
    pub(crate) fn sender(&self) -> CursorRunSender {
        self.sender.clone()
    }

    pub(crate) async fn next_message(
        &mut self,
    ) -> Result<Option<QueuedRunMessage>, CursorTransportError> {
        match self.incoming.recv().await {
            Some(Ok(message)) => Ok(Some(message)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
    }

    #[cfg(test)]
    pub(crate) fn test_connection(cancellation: CancellationToken) -> Self {
        let (request_tx, _request_rx) = mpsc::channel(1);
        let (incoming_tx, incoming) = mpsc::channel(1);
        drop(incoming_tx);
        Self {
            incoming,
            sender: CursorRunSender {
                tx: request_tx,
                cancellation: cancellation.clone(),
            },
            cancellation,
            _active_slot: active_run_slots()
                .try_acquire_owned()
                .expect("test active stream slot"),
        }
    }
}

async fn drive_run_response(
    mut stream: futures_util::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>,
    sender: CursorRunSender,
    incoming: mpsc::Sender<Result<QueuedRunMessage, CursorTransportError>>,
    cancellation: CancellationToken,
) {
    let mut decoder = ConnectFrameDecoder::with_max_frame_bytes(MAX_RUN_FRAME_BYTES);
    let mut raw_pending_permit: Option<OwnedSemaphorePermit> = None;
    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.tick().await;
    'driver: loop {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => break,
            _ = heartbeat.tick() => {
                if let Err(error) = send_client_heartbeat(&sender).await {
                    let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                    break;
                }
            }
            bytes = stream.next() => {
                let Some(bytes) = bytes else {
                    if let Err(error) = decoder.finish() {
                        let _ = queue_run_event(&incoming, Err(error.into()), &sender, &cancellation, &mut heartbeat).await;
                    }
                    break;
                };
                let bytes = match bytes {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        let _ = queue_run_event(&incoming, Err(CursorTransportError::TransportUnavailable), &sender, &cancellation, &mut heartbeat).await;
                        break;
                    }
                };
                if bytes.len() > MAX_HTTP_RESPONSE_CHUNK_BYTES {
                    let _ = queue_run_event(&incoming, Err(CursorTransportError::ResponseBufferLimit), &sender, &cancellation, &mut heartbeat).await;
                    break;
                }
                let input_units = bytes
                    .len()
                    .saturating_add(RAW_FRAME_BUDGET_UNIT_BYTES - 1)
                    / RAW_FRAME_BUDGET_UNIT_BYTES;
                let input_units = u32::try_from(input_units.max(1)).unwrap_or(u32::MAX);
                let _input_permit = match acquire_with_heartbeat(
                    http_chunk_buffer_budget(),
                    input_units,
                    &sender,
                    &incoming,
                    &cancellation,
                    &mut heartbeat,
                ).await {
                    Ok(permit) => permit,
                    Err(error) => {
                        let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                        break;
                    }
                };
                let mut ended = false;
                for chunk in bytes.chunks(MAX_FRAME_DECODE_BATCH_BYTES) {
                    if raw_pending_permit.is_none() {
                        match reserve_next_raw_frame(
                            &decoder,
                            chunk,
                            chunk.len(),
                            &sender,
                            &incoming,
                            &cancellation,
                            &mut heartbeat,
                        )
                        .await
                        {
                            Ok(permit) => raw_pending_permit = permit,
                            Err(error) => {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                        }
                    }
                    let frames = match decoder.push(chunk) {
                        Ok(frames) => frames,
                        Err(error) => {
                            let _ = queue_run_event(&incoming, Err(error.into()), &sender, &cancellation, &mut heartbeat).await;
                            break 'driver;
                        }
                    };
                    for (index, frame) in frames.iter().enumerate() {
                        if frame.is_end_stream() {
                            if index + 1 != frames.len() {
                                let _ = queue_run_event(&incoming, Err(CursorTransportError::InvalidResponse), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                            if let Err(error) = validate_end_stream_frame(frame) {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                            }
                            ended = true;
                            break;
                        }
                        if let Err(error) = validate_server_frame_flags(frame) {
                            let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                            break 'driver;
                        }
                        let field_count = match preflight_server_message(&frame.payload) {
                            Ok(field_count) => field_count,
                            Err(error) => {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                        };
                        let Some(units) = response_budget_units(frame.payload.len(), field_count) else {
                            let _ = queue_run_event(&incoming, Err(CursorTransportError::ResponseBufferLimit), &sender, &cancellation, &mut heartbeat).await;
                            break 'driver;
                        };
                        let permit = match acquire_with_heartbeat(
                            response_queue_budget(),
                            units,
                            &sender,
                            &incoming,
                            &cancellation,
                            &mut heartbeat,
                        ).await {
                            Ok(permit) => permit,
                            Err(error) => {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                        };
                        let message = match decode_server_frame(frame) {
                            Ok(message) => message,
                            Err(error) => {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                        };
                        if is_server_heartbeat(&message) {
                            continue;
                        }
                        let queued = QueuedRunMessage {
                            message,
                            _budget: permit,
                        };
                        if queue_run_event(&incoming, Ok(queued), &sender, &cancellation, &mut heartbeat).await.is_err() {
                            break 'driver;
                        }
                    }
                    if ended {
                        break;
                    }
                    if !frames.is_empty() {
                        raw_pending_permit = None;
                    }
                    if raw_pending_permit.is_none() && decoder.pending_len() > 0 {
                        match reserve_next_raw_frame(
                            &decoder,
                            &[],
                            0,
                            &sender,
                            &incoming,
                            &cancellation,
                            &mut heartbeat,
                        )
                        .await
                        {
                            Ok(permit) => raw_pending_permit = permit,
                            Err(error) => {
                                let _ = queue_run_event(&incoming, Err(error), &sender, &cancellation, &mut heartbeat).await;
                                break 'driver;
                            }
                        }
                    }
                }
                if ended {
                    if let Err(error) = decoder.finish() {
                        let _ = queue_run_event(&incoming, Err(error.into()), &sender, &cancellation, &mut heartbeat).await;
                    }
                    break;
                }
            }
        }
    }
}

async fn reserve_next_raw_frame(
    decoder: &ConnectFrameDecoder,
    lookahead: &[u8],
    chunk_bytes: usize,
    sender: &CursorRunSender,
    incoming: &mpsc::Sender<Result<QueuedRunMessage, CursorTransportError>>,
    cancellation: &CancellationToken,
    heartbeat: &mut tokio::time::Interval,
) -> Result<Option<OwnedSemaphorePermit>, CursorTransportError> {
    let Some(frame_bytes) = decoder.next_frame_total_len_with(lookahead)? else {
        return Ok(None);
    };
    // Reserve atomically for Vec capacity growth plus the copied frame payload.
    // This prevents several connections from each holding a partial allocation
    // while waiting for more process-wide raw-frame permits.
    let estimated = frame_bytes
        .saturating_mul(3)
        .saturating_add(chunk_bytes.saturating_mul(8));
    let units =
        estimated.saturating_add(RAW_FRAME_BUDGET_UNIT_BYTES - 1) / RAW_FRAME_BUDGET_UNIT_BYTES;
    let units = u32::try_from(units.max(1)).unwrap_or(u32::MAX);
    acquire_with_heartbeat(
        raw_frame_budget(),
        units,
        sender,
        incoming,
        cancellation,
        heartbeat,
    )
    .await
    .map(Some)
}

async fn send_client_heartbeat(sender: &CursorRunSender) -> Result<(), CursorTransportError> {
    let message = AgentClientMessage {
        message: Some(agent_client_message::Message::ClientHeartbeat(
            crate::cursor_proto::agent::v1::ClientHeartbeat {},
        )),
    };
    sender.send_message(&message).await
}

async fn acquire_with_heartbeat(
    budget: Arc<Semaphore>,
    units: u32,
    sender: &CursorRunSender,
    incoming: &mpsc::Sender<Result<QueuedRunMessage, CursorTransportError>>,
    cancellation: &CancellationToken,
    heartbeat: &mut tokio::time::Interval,
) -> Result<OwnedSemaphorePermit, CursorTransportError> {
    let mut acquire = Box::pin(budget.acquire_many_owned(units));
    loop {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(CursorTransportError::Cancelled),
            result = &mut acquire => return result.map_err(|_| CursorTransportError::ResponseBufferLimit),
            _ = heartbeat.tick() => send_client_heartbeat(sender).await?,
        }
        if incoming.is_closed() {
            return Err(CursorTransportError::TransportUnavailable);
        }
    }
}

async fn queue_run_event(
    incoming: &mpsc::Sender<Result<QueuedRunMessage, CursorTransportError>>,
    event: Result<QueuedRunMessage, CursorTransportError>,
    sender: &CursorRunSender,
    cancellation: &CancellationToken,
    heartbeat: &mut tokio::time::Interval,
) -> Result<(), CursorTransportError> {
    let mut event = Some(event);
    loop {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(CursorTransportError::Cancelled),
            permit = incoming.reserve() => {
                permit
                    .map_err(|_| CursorTransportError::TransportUnavailable)?
                    .send(event.take().expect("queued event is sent once"));
                return Ok(());
            }
            _ = heartbeat.tick() => send_client_heartbeat(sender).await?,
        }
    }
}

fn is_server_heartbeat(message: &AgentServerMessage) -> bool {
    matches!(
        &message.message,
        Some(agent_server_message::Message::InteractionUpdate(update))
            if matches!(&update.message, Some(crate::cursor_proto::agent::v1::interaction_update::Message::Heartbeat(_)))
    )
}

fn decode_server_frame(frame: &ConnectFrame) -> Result<AgentServerMessage, CursorTransportError> {
    validate_server_frame_flags(frame)?;
    AgentServerMessage::decode(frame.payload.as_slice())
        .map_err(|_| CursorTransportError::InvalidResponse)
}

fn validate_server_frame_flags(frame: &ConnectFrame) -> Result<(), CursorTransportError> {
    if frame.flags & !(CONNECT_END_STREAM_FLAG | 0b0000_0001) != 0 {
        return Err(CursorTransportError::InvalidResponse);
    }
    if frame.flags & CONNECT_END_STREAM_FLAG != 0 {
        return Err(CursorTransportError::InvalidResponse);
    }
    if frame.flags & 0b0000_0001 != 0 {
        return Err(CursorTransportError::UnsupportedCompression);
    }
    Ok(())
}

fn validate_end_stream_frame(frame: &ConnectFrame) -> Result<(), CursorTransportError> {
    if frame.flags & !(CONNECT_END_STREAM_FLAG | 0b0000_0001) != 0 {
        return Err(CursorTransportError::InvalidResponse);
    }
    if frame.flags & 0b0000_0001 != 0 {
        return Err(CursorTransportError::UnsupportedCompression);
    }
    if frame.flags & CONNECT_END_STREAM_FLAG == 0 {
        return Err(CursorTransportError::InvalidResponse);
    }
    if frame.payload.is_empty() {
        return Ok(());
    }
    let value: serde_json::Value = serde_json::from_slice(&frame.payload)
        .map_err(|_| CursorTransportError::InvalidResponse)?;
    if value.get("error").is_some() {
        return Err(CursorTransportError::RemoteError);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, Response, Version};
    use axum::routing::post;
    use prost::Message;
    use rcgen::{CertificateParams, KeyPair};
    use rustls::pki_types::PrivateKeyDer;

    use super::*;
    use crate::cursor_connect::encode_connect_frame;
    use crate::cursor_proto::agent::v1::{
        AgentClientMessage, AgentServerMessage, InteractionUpdate, TextDeltaUpdate,
        agent_client_message, agent_server_message, interaction_update,
    };
    use crate::cursor_request::CursorRunPayload;

    #[derive(Clone, Debug)]
    struct CapturedRunRequest {
        authorization: Option<String>,
        content_type: Option<String>,
        version: Version,
        grok_headers: Vec<String>,
        first_frame: Vec<u8>,
        received_heartbeat: bool,
    }

    async fn capture_run(
        State(state): State<Arc<Mutex<Option<CapturedRunRequest>>>>,
        request: Request<Body>,
    ) -> Response<Body> {
        use futures_util::StreamExt;

        let authorization = request
            .headers()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let content_type = request
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let version = request.version();
        let grok_headers = request
            .headers()
            .keys()
            .filter(|name| name.as_str().starts_with("x-grok-"))
            .map(|name| name.as_str().to_owned())
            .collect();
        let mut body = request.into_body().into_data_stream();
        let first_frame = body
            .next()
            .await
            .expect("Run request frame")
            .expect("Run request bytes")
            .to_vec();
        *state.lock().expect("state lock") = Some(CapturedRunRequest {
            authorization,
            content_type,
            version,
            grok_headers,
            first_frame,
            received_heartbeat: false,
        });
        let response_stream = stream::once(async move {
            // Hold the response open until the client writes another message;
            // this exercises the bidirectional request-body channel.
            let heartbeat = body
                .next()
                .await
                .expect("client heartbeat frame")
                .expect("client heartbeat bytes");
            let mut decoder = ConnectFrameDecoder::default();
            let frames = decoder.push(&heartbeat).expect("heartbeat frame");
            let heartbeat_received = frames.iter().any(|frame| {
                matches!(
                    AgentClientMessage::decode(frame.payload.as_slice())
                        .ok()
                        .and_then(|message| message.message),
                    Some(agent_client_message::Message::ClientHeartbeat(_))
                )
            });
            state
                .lock()
                .expect("state lock")
                .as_mut()
                .expect("captured request")
                .received_heartbeat = heartbeat_received;
            let response_message = AgentServerMessage {
                message: Some(agent_server_message::Message::InteractionUpdate(
                    InteractionUpdate {
                        message: Some(interaction_update::Message::TextDelta(TextDeltaUpdate {
                            text: "Cursor stream connected".to_owned(),
                        })),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            };
            let data = encode_connect_frame(CONNECT_DATA_FRAME, &response_message.encode_to_vec())
                .expect("data frame");
            let end = encode_connect_frame(CONNECT_END_STREAM_FLAG, b"{}").expect("end frame");
            Ok::<Bytes, Infallible>(Bytes::from([data, end].concat()))
        });
        Response::builder()
            .header(CONTENT_TYPE, "application/connect+proto")
            .body(Body::from_stream(response_stream))
            .expect("response")
    }

    async fn start_server(
        state: Arc<Mutex<Option<CapturedRunRequest>>>,
    ) -> (String, reqwest::Certificate, tokio::task::JoinHandle<()>) {
        let key_pair = KeyPair::generate().expect("test key");
        let certificate = CertificateParams::new(vec!["localhost".to_owned()])
            .expect("certificate params")
            .self_signed(&key_pair)
            .expect("self-sign");
        let private_key = PrivateKeyDer::Pkcs8(key_pair.serialize_der().into());
        let mut config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate.der().clone()], private_key)
            .expect("server config");
        config.alpn_protocols = vec![b"h2".to_vec()];
        let tls = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(config));
        let app = Router::new()
            .route(RUN_RPC, post(capture_run))
            .with_state(state);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind listener");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            axum_server::from_tcp_rustls(listener, tls)
                .expect("build TLS server")
                .serve(app.into_make_service())
                .await
                .expect("serve fake Cursor");
        });
        let certificate =
            reqwest::Certificate::from_der(certificate.der().as_ref()).expect("test certificate");
        (format!("https://localhost:{port}"), certificate, server)
    }

    fn test_transport(origin: &str, certificate: reqwest::Certificate) -> CursorRunTransport {
        let http = Client::builder()
            .add_root_certificate(certificate)
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .no_proxy()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .build()
            .expect("test HTTP/2 client");
        CursorRunTransport {
            http,
            agent_origin: Url::parse(origin).expect("test origin"),
        }
    }

    fn simple_payload() -> CursorRunPayload {
        let request = xai_grok_sampling_types::ConversationRequest {
            items: vec![xai_grok_sampling_types::ConversationItem::User(
                xai_grok_sampling_types::conversation::UserItem {
                    content: vec![xai_grok_sampling_types::conversation::ContentPart::Text {
                        text: "Current question".into(),
                    }],
                    ..Default::default()
                },
            )],
            model: Some("cursor:gpt-5.6".to_owned()),
            x_grok_session_id: Some("session-1".to_owned()),
            ..Default::default()
        };
        crate::cursor_request::build_cursor_run_payload(request).expect("payload")
    }

    #[tokio::test]
    async fn opens_run_over_http2_and_decodes_incremental_text_frame() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let state = Arc::new(Mutex::new(None));
        let (origin, certificate, server) = start_server(state.clone()).await;
        let transport = test_transport(&origin, certificate);
        let cancellation = CancellationToken::new();
        let mut connection = transport
            .open(
                simple_payload().initial_frame,
                "fixture-cursor-token",
                cancellation,
            )
            .await
            .expect("open Cursor stream");
        connection
            .sender()
            .send_message(&AgentClientMessage {
                message: Some(agent_client_message::Message::ClientHeartbeat(
                    crate::cursor_proto::agent::v1::ClientHeartbeat {},
                )),
            })
            .await
            .expect("send client heartbeat");
        let message = connection
            .next_message()
            .await
            .expect("receive Cursor message")
            .expect("text message");
        let Some(agent_server_message::Message::InteractionUpdate(update)) =
            message.message.message
        else {
            panic!("expected interaction update");
        };
        let Some(interaction_update::Message::TextDelta(text)) = update.message else {
            panic!("expected text delta");
        };
        assert_eq!(text.text, "Cursor stream connected");
        assert!(
            connection
                .next_message()
                .await
                .expect("end stream")
                .is_none()
        );

        let captured = state
            .lock()
            .expect("captured state")
            .clone()
            .expect("request");
        assert_eq!(captured.version, Version::HTTP_2);
        assert_eq!(
            captured.authorization.as_deref(),
            Some("Bearer fixture-cursor-token")
        );
        assert_eq!(
            captured.content_type.as_deref(),
            Some("application/connect+proto")
        );
        assert!(captured.grok_headers.is_empty());
        assert!(captured.received_heartbeat);
        let mut decoder = ConnectFrameDecoder::default();
        let frames = decoder.push(&captured.first_frame).expect("decode request");
        assert_eq!(frames.len(), 1);
        let initial =
            AgentClientMessage::decode(frames[0].payload.as_slice()).expect("Run request");
        assert!(matches!(
            initial.message,
            Some(agent_client_message::Message::RunRequest(_))
        ));
        server.abort();
    }

    #[tokio::test]
    async fn cursor_full_response_queue_applies_backpressure_without_dropping_frames() {
        let (request_tx, _request_rx) = mpsc::channel(REQUEST_CHANNEL_CAPACITY);
        let cancellation = CancellationToken::new();
        let sender = CursorRunSender {
            tx: request_tx,
            cancellation: cancellation.clone(),
        };
        let payload = AgentServerMessage::default().encode_to_vec();
        let frame = encode_connect_frame(CONNECT_DATA_FRAME, &payload).expect("frame");
        let message_count = RESPONSE_CHANNEL_CAPACITY + 4;
        let body_bytes = (0..message_count)
            .flat_map(|_| frame.iter().copied())
            .collect::<Vec<_>>();
        let body = stream::iter([Ok::<Bytes, reqwest::Error>(Bytes::from(body_bytes))]).boxed();
        let (incoming_tx, incoming_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);

        let driver = tokio::spawn(drive_run_response(body, sender, incoming_tx, cancellation));
        tokio::time::timeout(Duration::from_secs(1), async {
            while incoming_rx.len() < RESPONSE_CHANNEL_CAPACITY {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the bounded queue fills before applying backpressure");
        assert!(
            !driver.is_finished(),
            "reader backpressure must pause the driver"
        );
        assert_eq!(incoming_rx.len(), RESPONSE_CHANNEL_CAPACITY);
        let mut incoming_rx = incoming_rx;
        for _ in 0..message_count {
            incoming_rx
                .recv()
                .await
                .expect("every response frame must be delivered")
                .expect("frame is valid");
        }
        tokio::time::timeout(Duration::from_secs(1), driver)
            .await
            .expect("driver completes after the queue drains")
            .expect("driver task");
    }

    #[test]
    fn protobuf_preflight_rejects_excessive_empty_repeated_fields() {
        let wire = (0..=MAX_RUN_PROTO_FIELDS)
            .flat_map(|_| [0x0a_u8, 0x00])
            .collect::<Vec<_>>();
        assert_eq!(
            preflight_server_message(&wire),
            Err(CursorTransportError::ResponseBufferLimit)
        );
    }

    #[test]
    fn protobuf_preflight_counts_known_fields_before_a_matched_unknown_group() {
        fn push_varint(mut value: usize, output: &mut Vec<u8>) {
            while value >= 0x80 {
                output.push((value as u8) | 0x80);
                value >>= 7;
            }
            output.push(value as u8);
        }

        let mut checkpoint = vec![0xa3, 0x06, 0xa4, 0x06]; // unknown group 100
        checkpoint.extend((0..=MAX_RUN_PROTO_FIELDS).flat_map(|_| [0x42_u8, 0x00]));
        let mut wire = vec![0x1a]; // AgentServerMessage.conversation_checkpoint_update
        push_varint(checkpoint.len(), &mut wire);
        wire.extend(checkpoint);
        assert_eq!(
            preflight_server_message(&wire),
            Err(CursorTransportError::ResponseBufferLimit)
        );
    }

    #[tokio::test]
    async fn raw_frame_reservations_are_atomic_for_large_concurrent_frames() {
        let cancellation = CancellationToken::new();
        let (request_tx, _request_rx) = mpsc::channel(REQUEST_CHANNEL_CAPACITY);
        let sender = CursorRunSender {
            tx: request_tx,
            cancellation: cancellation.clone(),
        };
        let (incoming, _incoming_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
        heartbeat.tick().await;
        let mut header = vec![CONNECT_DATA_FRAME];
        header.extend_from_slice(&(MAX_RUN_FRAME_BYTES as u32).to_be_bytes());
        let first_decoder = ConnectFrameDecoder::with_max_frame_bytes(MAX_RUN_FRAME_BYTES);
        let first = reserve_next_raw_frame(
            &first_decoder,
            &header,
            0,
            &sender,
            &incoming,
            &cancellation,
            &mut heartbeat,
        )
        .await
        .expect("first reservation")
        .expect("declared frame is reserved");
        let second_decoder = ConnectFrameDecoder::with_max_frame_bytes(MAX_RUN_FRAME_BYTES);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                reserve_next_raw_frame(
                    &second_decoder,
                    &header,
                    0,
                    &sender,
                    &incoming,
                    &cancellation,
                    &mut heartbeat,
                ),
            )
            .await
            .is_err(),
            "second stream waits without incrementally consuming the frame budget"
        );
        drop(first);
        let second = reserve_next_raw_frame(
            &second_decoder,
            &header,
            0,
            &sender,
            &incoming,
            &cancellation,
            &mut heartbeat,
        )
        .await
        .expect("second reservation proceeds after the first releases");
        assert!(second.is_some());
    }
}
