//! Cursor Run stream adaptation to the sampler's normalized event contract.
//!
//! A parked bridge is kept only in process memory. This is what lets the next
//! Grok sampler request return host-executed MCP results on the same Cursor
//! Run stream; it never persists Cursor credentials or protocol state.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use async_stream::stream;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use prost::Message;
use reqwest::StatusCode;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use xai_grok_sampling_types::{
    AssistantItem, ConversationItem, ConversationRequest, ConversationResponse, SamplingError,
    StopReason, TokenUsage, ToolCall,
};

use crate::config::SamplerConfig;
use crate::cursor_proto::agent::v1::{
    AgentClientMessage, AskQuestionInteractionResponse, AskQuestionRejected, AskQuestionResult,
    BackgroundShellSpawnResult, ComputerUseError, ComputerUseResult, ConversationStateStructure,
    CreatePlanError, CreatePlanRequestResponse, CreatePlanResult, DeleteRejected, DeleteResult,
    DiagnosticsRejected, DiagnosticsResult, ExaFetchRequestResponse, ExaSearchRequestResponse,
    ExecClientMessage, FetchError, FetchResult, GetBlobResult, GrepError, GrepResult,
    InteractionResponse, ListMcpResourcesExecResult, ListMcpResourcesRejected, LsRejected,
    LsResult, McpArgs, McpResult, McpSuccess, McpTextContent, McpToolNotFound,
    McpToolResultContentItem, ReadMcpResourceExecResult, ReadMcpResourceRejected, ReadRejected,
    ReadResult, RecordScreenFailure, RecordScreenResult, RequestContext, RequestContextResult,
    RequestContextSuccess, SetBlobResult, ShellRejected, ShellResult, ShellStream,
    SwitchModeRequestResponse, WebSearchRequestResponse, WriteRejected, WriteResult,
    WriteShellStdinError, WriteShellStdinResult, agent_client_message, agent_server_message,
    ask_question_result, background_shell_spawn_result, computer_use_result, create_plan_result,
    delete_result, diagnostics_result, exa_fetch_request_response, exa_search_request_response,
    exec_client_message, exec_server_message, fetch_result, grep_result, interaction_query,
    interaction_response, interaction_update, kv_client_message, kv_server_message,
    list_mcp_resources_exec_result, ls_result, mcp_result, mcp_tool_result_content_item,
    read_mcp_resource_exec_result, read_result, record_screen_result, request_context_result,
    shell_result, shell_stream, switch_mode_request_response, web_search_request_response,
    write_result, write_shell_stdin_result,
};
use crate::cursor_request::{CursorModelRoute, CursorRunPayload, build_cursor_run_payload};
use crate::cursor_transport::{CursorRunConnection, CursorRunTransport, CursorTransportError};
use crate::events::{SamplingChannel, SamplingErrorInfo, SamplingEvent};
use crate::metrics::InferenceLatencyStats;
use crate::types::RequestId;

const MAX_CURSOR_BLOB_BYTES: usize = 16 * 1024 * 1024;
const MAX_CURSOR_BLOB_STORE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CURSOR_BLOB_COUNT: usize = 10_000;
const CURSOR_BLOB_ID_BYTES: usize = 32;
const MAX_CURSOR_TOOL_RESULT_BYTES: usize = 8 * 1024 * 1024;
const MAX_CURSOR_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_CURSOR_OUTPUT_CHUNKS: usize = 65_536;
const MAX_CURSOR_TOOL_CALLS: usize = 512;
const MAX_CURSOR_EXEC_ID_BYTES: usize = 256;
const MAX_CURSOR_TOOL_NAME_BYTES: usize = 128;
const PARKED_RUN_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_PARKED_RUNS: usize = 16;
const MAX_PARKED_STATE_BYTES: usize = 128 * 1024 * 1024;
const CURSOR_STATE_MEMORY_UNIT_BYTES: usize = 4 * 1024;
const CURSOR_PAYLOAD_BUILD_TIMEOUT: Duration = Duration::from_secs(10);
const MCP_BATCH_WINDOW: Duration = Duration::from_millis(100);

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CursorBridgeKey {
    provider: &'static str,
    session_id: String,
    model: String,
    generation: String,
}

#[derive(Clone, Debug)]
struct PendingExec {
    id: u32,
    exec_id: String,
    cursor_tool_call_id: String,
}

struct CursorRunState {
    connection: CursorRunConnection,
    blobs: HashMap<Vec<u8>, Vec<u8>>,
    advertised_tools: HashSet<String>,
    tool_definitions: Vec<crate::cursor_proto::agent::v1::McpToolDefinition>,
    pending_execs: HashMap<String, PendingExec>,
    generation: String,
    checkpoint: Option<ConversationStateStructure>,
    state_memory_permit: Option<OwnedSemaphorePermit>,
    state_memory_units: u32,
    last_used: Instant,
}

impl std::fmt::Debug for CursorRunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorRunState")
            .field("blob_count", &self.blobs.len())
            .field("advertised_tool_count", &self.advertised_tools.len())
            .field("pending_exec_count", &self.pending_execs.len())
            .field("generation", &self.generation)
            .field("has_checkpoint", &self.checkpoint.is_some())
            .finish_non_exhaustive()
    }
}

static PARKED_RUNS: OnceLock<Mutex<HashMap<CursorBridgeKey, CursorRunState>>> = OnceLock::new();
static CURSOR_STATE_MEMORY: OnceLock<std::sync::Arc<tokio::sync::Semaphore>> = OnceLock::new();
static CURSOR_PAYLOAD_BUILD_SLOT: OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
    OnceLock::new();

fn parked_runs() -> &'static Mutex<HashMap<CursorBridgeKey, CursorRunState>> {
    PARKED_RUNS.get_or_init(|| Mutex::new(HashMap::new()))
}

struct CursorConversationRotation {
    wire_id: String,
    succeeded: bool,
}

static CURSOR_CONVERSATION_ROTATIONS: OnceLock<Mutex<HashMap<String, CursorConversationRotation>>> =
    OnceLock::new();

fn cursor_conversation_rotations() -> &'static Mutex<HashMap<String, CursorConversationRotation>> {
    CURSOR_CONVERSATION_ROTATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Overlay a rotated Cursor conversation id after the previous id was poisoned.
/// Side calls pass a distinct `x_grok_conv_id` and keep that identity.
pub(crate) fn resolved_cursor_conversation_id(session_id: &str, conv_id: Option<&str>) -> String {
    let requested = conv_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(session_id);
    if requested != session_id {
        return requested.to_owned();
    }
    let rotations = cursor_conversation_rotations()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    rotations
        .get(session_id)
        .map(|rotation| rotation.wire_id.clone())
        .unwrap_or_else(|| session_id.to_owned())
}

/// Force a fresh Cursor conversation id for this Grok session and drop parked Runs.
/// Used after local compaction so the next Run is not merged into the uncompacted server history.
pub fn rotate_cursor_conversation(session_id: &str) -> String {
    cancel_cursor_session(session_id);
    let wire_id = uuid::Uuid::new_v4().to_string();
    let mut rotations = cursor_conversation_rotations()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    rotations.insert(
        session_id.to_owned(),
        CursorConversationRotation {
            wire_id: wire_id.clone(),
            succeeded: false,
        },
    );
    wire_id
}

fn note_cursor_stream_poison(session_id: &str, saw_output: bool, error: &CursorTransportError) {
    if saw_output || !matches!(error, CursorTransportError::ResourceExhausted) {
        return;
    }
    let rotations = cursor_conversation_rotations()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(current) = rotations.get(session_id)
        && !current.succeeded
    {
        return;
    }
    drop(rotations);
    rotate_cursor_conversation(session_id);
}

fn mark_cursor_conversation_healthy(session_id: &str) {
    let mut rotations = cursor_conversation_rotations()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(current) = rotations.get_mut(session_id) {
        current.succeeded = true;
    }
}

fn cursor_state_memory() -> std::sync::Arc<tokio::sync::Semaphore> {
    CURSOR_STATE_MEMORY
        .get_or_init(|| {
            std::sync::Arc::new(tokio::sync::Semaphore::new(
                (MAX_PARKED_STATE_BYTES / CURSOR_STATE_MEMORY_UNIT_BYTES) as u32 as usize,
            ))
        })
        .clone()
}

fn cursor_payload_build_slot() -> std::sync::Arc<tokio::sync::Semaphore> {
    CURSOR_PAYLOAD_BUILD_SLOT
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(1)))
        .clone()
}

/// Close paused Cursor streams owned by a Grok session after turn cancellation.
pub fn cancel_cursor_session(session_id: &str) -> usize {
    let mut registry = parked_runs()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let keys = registry
        .keys()
        .filter(|key| key.session_id == session_id)
        .cloned()
        .collect::<Vec<_>>();
    let removed = keys.len();
    for key in keys {
        registry.remove(&key);
    }
    removed
}

fn take_parked_run(key: &CursorBridgeKey, is_continuation: bool) -> Option<CursorRunState> {
    let mut registry = parked_runs()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    registry.retain(|_, parked| now.duration_since(parked.last_used) < PARKED_RUN_TTL);
    is_continuation.then(|| registry.remove(key)).flatten()
}

/// Prepare a fresh or resumed Cursor stream and translate it into sampler events.
///
/// A fresh Run resolves Cursor's current desktop credential. A continuation
/// stays on the already-authenticated Cursor stream, so token rotation or a
/// temporary local database read failure cannot break an in-flight tool turn.
/// The token itself is never placed in sampler config, tracing, or the parked
/// bridge registry.
pub(crate) async fn open_run_stream(
    config: &SamplerConfig,
    mut request: ConversationRequest,
    request_id: RequestId,
    idle_timeout: Duration,
    cancellation: CancellationToken,
) -> Result<BoxStream<'static, SamplingEvent>, SamplingError> {
    if request.model.is_none() {
        request.model = Some(config.model.clone());
    }
    request.temperature = None;
    request.top_p = None;
    request.reasoning_effort = None;
    request.json_schema = None;
    request.max_output_tokens =
        effective_output_limit(request.max_output_tokens, config.max_completion_tokens);
    let request_max_output_tokens = request.max_output_tokens;

    let session_id = request
        .x_grok_session_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            SamplingError::EventStreamError(
                "Cursor requests need a stable Grok session identifier.".to_owned(),
            )
        })?;
    request.x_grok_conv_id = Some(resolved_cursor_conversation_id(
        session_id,
        request.x_grok_conv_id.as_deref(),
    ));
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| config.model.clone());
    if !valid_cursor_identifier(session_id, MAX_CURSOR_EXEC_ID_BYTES)
        || !valid_cursor_identifier(&model, 512)
    {
        return Err(SamplingError::EventStreamError(
            "Cursor requests need bounded session and model identifiers.".to_owned(),
        ));
    }
    let tool_results = trailing_tool_results(&request)?;
    #[cfg(any(test, feature = "test-support"))]
    let test_transport = crate::cursor_transport::test_transport(session_id);
    let access_token = if tool_results.is_none() {
        #[cfg(any(test, feature = "test-support"))]
        let test_access_token = test_transport
            .as_ref()
            .map(|override_| override_.access_token.clone());
        #[cfg(not(any(test, feature = "test-support")))]
        let test_access_token: Option<String> = None;
        if let Some(access_token) = test_access_token {
            Some(Zeroizing::new(access_token))
        } else {
            let resolver = config
                .bearer_resolver
                .as_deref()
                .ok_or_else(|| cursor_missing_auth(None))?;
            resolver.prepare_for_send().await;
            Some(Zeroizing::new(
                resolver
                    .current_bearer()
                    .ok_or_else(|| cursor_missing_auth(Some(resolver)))?,
            ))
        }
    } else {
        None
    };
    let response_model = CursorModelRoute::from_model_string(&model)
        .map_err(|error| SamplingError::EventStreamError(error.to_string()))?
        .map_or_else(|| model.clone(), |route| route.model_id);
    let generation = match tool_results.as_ref() {
        Some(results) => continuation_generation(results)?,
        None => uuid::Uuid::new_v4().simple().to_string(),
    };
    let key = CursorBridgeKey {
        provider: "cursor",
        session_id: session_id.to_owned(),
        model: model.clone(),
        generation: generation.clone(),
    };

    // Keep sibling bridges independent. Auxiliary calls may share a Grok
    // session and model while a permission-gated turn is parked.
    let mut state = take_parked_run(&key, tool_results.is_some());

    match (state.as_mut(), tool_results.as_ref()) {
        (Some(parked), Some(results)) => {
            validate_pending_results(parked, results)?;
            send_tool_results(parked, results, &cancellation).await?;
            parked.pending_execs.clear();
            parked.last_used = Instant::now();
        }
        (Some(_), None) => {
            return Err(SamplingError::EventStreamError(
                "Cursor continuation state did not match the request's tool results.".to_owned(),
            ));
        }
        (None, Some(_)) => {
            return Err(SamplingError::EventStreamError(
                "Cursor's paused tool stream is no longer available; start a new turn.".to_owned(),
            ));
        }
        (None, None) => {}
    }

    if state.is_none() {
        // Payload construction can temporarily hold both the encoded Run frame
        // and cloned prompt/tool data. Serialize that bounded build, then
        // account for the resulting payload before any transport wait or lazy
        // stream return can retain it.
        let payload_build_slot = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                return Err(SamplingError::EventStreamError("Cursor request was cancelled.".to_owned()));
            }
            result = tokio::time::timeout(
                CURSOR_PAYLOAD_BUILD_TIMEOUT,
                cursor_payload_build_slot().acquire_owned(),
            ) => match result {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) | Err(_) => {
                    return Err(SamplingError::EventStreamError(
                        "Cursor could not reserve request-building capacity; retry the request."
                            .to_owned(),
                    ));
                }
            }
        };
        let payload_builder = tokio::task::spawn_blocking(move || {
            (build_cursor_run_payload(request), payload_build_slot)
        });
        let (payload_result, payload_build_slot) = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                // The bounded blocking build may finish in the background, but
                // it retains the sole build slot until it hands off the result.
                return Err(SamplingError::EventStreamError("Cursor request was cancelled.".to_owned()));
            }
            result = payload_builder => {
                result
                    .map_err(|_| SamplingError::EventStreamError(
                        "Cursor request could not be prepared.".to_owned(),
                    ))?
            }
        };
        let payload =
            payload_result.map_err(|error| SamplingError::EventStreamError(error.to_string()))?;
        let payload_memory_bytes = cursor_payload_state_bytes(&key, &payload, &generation);
        let (state_memory_permit, state_memory_units) =
            reserve_cursor_state_memory(payload_memory_bytes).ok_or_else(|| {
                SamplingError::EventStreamError(
                    "Cursor stream state exceeded Grok Build's local memory budget.".to_owned(),
                )
            })?;
        drop(payload_build_slot);
        #[cfg(any(test, feature = "test-support"))]
        let transport = test_transport
            .as_ref()
            .map(|override_| override_.transport.clone())
            .map_or_else(CursorRunTransport::new, Ok)
            .map_err(transport_error)?;
        #[cfg(not(any(test, feature = "test-support")))]
        let transport = CursorRunTransport::new().map_err(transport_error)?;
        let CursorRunPayload {
            initial_frame,
            blobs,
            advertised_tools,
            advertised_tool_definitions,
            ..
        } = payload;
        let connection = transport
            .open(
                initial_frame,
                access_token
                    .as_deref()
                    .expect("fresh Cursor Run has a credential")
                    .as_str(),
                cancellation.clone(),
            )
            .await
            .map_err(transport_error)?;
        state = Some(CursorRunState {
            connection,
            blobs,
            advertised_tools,
            tool_definitions: advertised_tool_definitions,
            pending_execs: HashMap::new(),
            generation,
            checkpoint: None,
            state_memory_permit: Some(state_memory_permit),
            state_memory_units,
            last_used: Instant::now(),
        });
    }

    let mut state = state.expect("fresh or resumed Cursor run state");
    if !refresh_cursor_state_memory(&key, &mut state) {
        return Err(SamplingError::EventStreamError(
            "Cursor stream state exceeded Grok Build's local memory budget.".to_owned(),
        ));
    }
    Ok(run_state_stream(
        state,
        key,
        response_model,
        request_id,
        idle_timeout,
        request_max_output_tokens,
        cancellation,
    ))
}

fn run_state_stream(
    state: CursorRunState,
    key: CursorBridgeKey,
    model: String,
    request_id: RequestId,
    idle_timeout: Duration,
    max_output_tokens: Option<u32>,
    cancellation: CancellationToken,
) -> BoxStream<'static, SamplingEvent> {
    let stream = stream! {
        let start = Instant::now();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        let mut timestamps = Vec::new();
        let mut chunk_index = 0_u64;
        let mut first_token_emitted = false;
        let mut output_tokens = 0_u32;
        let mut message_chunks_emitted = 0_u64;
        let mut output_bytes = 0_usize;
        let output_byte_limit = max_output_tokens
            .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX).saturating_mul(8))
            .unwrap_or(MAX_CURSOR_OUTPUT_BYTES)
            .min(MAX_CURSOR_OUTPUT_BYTES);
        let mut last_progress = Instant::now();
        let mut batch_deadline = None;
        let mut state = state;

        if !refresh_cursor_state_memory(&key, &mut state) {
            yield failed_event(&request_id, SamplingError::EventStreamError(
                "Cursor stream state exceeded Grok Build's local memory budget.".to_owned()
            ));
            return;
        }

        yield SamplingEvent::StreamStarted {
            request_id: request_id.clone(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        };

        loop {
            let progress_remaining = idle_timeout.saturating_sub(last_progress.elapsed());
            let batch_remaining = batch_deadline
                .map(|deadline: Instant| deadline.saturating_duration_since(Instant::now()));
            let remaining = batch_remaining.map_or(progress_remaining, |batch| batch.min(progress_remaining));
            if remaining.is_zero() && !state.pending_execs.is_empty() {
                let used_tokens = checkpoint_used_tokens(&state);
                if !park_cursor_run(&key, state) {
                    yield failed_event(&request_id, SamplingError::EventStreamError("Cursor tool continuation could not be retained within the local memory limit.".to_owned()));
                    return;
                }
                let metrics = InferenceLatencyStats::from_timestamps(start, &timestamps, Instant::now());
                let response = build_response(
                    content,
                    reasoning,
                    tool_calls,
                    model,
                    Some(StopReason::ToolCalls),
                    output_tokens,
                    used_tokens,
                    message_chunks_emitted,
                );
                yield SamplingEvent::Completed {
                    request_id: request_id.clone(),
                    response: Box::new(response),
                    metrics,
                };
                return;
            }

            let wait = remaining;
            let incoming = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    state.connection.cancel();
                    yield failed_event(&request_id, SamplingError::EventStreamError("Cursor request was cancelled.".to_owned()));
                    return;
                }
                result = tokio::time::timeout(wait, state.connection.next_message()) => result,
            };

            let server_message = match incoming {
                Err(_elapsed) if !state.pending_execs.is_empty() => {
                    let used_tokens = checkpoint_used_tokens(&state);
                    if !park_cursor_run(&key, state) {
                        yield failed_event(&request_id, SamplingError::EventStreamError("Cursor tool continuation could not be retained within the local memory limit.".to_owned()));
                        return;
                    }
                    let metrics = InferenceLatencyStats::from_timestamps(start, &timestamps, Instant::now());
                    let response = build_response(
                        content,
                        reasoning,
                        tool_calls,
                        model,
                        Some(StopReason::ToolCalls),
                        output_tokens,
                        used_tokens,
                        message_chunks_emitted,
                    );
                    yield SamplingEvent::Completed {
                        request_id: request_id.clone(),
                        response: Box::new(response),
                        metrics,
                    };
                    return;
                }
                Err(_elapsed) => {
                    yield failed_event(&request_id, SamplingError::IdleTimeout { elapsed_secs: idle_timeout.as_secs() });
                    return;
                }
                Ok(Err(error)) => {
                    note_cursor_stream_poison(
                        &key.session_id,
                        output_tokens > 0 || first_token_emitted,
                        &error,
                    );
                    yield failed_event(&request_id, transport_error(error));
                    return;
                }
                Ok(Ok(None)) if !state.pending_execs.is_empty() => {
                    yield failed_event(&request_id, SamplingError::EventStreamError(
                        "Cursor closed the Run stream before host tool results could be returned.".to_owned()
                    ));
                    return;
                }
                Ok(Ok(None)) => {
                    yield failed_event(&request_id, SamplingError::EventStreamError(
                        "Cursor closed the Run stream without ending the turn.".to_owned()
                    ));
                    return;
                }
                Ok(Ok(Some(message))) => message,
            };

            let Some(message) = server_message.message.message else {
                yield failed_event(&request_id, SamplingError::EventStreamError(
                    "Cursor returned an empty server message.".to_owned()
                ));
                return;
            };

            match message {
                agent_server_message::Message::InteractionUpdate(update) => {
                    let Some(update_message) = update.message else {
                        yield failed_event(&request_id, SamplingError::EventStreamError("Cursor returned an unknown interaction update; the run was stopped.".to_owned()));
                        return;
                    };
                    match update_message {
                        interaction_update::Message::TextDelta(delta) if !delta.text.is_empty() => {
                            if timestamps.len() >= MAX_CURSOR_OUTPUT_CHUNKS {
                                yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local chunk limit.".to_owned()));
                                return;
                            }
                            if output_bytes.saturating_add(delta.text.len()) > output_byte_limit {
                                yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local output budget.".to_owned()));
                                return;
                            }
                            output_bytes += delta.text.len();
                            last_progress = Instant::now();
                            if !first_token_emitted {
                                first_token_emitted = true;
                                yield SamplingEvent::FirstToken { request_id: request_id.clone() };
                            }
                            chunk_index += 1;
                            message_chunks_emitted += 1;
                            timestamps.push(Instant::now());
                            content.push_str(&delta.text);
                            yield SamplingEvent::ChannelToken {
                                request_id: request_id.clone(),
                                channel: SamplingChannel::Text,
                                text: delta.text,
                                chunk_index,
                            };
                        }
                        interaction_update::Message::TextDelta(_) => {}
                        interaction_update::Message::ThinkingDelta(delta) if !delta.text.is_empty() => {
                            if timestamps.len() >= MAX_CURSOR_OUTPUT_CHUNKS {
                                yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local chunk limit.".to_owned()));
                                return;
                            }
                            if output_bytes.saturating_add(delta.text.len()) > output_byte_limit {
                                yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local output budget.".to_owned()));
                                return;
                            }
                            output_bytes += delta.text.len();
                            last_progress = Instant::now();
                            if !first_token_emitted {
                                first_token_emitted = true;
                                yield SamplingEvent::FirstToken { request_id: request_id.clone() };
                            }
                            chunk_index += 1;
                            timestamps.push(Instant::now());
                            reasoning.push_str(&delta.text);
                            yield SamplingEvent::ChannelToken {
                                request_id: request_id.clone(),
                                channel: SamplingChannel::Reasoning,
                                text: delta.text,
                                chunk_index,
                            };
                        }
                        interaction_update::Message::ThinkingDelta(_) => {}
                        interaction_update::Message::TokenDelta(delta) => {
                            let received_tokens = delta.tokens.max(0) as u32;
                            let updated_tokens = output_tokens.saturating_add(received_tokens);
                            if max_output_tokens.is_some_and(|limit| updated_tokens > limit) {
                                yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local token budget.".to_owned()));
                                return;
                            }
                            if received_tokens > 0 {
                                last_progress = Instant::now();
                            }
                            output_tokens = updated_tokens;
                        }
                        interaction_update::Message::TurnEnded(_) => {
                            if !state.pending_execs.is_empty() {
                                let used_tokens = checkpoint_used_tokens(&state);
                                if !park_cursor_run(&key, state) {
                                    yield failed_event(&request_id, SamplingError::EventStreamError("Cursor tool continuation could not be retained within the local memory limit.".to_owned()));
                                    return;
                                }
                                let metrics = InferenceLatencyStats::from_timestamps(start, &timestamps, Instant::now());
                                let response = build_response(
                                    content,
                                    reasoning,
                                    tool_calls,
                                    model,
                                    Some(StopReason::ToolCalls),
                                    output_tokens,
                                    used_tokens,
                                    message_chunks_emitted,
                                );
                                yield SamplingEvent::Completed {
                                    request_id: request_id.clone(),
                                    response: Box::new(response),
                                    metrics,
                                };
                                return;
                            }
                            break;
                        }
                        interaction_update::Message::Heartbeat(_)
                        | interaction_update::Message::ThinkingCompleted(_)
                        | interaction_update::Message::Summary(_)
                        | interaction_update::Message::SummaryStarted(_)
                        | interaction_update::Message::SummaryCompleted(_)
                        | interaction_update::Message::StepStarted(_)
                        | interaction_update::Message::StepCompleted(_) => {}
                        // Cursor-native tool updates are not part of the Grok
                        // tool lifecycle. Fail closed before executing anything.
                        interaction_update::Message::ToolCallStarted(_)
                        | interaction_update::Message::ToolCallCompleted(_)
                        | interaction_update::Message::ToolCallDelta(_)
                        | interaction_update::Message::PartialToolCall(_) => {
                            // These are informational lifecycle updates. Actual
                            // tool execution requests arrive as ExecServerMessage
                            // and are separately allowlisted or rejected below.
                            last_progress = Instant::now();
                        }
                        interaction_update::Message::UserMessageAppended(_)
                        | interaction_update::Message::ShellOutputDelta(_) => {
                            yield failed_event(&request_id, SamplingError::EventStreamError(
                                "Cursor requested an unsupported native agent action; the run was stopped.".to_owned()
                            ));
                            return;
                        }
                    }
                }
                agent_server_message::Message::KvServerMessage(kv) => {
                    if let Err(error) = handle_kv_message(&mut state, kv, &cancellation).await {
                        yield failed_event(&request_id, error);
                        return;
                    }
                }
                agent_server_message::Message::ExecServerMessage(exec) => {
                    match exec.message {
                        Some(exec_server_message::Message::RequestContextArgs(_)) => {
                            let context = RequestContext {
                                tools: state.tool_definitions.clone(),
                                ..Default::default()
                            };
                            let response = AgentClientMessage {
                                message: Some(agent_client_message::Message::ExecClientMessage(
                                    ExecClientMessage {
                                        id: exec.id,
                                        exec_id: exec.exec_id,
                                        message: Some(exec_client_message::Message::RequestContextResult(
                                            RequestContextResult {
                                                result: Some(request_context_result::Result::Success(
                                                    RequestContextSuccess { request_context: Some(context) },
                                                )),
                                            },
                                        )),
                                    },
                                )),
                            };
                            if let Err(error) = state.connection.sender().send_message_cancellable(&response, &cancellation).await {
                                yield failed_event(&request_id, transport_error(error));
                                return;
                            }
                        }
                        Some(exec_server_message::Message::McpArgs(args)) => {
                            match handle_mcp_call(
                                &mut state,
                                exec.id,
                                exec.exec_id,
                                args,
                                &cancellation,
                            ).await {
                                Ok(Some((tool_call, arguments))) => {
                                    if tool_calls.len() >= MAX_CURSOR_TOOL_CALLS {
                                        yield failed_event(&request_id, SamplingError::EventStreamError("Cursor returned too many tool calls.".to_owned()));
                                        return;
                                    }
                                    if output_bytes.saturating_add(arguments.len()) > output_byte_limit {
                                        yield failed_event(&request_id, SamplingError::EventStreamError("Cursor output exceeded Grok Build's local output budget.".to_owned()));
                                        return;
                                    }
                                    output_bytes += arguments.len();
                                    if !first_token_emitted {
                                        first_token_emitted = true;
                                        yield SamplingEvent::FirstToken { request_id: request_id.clone() };
                                    }
                                    chunk_index += 1;
                                    timestamps.push(Instant::now());
                                    yield SamplingEvent::ToolCallDelta {
                                        request_id: request_id.clone(),
                                        tool_index: u32::try_from(tool_calls.len()).unwrap_or(u32::MAX),
                                        id: Some(tool_call.id.to_string()),
                                        name: Some(tool_call.name.clone()),
                                        arguments_delta: Some(arguments),
                                    };
                                    last_progress = Instant::now();
                                    tool_calls.push(tool_call);
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield failed_event(&request_id, error);
                                    return;
                                }
                            }
                            if !state.pending_execs.is_empty() && batch_deadline.is_none() {
                                batch_deadline = Some(Instant::now() + MCP_BATCH_WINDOW);
                            }
                        }
                        None => {
                            yield failed_event(&request_id, SamplingError::EventStreamError(
                                "Cursor sent an empty native tool request; the run was stopped.".to_owned()
                            ));
                            return;
                        }
                        // No Cursor-native filesystem, shell, web, computer,
                        // or interaction executor is enabled by this adapter.
                        Some(unsupported) => {
                            if let Some(response) = reject_native_exec(
                                exec.id,
                                exec.exec_id.clone(),
                                &unsupported,
                            ) {
                                if let Err(error) = state
                                    .connection
                                    .sender()
                                    .send_message_cancellable(&response, &cancellation)
                                    .await
                                {
                                    yield failed_event(&request_id, transport_error(error));
                                    return;
                                }
                            } else {
                                yield failed_event(&request_id, SamplingError::EventStreamError(
                                    "Cursor requested an unsupported native tool; the run was stopped.".to_owned()
                                ));
                                return;
                            }
                        }
                    }
                }
                agent_server_message::Message::ConversationCheckpointUpdate(checkpoint) => {
                    state.checkpoint = Some(checkpoint);
                    state.last_used = Instant::now();
                    last_progress = Instant::now();
                }
                agent_server_message::Message::InteractionQuery(query) => {
                    if let Some(response) = reject_interaction_query(query.id, query.query.as_ref()) {
                        if let Err(error) = state.connection.sender()
                            .send_message_cancellable(&response, &cancellation).await
                        {
                            yield failed_event(&request_id, transport_error(error));
                            return;
                        }
                    } else {
                        yield failed_event(&request_id, SamplingError::EventStreamError(
                            "Cursor requested an unsupported native interaction; the run was stopped.".to_owned()
                        ));
                        return;
                    }
                }
                agent_server_message::Message::ExecServerControlMessage(_) => {
                    yield failed_event(&request_id, SamplingError::EventStreamError(
                        "Cursor returned an unsupported execution control message.".to_owned()
                    ));
                    return;
                }
            }
            if !refresh_cursor_state_memory(&key, &mut state) {
                yield failed_event(&request_id, SamplingError::EventStreamError(
                    "Cursor stream state exceeded Grok Build's local memory budget.".to_owned()
                ));
                return;
            }
        }

        let metrics = InferenceLatencyStats::from_timestamps(start, &timestamps, Instant::now());
        let used_tokens = checkpoint_used_tokens(&state);
        mark_cursor_conversation_healthy(&key.session_id);
        let response = build_response(
            content,
            reasoning,
            tool_calls,
            model,
            Some(StopReason::Stop),
            output_tokens,
            used_tokens,
            message_chunks_emitted,
        );
        yield SamplingEvent::Completed {
            request_id: request_id.clone(),
            response: Box::new(response),
            metrics,
        };
    };
    stream.boxed()
}

fn cursor_missing_auth(resolver: Option<&dyn crate::config::BearerResolver>) -> SamplingError {
    let message = resolver
        .and_then(crate::config::BearerResolver::last_error_message)
        .unwrap_or_else(|| {
            "No signed-in Cursor Desktop session was found. Sign in to Cursor Desktop and retry."
                .to_owned()
        });
    SamplingError::Auth {
        message,
        credential: xai_grok_sampling_types::SentCredential::Missing,
    }
}

fn effective_output_limit(request_limit: Option<u32>, config_limit: Option<u32>) -> Option<u32> {
    request_limit.or(config_limit)
}

fn transport_error(error: CursorTransportError) -> SamplingError {
    match error {
        CursorTransportError::MissingCredential => cursor_missing_auth(None),
        CursorTransportError::Unauthorized => SamplingError::Auth {
            message: error.to_string(),
            credential: xai_grok_sampling_types::SentCredential::Sent,
        },
        CursorTransportError::HttpStatus(code) => SamplingError::Api {
            status: StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY),
            message: error.to_string(),
            model_metadata: None,
            retry_after_secs: None,
            should_retry: Some(false),
            error_code: None,
        },
        CursorTransportError::ResourceExhausted => SamplingError::EventStreamError(
            "Connect error resource_exhausted: Cursor rejected this conversation.".to_owned(),
        ),
        _ => SamplingError::EventStreamError(error.to_string()),
    }
}

fn failed_event(request_id: &RequestId, error: SamplingError) -> SamplingEvent {
    SamplingEvent::Failed {
        request_id: request_id.clone(),
        error: SamplingErrorInfo::from(&error),
    }
}

fn trailing_tool_results(
    request: &ConversationRequest,
) -> Result<Option<Vec<xai_grok_sampling_types::ToolResultItem>>, SamplingError> {
    let Some(last) = request.items.last() else {
        return Ok(None);
    };
    if !matches!(last, ConversationItem::ToolResult(_)) {
        return Ok(None);
    }
    let mut results = Vec::new();
    for item in request.items.iter().rev() {
        match item {
            ConversationItem::ToolResult(result) => {
                if !result.images.is_empty() {
                    return Err(SamplingError::EventStreamError(
                        "Cursor tool continuation does not yet support image results.".to_owned(),
                    ));
                }
                results.push(result.clone());
            }
            ConversationItem::System(_) => continue,
            _ => break,
        }
    }
    results.reverse();
    let mut seen = HashSet::new();
    if results
        .iter()
        .any(|result| !seen.insert(result.tool_call_id.clone()))
    {
        return Err(SamplingError::EventStreamError(
            "Cursor tool continuation contains duplicate tool result IDs.".to_owned(),
        ));
    }
    Ok(Some(results))
}

fn continuation_generation(
    results: &[xai_grok_sampling_types::ToolResultItem],
) -> Result<String, SamplingError> {
    let mut generation = None;
    for result in results {
        let Some((current, cursor_id)) = result
            .tool_call_id
            .strip_prefix("cursor-")
            .and_then(|value| value.split_once('-'))
        else {
            return Err(SamplingError::EventStreamError(
                "Cursor tool result has no provider-generation identity.".to_owned(),
            ));
        };
        if current.is_empty()
            || current.len() > 64
            || !current.bytes().all(|byte| byte.is_ascii_alphanumeric())
            || !valid_cursor_identifier(cursor_id, MAX_CURSOR_EXEC_ID_BYTES)
            || result.tool_call_id.len() > 512
            || generation.as_deref().is_some_and(|prior| prior != current)
        {
            return Err(SamplingError::EventStreamError(
                "Cursor tool results span multiple paused Run generations.".to_owned(),
            ));
        }
        generation = Some(current.to_owned());
    }
    generation.ok_or_else(|| {
        SamplingError::EventStreamError(
            "Cursor tool continuation contains no matching Run generation.".to_owned(),
        )
    })
}

fn validate_pending_results(
    parked: &CursorRunState,
    results: &[xai_grok_sampling_types::ToolResultItem],
) -> Result<(), SamplingError> {
    let actual: HashSet<&str> = results
        .iter()
        .map(|result| result.tool_call_id.as_str())
        .collect();
    let expected: HashSet<&str> = parked.pending_execs.keys().map(String::as_str).collect();
    if actual != expected || actual.len() != results.len() {
        return Err(SamplingError::EventStreamError(
            "Cursor tool results do not match the calls pending on the paused Run stream."
                .to_owned(),
        ));
    }
    Ok(())
}

async fn send_tool_results(
    state: &mut CursorRunState,
    results: &[xai_grok_sampling_types::ToolResultItem],
    cancellation: &CancellationToken,
) -> Result<(), SamplingError> {
    for result in results {
        if result.content.len() > MAX_CURSOR_TOOL_RESULT_BYTES {
            return Err(SamplingError::EventStreamError(
                "Cursor tool result exceeds the supported size limit.".to_owned(),
            ));
        }
        let pending = state
            .pending_execs
            .get(&result.tool_call_id)
            .ok_or_else(|| {
                SamplingError::EventStreamError(
                    "Cursor tool result does not match the paused Run stream.".to_owned(),
                )
            })?;
        let response = AgentClientMessage {
            message: Some(agent_client_message::Message::ExecClientMessage(
                ExecClientMessage {
                    id: pending.id,
                    exec_id: pending.exec_id.clone(),
                    message: Some(exec_client_message::Message::McpResult(McpResult {
                        result: Some(mcp_result::Result::Success(McpSuccess {
                            content: vec![McpToolResultContentItem {
                                content: Some(mcp_tool_result_content_item::Content::Text(
                                    McpTextContent {
                                        text: result.content.to_string(),
                                        output_location: None,
                                    },
                                )),
                            }],
                            is_error: result.is_error,
                        })),
                    })),
                },
            )),
        };
        state
            .connection
            .sender()
            .send_message_cancellable(&response, cancellation)
            .await
            .map_err(transport_error)?;
    }
    Ok(())
}

async fn handle_kv_message(
    state: &mut CursorRunState,
    message: crate::cursor_proto::agent::v1::KvServerMessage,
    cancellation: &CancellationToken,
) -> Result<(), SamplingError> {
    let response = match message.message {
        Some(kv_server_message::Message::GetBlobArgs(args)) => {
            if args.blob_id.len() != CURSOR_BLOB_ID_BYTES {
                return Err(SamplingError::EventStreamError(
                    "Cursor requested conversation data with an invalid content identifier."
                        .to_owned(),
                ));
            }
            let Some(blob) = state.blobs.get(&args.blob_id) else {
                return Err(SamplingError::EventStreamError(
                    "Cursor requested conversation data that is no longer available.".to_owned(),
                ));
            };
            AgentClientMessage {
                message: Some(agent_client_message::Message::KvClientMessage(
                    crate::cursor_proto::agent::v1::KvClientMessage {
                        id: message.id,
                        message: Some(kv_client_message::Message::GetBlobResult(GetBlobResult {
                            blob_data: Some(blob.clone()),
                        })),
                    },
                )),
            }
        }
        Some(kv_server_message::Message::SetBlobArgs(args)) => {
            if args.blob_id.len() != CURSOR_BLOB_ID_BYTES {
                return Err(SamplingError::EventStreamError(
                    "Cursor stored conversation data with an invalid content identifier."
                        .to_owned(),
                ));
            }
            let total_bytes: usize = state
                .blobs
                .iter()
                .map(|(key, value)| key.len().saturating_add(value.len()))
                .sum();
            let new_size = if let Some(previous) = state.blobs.get(&args.blob_id) {
                total_bytes
                    .saturating_sub(args.blob_id.len())
                    .saturating_sub(previous.len())
            } else {
                total_bytes
            }
            .saturating_add(args.blob_id.len())
            .saturating_add(args.blob_data.len());
            if args.blob_data.len() > MAX_CURSOR_BLOB_BYTES
                || args.blob_data.len() > MAX_CURSOR_BLOB_STORE_BYTES
                || new_size > MAX_CURSOR_BLOB_STORE_BYTES
                || (!state.blobs.contains_key(&args.blob_id)
                    && state.blobs.len() >= MAX_CURSOR_BLOB_COUNT)
            {
                return Err(SamplingError::EventStreamError(
                    "Cursor conversation data exceeded the supported storage limit.".to_owned(),
                ));
            }
            state.blobs.insert(args.blob_id, args.blob_data);
            AgentClientMessage {
                message: Some(agent_client_message::Message::KvClientMessage(
                    crate::cursor_proto::agent::v1::KvClientMessage {
                        id: message.id,
                        message: Some(kv_client_message::Message::SetBlobResult(SetBlobResult {
                            error: None,
                        })),
                    },
                )),
            }
        }
        None => {
            return Err(SamplingError::EventStreamError(
                "Cursor returned an unsupported conversation-storage message.".to_owned(),
            ));
        }
    };
    state
        .connection
        .sender()
        .send_message_cancellable(&response, cancellation)
        .await
        .map_err(transport_error)
}

#[allow(clippy::too_many_arguments)]
async fn handle_mcp_call(
    state: &mut CursorRunState,
    exec_message_id: u32,
    exec_id: String,
    args: McpArgs,
    cancellation: &CancellationToken,
) -> Result<Option<(ToolCall, String)>, SamplingError> {
    let tool_name = if args.tool_name.is_empty() {
        args.name.clone()
    } else {
        args.tool_name.clone()
    };
    if exec_id.len() > MAX_CURSOR_EXEC_ID_BYTES
        || exec_id.chars().any(char::is_control)
        || !valid_cursor_identifier(&tool_name, MAX_CURSOR_TOOL_NAME_BYTES)
        || args.provider_identifier.len() > MAX_CURSOR_TOOL_NAME_BYTES
        || args.provider_identifier.chars().any(char::is_control)
    {
        return Err(SamplingError::EventStreamError(
            "Cursor returned an invalid tool request identifier.".to_owned(),
        ));
    }
    let (cursor_tool_call_id, local_tool_call_id) = prepare_cursor_tool_call_id(args.tool_call_id)?;
    if args.provider_identifier != "grok" || !state.advertised_tools.contains(&tool_name) {
        let response = AgentClientMessage {
            message: Some(agent_client_message::Message::ExecClientMessage(
                ExecClientMessage {
                    id: exec_message_id,
                    exec_id,
                    message: Some(exec_client_message::Message::McpResult(McpResult {
                        result: Some(mcp_result::Result::ToolNotFound(McpToolNotFound {
                            name: tool_name.clone(),
                            available_tools: state.advertised_tools.iter().cloned().collect(),
                        })),
                    })),
                },
            )),
        };
        state
            .connection
            .sender()
            .send_message_cancellable(&response, cancellation)
            .await
            .map_err(transport_error)?;
        return Ok(None);
    }
    if state.pending_execs.len() >= MAX_CURSOR_TOOL_CALLS
        || state
            .pending_execs
            .values()
            .any(|entry| entry.cursor_tool_call_id == cursor_tool_call_id)
    {
        return Err(SamplingError::EventStreamError(
            "Cursor returned a missing or duplicate tool call identifier.".to_owned(),
        ));
    }
    let decoded = decode_mcp_arguments(&args.args)?;
    let arguments = serde_json::to_string(&decoded).map_err(|_| {
        SamplingError::EventStreamError("Cursor returned invalid tool arguments.".to_owned())
    })?;
    let generation_id = format!("cursor-{}-{local_tool_call_id}", state.generation);
    if state.pending_execs.contains_key(&generation_id) {
        return Err(SamplingError::EventStreamError(
            "Cursor returned a duplicate tool call identifier.".to_owned(),
        ));
    }
    let tool_call = ToolCall {
        id: std::sync::Arc::<str>::from(generation_id.clone()),
        name: tool_name,
        arguments: std::sync::Arc::<str>::from(arguments.as_str()),
    };
    state.pending_execs.insert(
        generation_id.clone(),
        PendingExec {
            id: exec_message_id,
            exec_id,
            cursor_tool_call_id,
        },
    );
    Ok(Some((tool_call, arguments)))
}

fn prepare_cursor_tool_call_id(tool_call_id: String) -> Result<(String, String), SamplingError> {
    let cursor_tool_call_id = if tool_call_id.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        tool_call_id
    };
    if !valid_cursor_tool_call_identifier(&cursor_tool_call_id) {
        return Err(SamplingError::EventStreamError(
            "Cursor returned an invalid tool request identifier.".to_owned(),
        ));
    }
    let local_tool_call_id = if cursor_tool_call_id.contains('\n') {
        format!(
            "opaque-{}",
            blake3::hash(cursor_tool_call_id.as_bytes()).to_hex()
        )
    } else {
        cursor_tool_call_id.clone()
    };
    Ok((cursor_tool_call_id, local_tool_call_id))
}

fn valid_cursor_tool_call_identifier(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_CURSOR_EXEC_ID_BYTES {
        return false;
    }
    if !value.contains('\n') {
        return !value.chars().any(char::is_control);
    }
    value
        .split_once('\n')
        .is_some_and(|(call_id, provider_id)| {
            valid_cursor_composite_component(call_id, "call-")
                && valid_cursor_composite_component(provider_id, "fc_")
        })
}

fn valid_cursor_composite_component(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(|component| {
        !component.is_empty()
            && component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    })
}

fn decode_mcp_arguments(
    args: &HashMap<String, Vec<u8>>,
) -> Result<serde_json::Value, SamplingError> {
    if args.len() > 256 {
        return Err(SamplingError::EventStreamError(
            "Cursor returned too many tool arguments.".to_owned(),
        ));
    }
    let mut object = serde_json::Map::new();
    let mut total = 0_usize;
    for (key, bytes) in args {
        total = total.saturating_add(key.len()).saturating_add(bytes.len());
        if total > MAX_CURSOR_TOOL_RESULT_BYTES {
            return Err(SamplingError::EventStreamError(
                "Cursor tool arguments exceed the supported size limit.".to_owned(),
            ));
        }
        let value = match prost_types::Value::decode(bytes.as_slice()) {
            Ok(value) => proto_value_to_json(value),
            Err(_) => {
                serde_json::Value::String(String::from_utf8(bytes.clone()).map_err(|_| {
                    SamplingError::EventStreamError(
                        "Cursor returned invalid tool argument data.".to_owned(),
                    )
                })?)
            }
        };
        object.insert(key.clone(), value);
    }
    Ok(serde_json::Value::Object(object))
}

fn valid_cursor_identifier(value: &str, max_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn reject_native_exec(
    id: u32,
    exec_id: String,
    message: &exec_server_message::Message,
) -> Option<AgentClientMessage> {
    let reason = "Cursor-native tools are disabled; use a Grok Build-hosted tool.".to_owned();
    let response = match message {
        exec_server_message::Message::ReadArgs(args) => {
            exec_client_message::Message::ReadResult(ReadResult {
                result: Some(read_result::Result::Rejected(ReadRejected {
                    path: args.path.clone(),
                    reason,
                })),
            })
        }
        exec_server_message::Message::LsArgs(args) => {
            exec_client_message::Message::LsResult(LsResult {
                result: Some(ls_result::Result::Rejected(LsRejected {
                    path: args.path.clone(),
                    reason,
                })),
            })
        }
        exec_server_message::Message::WriteArgs(args) => {
            exec_client_message::Message::WriteResult(WriteResult {
                result: Some(write_result::Result::Rejected(WriteRejected {
                    path: args.path.clone(),
                    reason,
                })),
            })
        }
        exec_server_message::Message::DeleteArgs(args) => {
            exec_client_message::Message::DeleteResult(DeleteResult {
                result: Some(delete_result::Result::Rejected(DeleteRejected {
                    path: args.path.clone(),
                    reason,
                })),
            })
        }
        exec_server_message::Message::GrepArgs(_) => {
            exec_client_message::Message::GrepResult(GrepResult {
                result: Some(grep_result::Result::Error(GrepError { error: reason })),
            })
        }
        exec_server_message::Message::DiagnosticsArgs(args) => {
            exec_client_message::Message::DiagnosticsResult(DiagnosticsResult {
                result: Some(diagnostics_result::Result::Rejected(DiagnosticsRejected {
                    path: args.path.clone(),
                    reason,
                })),
            })
        }
        exec_server_message::Message::ShellArgs(args) => {
            exec_client_message::Message::ShellResult(ShellResult {
                result: Some(shell_result::Result::Rejected(ShellRejected {
                    command: args.command.clone(),
                    working_directory: args.working_directory.clone(),
                    reason,
                    is_readonly: false,
                })),
                ..Default::default()
            })
        }
        exec_server_message::Message::ShellStreamArgs(args) => {
            exec_client_message::Message::ShellStream(ShellStream {
                event: Some(shell_stream::Event::Rejected(ShellRejected {
                    command: args.command.clone(),
                    working_directory: args.working_directory.clone(),
                    reason,
                    is_readonly: false,
                })),
            })
        }
        exec_server_message::Message::BackgroundShellSpawnArgs(args) => {
            exec_client_message::Message::BackgroundShellSpawnResult(BackgroundShellSpawnResult {
                result: Some(background_shell_spawn_result::Result::Rejected(
                    ShellRejected {
                        command: args.command.clone(),
                        working_directory: args.working_directory.clone(),
                        reason,
                        is_readonly: false,
                    },
                )),
            })
        }
        exec_server_message::Message::ListMcpResourcesExecArgs(_) => {
            exec_client_message::Message::ListMcpResourcesExecResult(ListMcpResourcesExecResult {
                result: Some(list_mcp_resources_exec_result::Result::Rejected(
                    ListMcpResourcesRejected { reason },
                )),
            })
        }
        exec_server_message::Message::ReadMcpResourceExecArgs(args) => {
            exec_client_message::Message::ReadMcpResourceExecResult(ReadMcpResourceExecResult {
                result: Some(read_mcp_resource_exec_result::Result::Rejected(
                    ReadMcpResourceRejected {
                        uri: args.uri.clone(),
                        reason,
                    },
                )),
            })
        }
        exec_server_message::Message::FetchArgs(args) => {
            exec_client_message::Message::FetchResult(FetchResult {
                result: Some(fetch_result::Result::Error(FetchError {
                    url: args.url.clone(),
                    error: reason,
                })),
            })
        }
        exec_server_message::Message::RecordScreenArgs(_) => {
            exec_client_message::Message::RecordScreenResult(RecordScreenResult {
                result: Some(record_screen_result::Result::Failure(RecordScreenFailure {
                    error: reason,
                })),
            })
        }
        exec_server_message::Message::ComputerUseArgs(args) => {
            exec_client_message::Message::ComputerUseResult(ComputerUseResult {
                result: Some(computer_use_result::Result::Error(ComputerUseError {
                    error: reason,
                    action_count: i32::try_from(args.actions.len()).unwrap_or(i32::MAX),
                    duration_ms: 0,
                    log: None,
                    screenshot: None,
                    screenshot_path: None,
                })),
            })
        }
        exec_server_message::Message::WriteShellStdinArgs(_) => {
            exec_client_message::Message::WriteShellStdinResult(WriteShellStdinResult {
                result: Some(write_shell_stdin_result::Result::Error(
                    WriteShellStdinError { error: reason },
                )),
            })
        }
        exec_server_message::Message::RequestContextArgs(_)
        | exec_server_message::Message::McpArgs(_) => return None,
    };
    Some(AgentClientMessage {
        message: Some(agent_client_message::Message::ExecClientMessage(
            ExecClientMessage {
                id,
                exec_id,
                message: Some(response),
            },
        )),
    })
}

fn reject_interaction_query(
    id: u32,
    query: Option<&interaction_query::Query>,
) -> Option<AgentClientMessage> {
    let reason = "Cursor-hosted interactions are disabled by Grok Build.".to_owned();
    let result = match query? {
        interaction_query::Query::WebSearchRequestQuery(_) => {
            interaction_response::Result::WebSearchRequestResponse(WebSearchRequestResponse {
                result: Some(web_search_request_response::Result::Rejected(
                    crate::cursor_proto::agent::v1::WebSearchRequestResponseRejected { reason },
                )),
            })
        }
        interaction_query::Query::AskQuestionInteractionQuery(_) => {
            interaction_response::Result::AskQuestionInteractionResponse(
                AskQuestionInteractionResponse {
                    result: Some(AskQuestionResult {
                        result: Some(ask_question_result::Result::Rejected(AskQuestionRejected {
                            reason,
                        })),
                    }),
                },
            )
        }
        interaction_query::Query::SwitchModeRequestQuery(_) => {
            interaction_response::Result::SwitchModeRequestResponse(SwitchModeRequestResponse {
                result: Some(switch_mode_request_response::Result::Rejected(
                    crate::cursor_proto::agent::v1::SwitchModeRequestResponseRejected { reason },
                )),
            })
        }
        interaction_query::Query::ExaSearchRequestQuery(_) => {
            interaction_response::Result::ExaSearchRequestResponse(ExaSearchRequestResponse {
                result: Some(exa_search_request_response::Result::Rejected(
                    crate::cursor_proto::agent::v1::ExaSearchRequestResponseRejected { reason },
                )),
            })
        }
        interaction_query::Query::ExaFetchRequestQuery(_) => {
            interaction_response::Result::ExaFetchRequestResponse(ExaFetchRequestResponse {
                result: Some(exa_fetch_request_response::Result::Rejected(
                    crate::cursor_proto::agent::v1::ExaFetchRequestResponseRejected { reason },
                )),
            })
        }
        interaction_query::Query::CreatePlanRequestQuery(_) => {
            interaction_response::Result::CreatePlanRequestResponse(CreatePlanRequestResponse {
                result: Some(CreatePlanResult {
                    plan_uri: String::new(),
                    result: Some(create_plan_result::Result::Error(CreatePlanError {
                        error: reason,
                    })),
                }),
            })
        }
        interaction_query::Query::SetupVmEnvironmentArgs(_) => return None,
    };
    Some(AgentClientMessage {
        message: Some(agent_client_message::Message::InteractionResponse(
            InteractionResponse {
                id,
                result: Some(result),
            },
        )),
    })
}

fn proto_value_to_json(value: prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::NumberValue(number)) => json_number_from_f64(number),
        Some(Kind::StringValue(text)) => serde_json::Value::String(text),
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(value),
        Some(Kind::StructValue(object)) => serde_json::Value::Object(
            object
                .fields
                .into_iter()
                .map(|(key, value)| (key, proto_value_to_json(value)))
                .collect(),
        ),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.into_iter().map(proto_value_to_json).collect())
        }
    }
}

fn json_number_from_f64(number: f64) -> serde_json::Value {
    if !number.is_finite() {
        return serde_json::Value::Null;
    }
    let as_int = number as i64;
    if as_int as f64 == number {
        return serde_json::Value::Number(as_int.into());
    }
    serde_json::Number::from_f64(number)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn checkpoint_used_tokens(state: &CursorRunState) -> u32 {
    state
        .checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.token_details.as_ref())
        .map(|details| details.used_tokens)
        .filter(|used| *used > 0)
        .unwrap_or(0)
}

fn build_response(
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    model: String,
    stop_reason: Option<StopReason>,
    output_tokens: u32,
    used_tokens: u32,
    message_chunks_emitted: u64,
) -> ConversationResponse {
    let mut items = Vec::with_capacity(2);
    if !reasoning.is_empty() {
        items.push(ConversationItem::Reasoning(
            xai_grok_sampling_types::synthesized_reasoning_item(reasoning),
        ));
    }
    items.push(ConversationItem::Assistant(AssistantItem {
        content: std::sync::Arc::<str>::from(content),
        tool_calls,
        model_id: Some(model),
        model_fingerprint: None,
        reasoning_effort: None,
    }));
    ConversationResponse {
        items,
        stop_reason,
        usage: Some(TokenUsage {
            prompt_tokens: used_tokens,
            completion_tokens: output_tokens,
            reasoning_tokens: 0,
            total_tokens: if used_tokens > 0 {
                used_tokens
            } else {
                output_tokens
            },
            ..Default::default()
        }),
        cost_usd_ticks: None,
        message_chunks_emitted,
        doom_loop_signals: Vec::new(),
        stop_message: None,
        message_id: None,
        raw_stop_reason: None,
        stop_sequence: None,
    }
}

fn park_cursor_run(key: &CursorBridgeKey, mut state: CursorRunState) -> bool {
    state.last_used = Instant::now();
    if !refresh_cursor_state_memory(key, &mut state) {
        return false;
    }
    let state_bytes = parked_state_bytes(key, &state);
    if state_bytes > MAX_PARKED_STATE_BYTES {
        return false;
    }
    let generation = state.generation.clone();
    {
        let mut registry = parked_runs()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry
            .retain(|_, parked| Instant::now().duration_since(parked.last_used) < PARKED_RUN_TTL);
        while registry.len() >= MAX_PARKED_RUNS
            || registry
                .iter()
                .fold(0_usize, |total, (parked_key, parked)| {
                    total.saturating_add(parked_state_bytes(parked_key, parked))
                })
                .saturating_add(state_bytes)
                > MAX_PARKED_STATE_BYTES
        {
            let oldest = registry
                .iter()
                .filter(|(parked_key, _)| *parked_key != key)
                .min_by_key(|(_, parked)| parked.last_used)
                .map(|(parked_key, _)| parked_key.clone());
            let Some(oldest) = oldest else { return false };
            registry.remove(&oldest);
        }
        registry.insert(key.clone(), state);
    }
    let key = key.clone();
    tokio::spawn(async move {
        tokio::time::sleep(PARKED_RUN_TTL).await;
        let mut registry = parked_runs()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if registry.get(&key).is_some_and(|parked| {
            parked.generation == generation
                && Instant::now().duration_since(parked.last_used) >= PARKED_RUN_TTL
        }) {
            registry.remove(&key);
        }
    });
    true
}

fn cursor_payload_state_bytes(
    key: &CursorBridgeKey,
    payload: &CursorRunPayload,
    generation: &str,
) -> usize {
    let blob_bytes = payload.blobs.iter().fold(
        payload.blobs.capacity().saturating_mul(128),
        |total, (id, data)| {
            total
                .saturating_add(id.len())
                .saturating_add(data.capacity())
                .saturating_add(64)
        },
    );
    let tool_bytes = payload.advertised_tool_definitions.iter().fold(
        payload
            .advertised_tool_definitions
            .capacity()
            .saturating_mul(std::mem::size_of::<
                crate::cursor_proto::agent::v1::McpToolDefinition,
            >()),
        |total, tool| total.saturating_add(tool.encoded_len()).saturating_add(128),
    );
    let advertised_bytes = payload.advertised_tools.iter().fold(
        payload.advertised_tools.capacity().saturating_mul(64),
        |total, name| total.saturating_add(name.len()).saturating_add(32),
    );
    blob_bytes
        .saturating_add(tool_bytes)
        .saturating_add(advertised_bytes)
        .saturating_add(payload.initial_frame.capacity())
        .saturating_add(key.session_id.len())
        .saturating_add(key.model.len())
        .saturating_add(key.generation.len())
        .saturating_add(generation.len())
        .saturating_add(256)
}

fn reserve_cursor_state_memory(bytes: usize) -> Option<(OwnedSemaphorePermit, u32)> {
    reserve_state_memory_from(cursor_state_memory(), bytes)
}

fn reserve_state_memory_from(
    budget: std::sync::Arc<tokio::sync::Semaphore>,
    bytes: usize,
) -> Option<(OwnedSemaphorePermit, u32)> {
    let units = bytes
        .max(1)
        .saturating_add(CURSOR_STATE_MEMORY_UNIT_BYTES - 1)
        / CURSOR_STATE_MEMORY_UNIT_BYTES;
    let units = u32::try_from(units).ok()?;
    let permit = budget.try_acquire_many_owned(units).ok()?;
    Some((permit, units))
}

fn parked_state_bytes(key: &CursorBridgeKey, state: &CursorRunState) -> usize {
    let blob_bytes = state.blobs.iter().fold(
        state.blobs.capacity().saturating_mul(128),
        |total, (id, data)| {
            total
                .saturating_add(id.len())
                .saturating_add(data.capacity())
                .saturating_add(64)
        },
    );
    let tool_bytes = state.tool_definitions.iter().fold(
        state
            .tool_definitions
            .capacity()
            .saturating_mul(std::mem::size_of::<
                crate::cursor_proto::agent::v1::McpToolDefinition,
            >()),
        |total, tool| total.saturating_add(tool.encoded_len()).saturating_add(128),
    );
    let advertised_bytes = state.advertised_tools.iter().fold(
        state.advertised_tools.capacity().saturating_mul(64),
        |total, name| total.saturating_add(name.len()).saturating_add(32),
    );
    let pending_bytes = state.pending_execs.iter().fold(
        state.pending_execs.capacity().saturating_mul(128),
        |total, (key, pending)| {
            total
                .saturating_add(key.len())
                .saturating_add(pending.exec_id.len())
                .saturating_add(pending.cursor_tool_call_id.len())
                .saturating_add(64)
        },
    );
    blob_bytes
        .saturating_add(tool_bytes)
        .saturating_add(advertised_bytes)
        .saturating_add(pending_bytes)
        .saturating_add(state.checkpoint.as_ref().map_or(0, |checkpoint| {
            conversation_state_allocated_bytes(checkpoint, 0)
        }))
        .saturating_add(key.session_id.len())
        .saturating_add(key.model.len())
        .saturating_add(key.generation.len())
        .saturating_add(state.generation.len())
        .saturating_add(256)
}

fn byte_string_vec_bytes(values: &[Vec<u8>], capacity: usize) -> usize {
    values.iter().fold(
        capacity.saturating_mul(std::mem::size_of::<Vec<u8>>()),
        |total, value| total.saturating_add(value.capacity()),
    )
}

fn string_vec_bytes(values: &[String], capacity: usize) -> usize {
    values.iter().fold(
        capacity.saturating_mul(std::mem::size_of::<String>()),
        |total, value| total.saturating_add(value.capacity()),
    )
}

/// Estimate retained heap use from actual capacities and nested elements.
/// Prost's `encoded_len` is a wire-size measure and severely undercounts
/// repeated empty bytes/messages retained as separate Vec allocations.
fn conversation_state_allocated_bytes(state: &ConversationStateStructure, depth: usize) -> usize {
    use crate::cursor_proto::agent::v1::SubagentPersistedState;

    if depth > 32 {
        return MAX_PARKED_STATE_BYTES.saturating_add(1);
    }
    let subagents = state.subagent_states.iter().fold(
        state
            .subagent_states
            .capacity()
            .saturating_mul(std::mem::size_of::<(String, SubagentPersistedState)>() * 2),
        |total, (name, subagent)| {
            total.saturating_add(name.capacity()).saturating_add(
                subagent.conversation_state.as_ref().map_or(0, |nested| {
                    conversation_state_allocated_bytes(nested, depth + 1)
                }),
            )
        },
    );
    let file_states = state.file_states.iter().fold(
        state
            .file_states
            .capacity()
            .saturating_mul(std::mem::size_of::<(String, Vec<u8>)>() * 2),
        |total, (path, bytes)| {
            total
                .saturating_add(path.capacity())
                .saturating_add(bytes.capacity())
        },
    );
    let file_states_v2 = state.file_states_v2.iter().fold(
        state.file_states_v2.capacity().saturating_mul(
            std::mem::size_of::<(String, crate::cursor_proto::agent::v1::FileStateStructure)>() * 2,
        ),
        |total, (path, value)| {
            total
                .saturating_add(path.capacity())
                .saturating_add(value.content.as_ref().map_or(0, Vec::capacity))
                .saturating_add(value.initial_content.as_ref().map_or(0, Vec::capacity))
        },
    );
    std::mem::size_of::<ConversationStateStructure>()
        .saturating_add(byte_string_vec_bytes(
            &state.turns_old,
            state.turns_old.capacity(),
        ))
        .saturating_add(byte_string_vec_bytes(
            &state.root_prompt_messages_json,
            state.root_prompt_messages_json.capacity(),
        ))
        .saturating_add(byte_string_vec_bytes(&state.turns, state.turns.capacity()))
        .saturating_add(byte_string_vec_bytes(&state.todos, state.todos.capacity()))
        .saturating_add(string_vec_bytes(
            &state.pending_tool_calls,
            state.pending_tool_calls.capacity(),
        ))
        .saturating_add(string_vec_bytes(
            &state.previous_workspace_uris,
            state.previous_workspace_uris.capacity(),
        ))
        .saturating_add(string_vec_bytes(
            &state.read_paths,
            state.read_paths.capacity(),
        ))
        .saturating_add(byte_string_vec_bytes(
            &state.summary_archives,
            state.summary_archives.capacity(),
        ))
        .saturating_add(
            state
                .turn_timings
                .capacity()
                .saturating_mul(std::mem::size_of::<
                    crate::cursor_proto::agent::v1::StepTiming,
                >()),
        )
        .saturating_add(file_states)
        .saturating_add(file_states_v2)
        .saturating_add(subagents)
        .saturating_add(state.summary.as_ref().map_or(0, Vec::capacity))
        .saturating_add(state.plan.as_ref().map_or(0, Vec::capacity))
        .saturating_add(state.summary_archive.as_ref().map_or(0, Vec::capacity))
        .saturating_add(state.extra_state.capacity())
        .saturating_add(state.client_name.capacity())
        .saturating_add(
            state
                .conversation_started_time_zone
                .as_ref()
                .map_or(0, String::capacity),
        )
}

fn refresh_cursor_state_memory(key: &CursorBridgeKey, state: &mut CursorRunState) -> bool {
    let bytes = parked_state_bytes(key, state);
    if bytes > MAX_PARKED_STATE_BYTES {
        return false;
    }
    let units = bytes
        .max(1)
        .saturating_add(CURSOR_STATE_MEMORY_UNIT_BYTES - 1)
        / CURSOR_STATE_MEMORY_UNIT_BYTES;
    let Ok(units) = u32::try_from(units) else {
        return false;
    };
    if units > state.state_memory_units {
        let additional = units - state.state_memory_units;
        let Ok(permit) = cursor_state_memory().try_acquire_many_owned(additional) else {
            return false;
        };
        if let Some(existing) = state.state_memory_permit.as_mut() {
            existing.merge(permit);
        } else {
            state.state_memory_permit = Some(permit);
        }
    } else if units < state.state_memory_units {
        if let Some(permit) = state.state_memory_permit.as_mut() {
            let released = (state.state_memory_units - units) as usize;
            let Some(excess) = permit.split(released) else {
                return false;
            };
            drop(excess);
        }
    }
    state.state_memory_units = units;
    true
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, Response};
    use axum::routing::post;
    use bytes::Bytes;
    use futures_util::{Stream, StreamExt};
    use prost::Message;
    use rcgen::{CertificateParams, KeyPair};
    use rustls::pki_types::PrivateKeyDer;
    use tokio::sync::Semaphore;

    use super::*;
    use crate::cursor_connect::{ConnectFrame, ConnectFrameDecoder, encode_connect_frame};
    use crate::cursor_proto::agent::v1::{
        AgentClientMessage, AgentServerMessage, ExecServerMessage, InteractionUpdate, McpArgs,
        RequestContextArgs, TurnEndedUpdate, agent_client_message, agent_server_message,
        exec_server_message, interaction_update,
    };
    use xai_grok_sampling_types::{ContentPart, ToolSpec, conversation::UserItem};

    const RUN_RPC: &str = "/agent.v1.AgentService/Run";

    #[tokio::test]
    async fn session_cancellation_drops_parked_cursor_connections() {
        let session_id = format!("cursor-cancel-{}", uuid::Uuid::new_v4());
        let cancellation = CancellationToken::new();
        let key = CursorBridgeKey {
            provider: "cursor",
            session_id: session_id.clone(),
            model: "test-model".to_owned(),
            generation: "test-generation".to_owned(),
        };
        let state = CursorRunState {
            connection: CursorRunConnection::test_connection(cancellation.clone()),
            blobs: HashMap::new(),
            advertised_tools: HashSet::new(),
            tool_definitions: Vec::new(),
            pending_execs: HashMap::new(),
            generation: key.generation.clone(),
            checkpoint: None,
            state_memory_permit: None,
            state_memory_units: 0,
            last_used: Instant::now(),
        };

        assert!(park_cursor_run(&key, state));
        assert_eq!(cancel_cursor_session(&session_id), 1);
        assert!(cancellation.is_cancelled());
        assert_eq!(cancel_cursor_session(&session_id), 0);
    }

    #[tokio::test]
    async fn fresh_run_lookup_preserves_sibling_parked_generations() {
        let session_id = format!("cursor-siblings-{}", uuid::Uuid::new_v4());
        let cancellation_a = CancellationToken::new();
        let cancellation_b = CancellationToken::new();
        let make_key = |generation: &str| CursorBridgeKey {
            provider: "cursor",
            session_id: session_id.clone(),
            model: "test-model".to_owned(),
            generation: generation.to_owned(),
        };
        let key_a = make_key("generation-a");
        let key_b = make_key("generation-b");
        for (key, cancellation) in [
            (&key_a, cancellation_a.clone()),
            (&key_b, cancellation_b.clone()),
        ] {
            let state = CursorRunState {
                connection: CursorRunConnection::test_connection(cancellation),
                blobs: HashMap::new(),
                advertised_tools: HashSet::new(),
                tool_definitions: Vec::new(),
                pending_execs: HashMap::new(),
                generation: key.generation.clone(),
                checkpoint: None,
                state_memory_permit: None,
                state_memory_units: 0,
                last_used: Instant::now(),
            };
            assert!(park_cursor_run(key, state));
        }

        let new_turn_key = make_key("new-user-turn");
        assert!(take_parked_run(&new_turn_key, false).is_none());
        {
            let registry = parked_runs().lock().expect("parked registry");
            assert!(registry.contains_key(&key_a));
            assert!(registry.contains_key(&key_b));
        }

        drop(take_parked_run(&key_a, true));
        assert!(cancellation_a.is_cancelled());
        assert!(!cancellation_b.is_cancelled());
        assert_eq!(cancel_cursor_session(&session_id), 1);
        assert!(cancellation_b.is_cancelled());
    }

    #[test]
    fn cursor_native_exec_requests_get_typed_fail_closed_results() {
        use crate::cursor_proto::agent::v1::exec_server_message::Message as ExecRequest;

        let requests = vec![
            ExecRequest::ReadArgs(Default::default()),
            ExecRequest::LsArgs(Default::default()),
            ExecRequest::WriteArgs(Default::default()),
            ExecRequest::DeleteArgs(Default::default()),
            ExecRequest::GrepArgs(Default::default()),
            ExecRequest::DiagnosticsArgs(Default::default()),
            ExecRequest::ShellArgs(Default::default()),
            ExecRequest::ShellStreamArgs(Default::default()),
            ExecRequest::BackgroundShellSpawnArgs(Default::default()),
            ExecRequest::ListMcpResourcesExecArgs(Default::default()),
            ExecRequest::ReadMcpResourceExecArgs(Default::default()),
            ExecRequest::FetchArgs(Default::default()),
            ExecRequest::RecordScreenArgs(Default::default()),
            ExecRequest::ComputerUseArgs(Default::default()),
            ExecRequest::WriteShellStdinArgs(Default::default()),
        ];

        for (index, request) in requests.iter().enumerate() {
            let response = reject_native_exec(index as u32, "exec-id".to_owned(), request)
                .expect("native request is rejected with a typed response");
            let Some(agent_client_message::Message::ExecClientMessage(response)) = response.message
            else {
                panic!("native request response uses the exec envelope");
            };
            assert_eq!(response.id, index as u32);
            assert_eq!(response.exec_id, "exec-id");
            assert!(
                response.message.is_some(),
                "typed response payload is present"
            );
        }

        assert!(
            reject_native_exec(
                0,
                String::new(),
                &ExecRequest::RequestContextArgs(Default::default())
            )
            .is_none()
        );
        assert!(
            reject_native_exec(0, String::new(), &ExecRequest::McpArgs(Default::default()))
                .is_none()
        );
    }

    #[test]
    fn cursor_hosted_interactions_are_rejected_except_unanswerable_vm_setup() {
        use crate::cursor_proto::agent::v1::interaction_query::Query as InteractionQuery;

        let rejectable = [
            InteractionQuery::WebSearchRequestQuery(Default::default()),
            InteractionQuery::AskQuestionInteractionQuery(Default::default()),
            InteractionQuery::SwitchModeRequestQuery(Default::default()),
            InteractionQuery::ExaSearchRequestQuery(Default::default()),
            InteractionQuery::ExaFetchRequestQuery(Default::default()),
            InteractionQuery::CreatePlanRequestQuery(Default::default()),
        ];
        for (index, query) in rejectable.iter().enumerate() {
            let response = reject_interaction_query(index as u32, Some(query))
                .expect("hosted interaction is rejected with a typed response");
            let Some(agent_client_message::Message::InteractionResponse(response)) =
                response.message
            else {
                panic!("interaction response has its protocol envelope");
            };
            assert_eq!(response.id, index as u32);
            assert!(response.result.is_some());
        }

        let setup_vm = InteractionQuery::SetupVmEnvironmentArgs(Default::default());
        assert!(reject_interaction_query(0, Some(&setup_vm)).is_none());
        assert!(reject_interaction_query(0, None).is_none());
    }

    #[derive(Default)]
    struct ServerState {
        run_requests: usize,
        tool_call_id: String,
        returned_tool_result_is_error: Option<bool>,
        returned_tool_result_text: Option<String>,
    }

    async fn next_client_message<S, E>(
        body: &mut S,
        decoder: &mut ConnectFrameDecoder,
        pending: &mut VecDeque<ConnectFrame>,
    ) -> AgentClientMessage
    where
        S: Stream<Item = Result<Bytes, E>> + Unpin,
        E: std::fmt::Debug,
    {
        loop {
            if let Some(frame) = pending.pop_front() {
                return AgentClientMessage::decode(frame.payload.as_slice())
                    .expect("client message protobuf");
            }
            let bytes = body
                .next()
                .await
                .expect("client stream remains open")
                .expect("client stream bytes");
            pending.extend(decoder.push(&bytes).expect("client Connect frame"));
        }
    }

    async fn run_endpoint(
        State(state): State<Arc<Mutex<ServerState>>>,
        request: Request<Body>,
    ) -> Response<Body> {
        let mut body = request.into_body().into_data_stream();
        let initial = body
            .next()
            .await
            .expect("initial request frame")
            .expect("initial request bytes");
        let mut initial_decoder = ConnectFrameDecoder::default();
        let initial_frames = initial_decoder
            .push(&initial)
            .expect("initial request Connect frame");
        let run = AgentClientMessage::decode(initial_frames[0].payload.as_slice())
            .expect("initial Run request");
        assert!(matches!(
            run.message,
            Some(agent_client_message::Message::RunRequest(_))
        ));
        state.lock().expect("test server state").run_requests += 1;
        let tool_call_id = state
            .lock()
            .expect("test server state")
            .tool_call_id
            .clone();

        let response_stream = async_stream::stream! {
            let request_context = AgentServerMessage {
                message: Some(agent_server_message::Message::ExecServerMessage(
                    ExecServerMessage {
                        id: 4,
                        exec_id: "context-exec".to_owned(),
                        message: Some(exec_server_message::Message::RequestContextArgs(
                            RequestContextArgs::default(),
                        )),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            };
            yield Ok::<Bytes, std::convert::Infallible>(
                Bytes::from(encode_connect_frame(0, &request_context.encode_to_vec()).unwrap())
            );

            let mut decoder = ConnectFrameDecoder::default();
            let mut pending = VecDeque::new();
            let context_response = next_client_message(&mut body, &mut decoder, &mut pending).await;
            assert!(matches!(
                context_response.message,
                Some(agent_client_message::Message::ExecClientMessage(_))
            ));

            let mcp = AgentServerMessage {
                message: Some(agent_server_message::Message::ExecServerMessage(
                    ExecServerMessage {
                        id: 9,
                        exec_id: String::new(),
                        message: Some(exec_server_message::Message::McpArgs(McpArgs {
                            name: "read_file".to_owned(),
                            args: HashMap::from([(
                                "path".to_owned(),
                                prost_types::Value {
                                    kind: Some(prost_types::value::Kind::StringValue(
                                        "README.md".to_owned(),
                                    )),
                                }
                                .encode_to_vec(),
                            )]),
                            tool_call_id,
                            provider_identifier: "grok".to_owned(),
                            tool_name: "read_file".to_owned(),
                        })),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            };
            yield Ok(Bytes::from(encode_connect_frame(0, &mcp.encode_to_vec()).unwrap()));

            let turn_ended = AgentServerMessage {
                message: Some(agent_server_message::Message::InteractionUpdate(
                    InteractionUpdate {
                        message: Some(interaction_update::Message::TurnEnded(
                            TurnEndedUpdate {},
                        )),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            };
            yield Ok(Bytes::from(encode_connect_frame(0, &turn_ended.encode_to_vec()).unwrap()));

            let result = next_client_message(&mut body, &mut decoder, &mut pending).await;
            let Some(agent_client_message::Message::ExecClientMessage(exec)) = result.message else {
                panic!("expected MCP result response");
            };
            assert_eq!(exec.id, 9);
            assert!(exec.exec_id.is_empty());
            let Some(exec_client_message::Message::McpResult(result)) = exec.message else {
                panic!("expected MCP result payload");
            };
            let Some(mcp_result::Result::Success(success)) = result.result else {
                panic!("expected MCP success envelope");
            };
            let result_text = success.content.iter().find_map(|item| {
                match item.content.as_ref() {
                    Some(mcp_tool_result_content_item::Content::Text(text)) => Some(text.text.clone()),
                    _ => None,
                }
            });
            {
                let mut state = state.lock().expect("test server state");
                state.returned_tool_result_is_error = Some(success.is_error);
                state.returned_tool_result_text = result_text;
            }

            let text = AgentServerMessage {
                message: Some(agent_server_message::Message::InteractionUpdate(
                    InteractionUpdate {
                        message: Some(interaction_update::Message::TextDelta(
                            crate::cursor_proto::agent::v1::TextDeltaUpdate {
                                text: "Permission denial was reported to Cursor.".to_owned(),
                            },
                        )),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            };
            yield Ok(Bytes::from(encode_connect_frame(0, &text.encode_to_vec()).unwrap()));
            yield Ok(Bytes::from(encode_connect_frame(0, &turn_ended.encode_to_vec()).unwrap()));
        };
        Response::builder()
            .header("content-type", "application/connect+proto")
            .body(Body::from_stream(response_stream))
            .expect("Connect response")
    }

    async fn start_server(
        state: Arc<Mutex<ServerState>>,
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
            .route(RUN_RPC, post(run_endpoint))
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

    #[derive(Debug)]
    struct FixedBearer;

    impl crate::BearerResolver for FixedBearer {
        fn current_bearer(&self) -> Option<String> {
            Some("fixture-cursor-token".to_owned())
        }
    }

    fn request(model: &str, session_id: &str) -> ConversationRequest {
        ConversationRequest {
            items: vec![ConversationItem::User(UserItem {
                content: vec![ContentPart::Text {
                    text: "Read README.md".into(),
                }],
                ..Default::default()
            })],
            tools: vec![ToolSpec {
                name: "read_file".to_owned(),
                description: Some("Read a file".to_owned()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            }],
            model: Some(model.to_owned()),
            x_grok_session_id: Some(session_id.to_owned()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn sampler_actor_routes_cursor_requests_through_dedicated_transport() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let server_state = Arc::new(Mutex::new(ServerState {
            tool_call_id: "cursor-workflow-call".to_owned(),
            ..ServerState::default()
        }));
        let (origin, certificate, server) = start_server(server_state.clone()).await;
        let config = SamplerConfig {
            api_backend: crate::ApiBackend::Cursor,
            model: "test-model".to_owned(),
            max_retries: Some(0),
            bearer_resolver: Some(Arc::new(FixedBearer)),
            ..Default::default()
        };
        let _transport = crate::cursor_transport::install_test_transport(
            "workflow-cursor-child",
            &origin,
            certificate,
            "fixture-token",
        );
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let sampler = crate::SamplerActor::spawn(config, crate::RetryPolicy::default(), event_tx);

        let (response, _) = sampler
            .submit_and_collect(
                RequestId::from("workflow-cursor-spawn"),
                request("test-model", "workflow-cursor-child"),
            )
            .await
            .expect("spawned Cursor request reaches the dedicated transport");

        assert_eq!(response.stop_reason, Some(StopReason::ToolCalls));
        assert_eq!(
            server_state.lock().expect("test server state").run_requests,
            1
        );
        cancel_cursor_session("workflow-cursor-child");
        server.abort();
    }

    #[test]
    fn request_output_limit_overrides_the_model_default() {
        assert_eq!(effective_output_limit(Some(96), Some(8_192)), Some(96));
        assert_eq!(effective_output_limit(None, Some(8_192)), Some(8_192));
        assert_eq!(effective_output_limit(None, None), None);
    }

    #[test]
    fn cursor_composite_tool_call_id_keeps_wire_identity_and_uses_safe_local_id() {
        let tool_call_id = "call-ee7424a7-2b6e-459c-9f60-c82655b4cbd5-75\n\
                            fc_p16xCke-4SRMt5-02fe4358-aws_uw2_0";
        let (wire_id, local_id) =
            prepare_cursor_tool_call_id(tool_call_id.to_owned()).expect("valid Cursor tool ID");

        assert_eq!(wire_id, tool_call_id);
        assert!(!local_id.chars().any(char::is_control));
        assert!(valid_cursor_identifier(&local_id, MAX_CURSOR_EXEC_ID_BYTES));
    }

    #[test]
    fn cursor_tool_call_id_still_rejects_other_control_characters() {
        let error = prepare_cursor_tool_call_id("call-id\tprovider-id".to_owned())
            .expect_err("tab-separated IDs remain invalid");

        assert_eq!(
            error.to_string(),
            "reqwest error stream: Cursor returned an invalid tool request identifier."
        );
    }

    #[test]
    fn cursor_composite_tool_call_id_rejects_malformed_values() {
        let oversized = format!("call-{}\nfc_metadata", "a".repeat(MAX_CURSOR_EXEC_ID_BYTES));
        let invalid_ids = [
            "other-id\nfc_metadata",
            "call-identifier\nother_metadata",
            "call-\nfc_metadata",
            "call-identifier\nfc_",
            "call-identifier\r\nfc_metadata",
            "call-identifier\nfc_metadata\nextra",
            oversized.as_str(),
        ];

        for tool_call_id in invalid_ids {
            assert!(
                prepare_cursor_tool_call_id(tool_call_id.to_owned()).is_err(),
                "accepted malformed Cursor tool ID: {tool_call_id:?}"
            );
        }
    }

    #[test]
    fn missing_cursor_tool_call_id_gets_a_safe_local_id() {
        let (wire_id, local_id) =
            prepare_cursor_tool_call_id(String::new()).expect("generated Cursor tool ID");

        assert_eq!(wire_id, local_id);
        assert!(uuid::Uuid::parse_str(&local_id).is_ok());
    }

    #[tokio::test]
    async fn fresh_unpolled_run_admission_is_limited_by_total_state_budget() {
        const PER_RUN_BYTES: usize = 8 * 1024 * 1024;
        let budget = Arc::new(Semaphore::new(
            (MAX_PARKED_STATE_BYTES / CURSOR_STATE_MEMORY_UNIT_BYTES) as u32 as usize,
        ));
        let mut tasks = Vec::new();
        for _ in 0..17 {
            let budget = budget.clone();
            tasks.push(tokio::spawn(async move {
                reserve_state_memory_from(budget, PER_RUN_BYTES)
            }));
        }

        // Each retained permit models an opened but unpolled fresh stream.
        let mut admitted = Vec::new();
        for task in tasks {
            if let Some(reservation) = task.await.expect("reservation task") {
                admitted.push(reservation);
            }
        }
        assert_eq!(admitted.len(), 16);
        assert!(reserve_state_memory_from(budget.clone(), PER_RUN_BYTES).is_none());

        drop(admitted.pop());
        assert!(reserve_state_memory_from(budget, PER_RUN_BYTES).is_some());
    }

    async fn cursor_tool_identifier_resumes_the_same_run_stream(tool_call_id: &str) {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let generation = format!("testgen{}", uuid::Uuid::new_v4().simple());
        let server_state = Arc::new(Mutex::new(ServerState {
            tool_call_id: tool_call_id.to_owned(),
            ..ServerState::default()
        }));
        let (origin, certificate, server) = start_server(server_state.clone()).await;
        let transport = CursorRunTransport::from_test_origin(&origin, certificate);
        let cancellation = CancellationToken::new();
        let request = request("test-model", "cursor-runtime-test-session");
        let payload = build_cursor_run_payload(request.clone()).expect("Cursor Run payload");
        let key = CursorBridgeKey {
            provider: "cursor",
            session_id: "cursor-runtime-test-session".to_owned(),
            model: "test-model".to_owned(),
            generation: generation.clone(),
        };
        let connection = transport
            .open(
                payload.initial_frame,
                "fixture-cursor-token",
                cancellation.clone(),
            )
            .await
            .expect("open fake Cursor Run");
        let state = CursorRunState {
            connection,
            blobs: payload.blobs,
            advertised_tools: payload.advertised_tools,
            tool_definitions: payload.advertised_tool_definitions,
            pending_execs: HashMap::new(),
            generation: generation.clone(),
            checkpoint: None,
            state_memory_permit: None,
            state_memory_units: 0,
            last_used: Instant::now(),
        };
        let first_events = run_state_stream(
            state,
            key.clone(),
            "test-model".to_owned(),
            RequestId::from("cursor-first-call"),
            Duration::from_secs(5),
            None,
            cancellation.clone(),
        )
        .collect::<Vec<_>>()
        .await;
        let first_response = first_events.into_iter().find_map(|event| match event {
            SamplingEvent::Completed { response, .. } => Some(response),
            SamplingEvent::Failed { error, .. } => panic!("Cursor run failed: {error:?}"),
            _ => None,
        });
        let first_response = first_response.expect("first sampler call completed");
        assert_eq!(first_response.stop_reason, Some(StopReason::ToolCalls));
        let tool_call = first_response
            .assistant()
            .expect("assistant response")
            .tool_calls
            .first()
            .expect("Grok-advertised tool call")
            .clone();
        assert_eq!(tool_call.name, "read_file");
        let generated_id = tool_call
            .id
            .strip_prefix(&format!("cursor-{generation}-"))
            .expect("provider generation prefix");
        if tool_call_id.is_empty() {
            assert!(uuid::Uuid::parse_str(generated_id).is_ok());
        } else {
            assert_eq!(
                generated_id,
                format!("opaque-{}", blake3::hash(tool_call_id.as_bytes()).to_hex())
            );
        }

        let continuation = ConversationRequest {
            items: vec![
                ConversationItem::User(UserItem {
                    content: vec![ContentPart::Text {
                        text: "Read README.md".into(),
                    }],
                    ..Default::default()
                }),
                ConversationItem::assistant_tool_calls(vec![tool_call.clone()]),
                ConversationItem::tool_result_error(
                    tool_call.id.to_string(),
                    "Permission denied by Grok Build.",
                ),
            ],
            model: Some("test-model".to_owned()),
            x_grok_session_id: Some("cursor-runtime-test-session".to_owned()),
            ..Default::default()
        };
        let config = SamplerConfig {
            model: "test-model".to_owned(),
            ..Default::default()
        };
        let resumed = open_run_stream(
            &config,
            continuation,
            RequestId::from("cursor-second-call"),
            Duration::from_secs(5),
            cancellation,
        )
        .await
        .expect("resume parked Cursor stream")
        .collect::<Vec<_>>()
        .await;
        let final_response = resumed.into_iter().find_map(|event| match event {
            SamplingEvent::Completed { response, .. } => Some(response),
            SamplingEvent::Failed { error, .. } => panic!("Cursor continuation failed: {error:?}"),
            _ => None,
        });
        assert_eq!(
            final_response
                .expect("continuation completes")
                .assistant_text(),
            "Permission denial was reported to Cursor."
        );
        let server_state = server_state.lock().expect("server state");
        assert_eq!(server_state.returned_tool_result_is_error, Some(true));
        assert_eq!(
            server_state.returned_tool_result_text.as_deref(),
            Some("Permission denied by Grok Build.")
        );
        drop(server_state);
        PARKED_RUNS
            .get()
            .expect("parked registry initialized")
            .lock()
            .expect("parked run registry")
            .remove(&key);
        server.abort();
    }

    #[tokio::test]
    async fn missing_cursor_tool_identifiers_resume_the_same_run_stream() {
        cursor_tool_identifier_resumes_the_same_run_stream("").await;
    }

    #[tokio::test]
    async fn cursor_composite_tool_identifier_resumes_the_same_run_stream() {
        cursor_tool_identifier_resumes_the_same_run_stream(
            "call-ee7424a7-2b6e-459c-9f60-c82655b4cbd5-75\n\
             fc_p16xCke-4SRMt5-02fe4358-aws_uw2_0",
        )
        .await;
    }

    #[test]
    fn integer_valued_json_numbers_are_coerced_to_integers() {
        let value = proto_value_to_json(prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(80.0)),
        });
        assert_eq!(value, serde_json::json!(80));
        assert!(value.as_u64() == Some(80) || value.as_i64() == Some(80));

        let nested = proto_value_to_json(prost_types::Value {
            kind: Some(prost_types::value::Kind::StructValue(prost_types::Struct {
                fields: [(
                    "offset".to_owned(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::NumberValue(80.0)),
                    },
                )]
                .into_iter()
                .collect(),
            })),
        });
        assert_eq!(nested["offset"], serde_json::json!(80));

        let fractional = proto_value_to_json(prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(1.5)),
        });
        assert_eq!(fractional, serde_json::json!(1.5));
    }

    #[test]
    fn checkpoint_occupancy_replaces_completion_only_totals() {
        let response = build_response(
            "ok".into(),
            String::new(),
            Vec::new(),
            "test-model".into(),
            Some(StopReason::Stop),
            50,
            12_345,
            1,
        );
        let usage = response.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 12_345);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 12_345);
    }

    #[test]
    fn zero_token_resource_exhausted_rotates_once_per_failure_streak() {
        let base = format!("rotate-{}", uuid::Uuid::new_v4());
        assert_eq!(resolved_cursor_conversation_id(&base, Some(&base)), base);
        note_cursor_stream_poison(&base, false, &CursorTransportError::ResourceExhausted);
        let rotated = resolved_cursor_conversation_id(&base, Some(&base));
        assert_ne!(rotated, base);
        note_cursor_stream_poison(&base, false, &CursorTransportError::ResourceExhausted);
        assert_eq!(
            resolved_cursor_conversation_id(&base, Some(&base)),
            rotated,
            "a second poison in the same streak must not rotate again"
        );
        mark_cursor_conversation_healthy(&base);
        note_cursor_stream_poison(&base, false, &CursorTransportError::ResourceExhausted);
        let rotated_again = resolved_cursor_conversation_id(&base, Some(&base));
        assert_ne!(rotated_again, rotated);
        assert_eq!(
            resolved_cursor_conversation_id(&base, Some("oneshot-conv")),
            "oneshot-conv"
        );
        note_cursor_stream_poison(&base, true, &CursorTransportError::ResourceExhausted);
        assert_eq!(
            resolved_cursor_conversation_id(&base, Some(&base)),
            rotated_again,
            "output tokens mean the conversation is not poisoned"
        );
    }
}
