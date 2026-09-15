//! Authenticated Cursor model discovery over Cursor's unary protobuf RPCs.
//!
//! This deliberately has its own request builder: it never reuses Grok's
//! default headers, configured model URL, API key, or redirect policy.

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Duration;

use futures_util::StreamExt;
use prost::Message;
use reqwest::header::{CONTENT_TYPE, HeaderValue};
use reqwest::{Client, StatusCode, Url};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::cursor_connect::{ConnectFrameDecoder, MAX_CONNECT_FRAME_BYTES};
use crate::cursor_proto::agent::v1::{GetUsableModelsRequest, GetUsableModelsResponse};
use crate::cursor_wire::{
    CursorParameterizedModel, decode_available_models_response, encode_available_models_request,
};

const DEFAULT_AGENT_ORIGIN: &str = "https://agentn.us.api5.cursor.sh";
const DEFAULT_AI_ORIGIN: &str = "https://api2.cursor.sh";
const DEFAULT_CURSOR_CLIENT_VERSION: &str = "cli-2026.07.23-e383d2b";
const MAX_USABLE_MODELS_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_AVAILABLE_MODELS_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_USABLE_MODELS: usize = 2_048;
const UNARY_TIMEOUT: Duration = Duration::from_secs(20);
const USABLE_MODELS_RPC: &str = "/agent.v1.AgentService/GetUsableModels";
const AVAILABLE_MODELS_RPC: &str = "/aiserver.v1.AiService/AvailableModels";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorDiscoveredModel {
    pub id: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub reasoning: bool,
    pub supports_images: Option<bool>,
    pub supports_max_mode: Option<bool>,
    pub max_mode: bool,
    pub context_window: u64,
    pub max_mode_context_window: Option<u64>,
    pub context_window_is_inferred: bool,
    pub max_output_tokens: u32,
    pub variants: Vec<crate::cursor_wire::CursorParameterizedVariant>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorModelCatalog {
    pub models: Vec<CursorDiscoveredModel>,
    pub parameterized_models: Vec<CursorParameterizedModel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CursorCatalogError {
    #[error("Cursor agent endpoint configuration is not a trusted HTTPS Cursor origin.")]
    UnsafeEndpoint,
    #[error("Could not initialize the Cursor inference transport.")]
    TransportUnavailable,
    #[error("Cursor rejected the local session. Sign in to Cursor Desktop and retry.")]
    Unauthorized,
    #[error("No signed-in Cursor Desktop access token is available.")]
    MissingCredential,
    #[error("Cursor model discovery failed with HTTP status {0}.")]
    HttpStatus(u16),
    #[error("Cursor model discovery was cancelled.")]
    Cancelled,
    #[error("Cursor model discovery response exceeded its size limit.")]
    ResponseTooLarge,
    #[error("Cursor returned an invalid model catalog response.")]
    InvalidResponse,
    #[error("Cursor returned no usable models for this account.")]
    NoModels,
}

/// Cursor's dedicated unary-RPC client with a validated Cursor-owned origin.
#[derive(Clone)]
pub struct CursorCatalogClient {
    http: Client,
    agent_origin: Url,
    ai_origin: Url,
}

impl std::fmt::Debug for CursorCatalogClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorCatalogClient")
            .field("agent_origin", &self.agent_origin)
            .field("ai_origin", &self.ai_origin)
            .finish_non_exhaustive()
    }
}

impl CursorCatalogClient {
    /// Resolve Cursor's regional agent URL from its local CLI config, validating
    /// that the bearer can only be sent to a TLS-protected Cursor-owned host.
    pub fn new() -> Result<Self, CursorCatalogError> {
        let agent_origin = resolve_agent_origin()?;
        let ai_origin =
            Url::parse(DEFAULT_AI_ORIGIN).map_err(|_| CursorCatalogError::TransportUnavailable)?;
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .no_proxy()
            .timeout(UNARY_TIMEOUT)
            .build()
            .map_err(|_| CursorCatalogError::TransportUnavailable)?;
        Ok(Self {
            http,
            agent_origin,
            ai_origin,
        })
    }

    /// Fetch both Cursor catalogs. Parameterized metadata is required so the
    /// account model list is not presented with silently inferred variants.
    pub async fn discover_models(
        &self,
        access_token: &str,
        cancellation: &CancellationToken,
    ) -> Result<CursorModelCatalog, CursorCatalogError> {
        let request = GetUsableModelsRequest::default().encode_to_vec();
        let usable_future = self.call_unary(
            &self.agent_origin,
            USABLE_MODELS_RPC,
            access_token,
            request,
            cancellation,
            MAX_USABLE_MODELS_RESPONSE_BYTES,
        );
        let parameter_future = self.call_unary(
            &self.ai_origin,
            AVAILABLE_MODELS_RPC,
            access_token,
            encode_available_models_request(),
            cancellation,
            MAX_AVAILABLE_MODELS_RESPONSE_BYTES,
        );
        let (usable_response, parameter_response) = tokio::join!(usable_future, parameter_future);

        // Cancellation must not be turned into a partial catalog just because
        // one of the two required RPCs completed first.
        if cancellation.is_cancelled() {
            return Err(CursorCatalogError::Cancelled);
        }

        let usable_response = usable_response?;
        let usable_payload = unwrap_unary_response(&usable_response)?;
        let usable_models = GetUsableModelsResponse::decode(usable_payload.as_ref())
            .map_err(|_| CursorCatalogError::InvalidResponse)?
            .models;
        if usable_models.len() > MAX_USABLE_MODELS {
            return Err(CursorCatalogError::InvalidResponse);
        }

        let parameter_response = parameter_response?;
        let parameter_payload = unwrap_unary_response(&parameter_response)?;
        let parameterized_models = decode_available_models_response(parameter_payload.as_ref())
            .map_err(|_| CursorCatalogError::InvalidResponse)?;

        let models = merge_model_catalog(usable_models, &parameterized_models);
        if models.is_empty() {
            return Err(CursorCatalogError::NoModels);
        }
        Ok(CursorModelCatalog {
            models,
            parameterized_models,
        })
    }

    async fn call_unary(
        &self,
        origin: &Url,
        rpc_path: &str,
        access_token: &str,
        request_body: Vec<u8>,
        cancellation: &CancellationToken,
        response_limit: usize,
    ) -> Result<Vec<u8>, CursorCatalogError> {
        if access_token.is_empty() {
            return Err(CursorCatalogError::MissingCredential);
        }
        if cancellation.is_cancelled() {
            return Err(CursorCatalogError::Cancelled);
        }
        let url = origin
            .join(rpc_path)
            .map_err(|_| CursorCatalogError::UnsafeEndpoint)?;
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = self
            .http
            .post(url)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/proto"))
            .header("connect-protocol-version", "1")
            .header("te", "trailers")
            .header("x-ghost-mode", "true")
            .header("x-cursor-client-version", DEFAULT_CURSOR_CLIENT_VERSION)
            .header("x-cursor-client-type", "cli")
            .header("x-request-id", request_id)
            .bearer_auth(access_token)
            .body(request_body)
            .build()
            .map_err(|_| CursorCatalogError::TransportUnavailable)?;

        // Do not follow redirects, and do not allow caller-provided extra headers.
        // The builder above has no route from model config or Grok's base URL.
        let response = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(CursorCatalogError::Cancelled),
            response = self.http.execute(request) => {
                response.map_err(|_| CursorCatalogError::TransportUnavailable)?
            }
        };
        let status = response.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(CursorCatalogError::Unauthorized);
        }
        if !status.is_success() {
            return Err(CursorCatalogError::HttpStatus(status.as_u16()));
        }

        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Err(CursorCatalogError::Cancelled),
                chunk = stream.next() => chunk,
            };
            let Some(chunk) = chunk else { break };
            let chunk = chunk.map_err(|_| CursorCatalogError::TransportUnavailable)?;
            if body.len().saturating_add(chunk.len()) > response_limit {
                return Err(CursorCatalogError::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}

pub(crate) fn resolve_agent_origin() -> Result<Url, CursorCatalogError> {
    if let Some(config_path) = cursor_cli_config_path()
        && let Ok(contents) = std::fs::read(config_path)
        && let Some(url) = parse_cursor_cli_agent_url(&contents)?
    {
        return Ok(url);
    }
    Url::parse(DEFAULT_AGENT_ORIGIN).map_err(|_| CursorCatalogError::TransportUnavailable)
}

fn parse_cursor_cli_agent_url(contents: &[u8]) -> Result<Option<Url>, CursorCatalogError> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(contents) else {
        return Ok(None);
    };
    let Some(raw) = value
        .pointer("/serverConfigCache/agentUrlConfig/agentnUrl")
        .or_else(|| value.pointer("/serverConfigCache/agentUrlConfig/agentUrl"))
        .and_then(serde_json::Value::as_str)
        .filter(|raw| !raw.trim().is_empty())
    else {
        return Ok(None);
    };
    let url = Url::parse(raw).map_err(|_| CursorCatalogError::UnsafeEndpoint)?;
    validate_cursor_agent_url(&url)?;
    Ok(Some(url))
}

fn cursor_cli_config_path() -> Option<PathBuf> {
    let config_dir = std::env::var_os("CURSOR_CONFIG_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(PathBuf::from)
                .map(|home| home.join(".cursor"))
        })?;
    Some(config_dir.join("cli-config.json"))
}

fn validate_cursor_agent_url(url: &Url) -> Result<(), CursorCatalogError> {
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let trusted_host = host == "cursor.sh"
        || host == "cursor.com"
        || host.ends_with(".cursor.sh")
        || host.ends_with(".cursor.com");
    let trusted_shape = url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && url.port_or_known_default() == Some(443)
        && url.query().is_none()
        && url.fragment().is_none()
        && (url.path().is_empty() || url.path() == "/");
    if trusted_host && trusted_shape {
        Ok(())
    } else {
        Err(CursorCatalogError::UnsafeEndpoint)
    }
}

fn unwrap_unary_response(bytes: &[u8]) -> Result<Cow<'_, [u8]>, CursorCatalogError> {
    let Some(&flags) = bytes.first() else {
        return Ok(Cow::Borrowed(bytes));
    };
    if !matches!(flags, 0 | 2) {
        return Ok(Cow::Borrowed(bytes));
    }

    let mut decoder = ConnectFrameDecoder::default();
    let frames = decoder
        .push(bytes)
        .map_err(|_| CursorCatalogError::InvalidResponse)?;
    decoder
        .finish()
        .map_err(|_| CursorCatalogError::InvalidResponse)?;
    let Some(frame) = frames.iter().find(|frame| !frame.is_end_stream()) else {
        return Ok(Cow::Borrowed(bytes));
    };
    if frame.payload.len() > MAX_CONNECT_FRAME_BYTES {
        return Err(CursorCatalogError::ResponseTooLarge);
    }
    Ok(Cow::Owned(frame.payload.clone()))
}

fn merge_model_catalog(
    models: Vec<crate::cursor_proto::agent::v1::ModelDetails>,
    parameterized: &[CursorParameterizedModel],
) -> Vec<CursorDiscoveredModel> {
    let mut parameter_models = std::collections::HashMap::with_capacity(parameterized.len());
    for (index, entry) in parameterized.iter().enumerate() {
        parameter_models
            .entry(entry.name.to_ascii_lowercase())
            .or_insert(index);
        if let Some(server_name) = entry.server_model_name.as_deref() {
            parameter_models
                .entry(server_name.to_ascii_lowercase())
                .or_insert(index);
        }
    }
    let mut merged = Vec::new();
    let mut seen_ids = std::collections::HashSet::with_capacity(models.len());
    for model in models {
        let id = model.model_id.trim();
        if id.is_empty() || !seen_ids.insert(id.to_ascii_lowercase()) {
            continue;
        }
        let name = if !model.display_name.is_empty() {
            model.display_name.clone()
        } else if !model.display_name_short.is_empty() {
            model.display_name_short.clone()
        } else if !model.display_model_id.is_empty() {
            model.display_model_id.clone()
        } else {
            id.to_owned()
        };
        let parameter_model = parameter_models
            .get(&id.to_ascii_lowercase())
            .or_else(|| parameter_models.get(&model.display_model_id.to_ascii_lowercase()))
            .and_then(|index| parameterized.get(*index));
        let inferred_context_window = infer_context_window(id, &name);
        let max_mode = model.max_mode.unwrap_or(false);
        let max_mode_context_window =
            parameter_model.and_then(|entry| entry.context_token_limit_for_max_mode);
        let context_window = parameter_model
            .and_then(|entry| {
                if max_mode {
                    max_mode_context_window.or(entry.context_token_limit)
                } else {
                    entry.context_token_limit
                }
            })
            .unwrap_or(inferred_context_window);
        let context_window_is_inferred = parameter_model
            .and_then(|entry| {
                if max_mode {
                    max_mode_context_window.or(entry.context_token_limit)
                } else {
                    entry.context_token_limit
                }
            })
            .is_none();
        merged.push(CursorDiscoveredModel {
            id: id.to_owned(),
            name: name.clone(),
            aliases: model.aliases,
            reasoning: model.thinking_details.is_some(),
            supports_images: parameter_model.and_then(|entry| entry.supports_images),
            supports_max_mode: parameter_model.and_then(|entry| entry.supports_max_mode),
            max_mode,
            context_window,
            max_mode_context_window,
            context_window_is_inferred,
            max_output_tokens: infer_max_output_tokens(id, &name),
            variants: parameter_model
                .map(|entry| entry.variants.clone())
                .unwrap_or_default(),
        });
    }
    merged.sort_by(|a, b| a.id.cmp(&b.id));
    merged
}

fn infer_context_window(id: &str, name: &str) -> u64 {
    let id_lower = id.to_ascii_lowercase();
    let text = format!("{id_lower} {}", name.to_ascii_lowercase());
    if text.contains("gpt-5.6") {
        if id_lower.split('-').any(|part| part == "1m") {
            500_000
        } else {
            272_000
        }
    } else if text.contains("1m") {
        1_000_000
    } else if text.contains("272k") {
        272_000
    } else if text.contains("256k") || text.contains("grok-4.5") || text.contains("grok-4.6") {
        256_000
    } else {
        200_000
    }
}

fn infer_max_output_tokens(id: &str, name: &str) -> u32 {
    let text = format!("{} {}", id.to_ascii_lowercase(), name.to_ascii_lowercase());
    if text.contains("gpt-5")
        || text.contains("claude-4.6")
        || text.contains("claude-5")
        || text.contains("opus 4.6")
        || text.contains("sonnet 4.6")
    {
        128_000
    } else {
        64_000
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::extract::State;
    use axum::http::{Request, Response, Version};
    use axum::routing::post;
    use prost::Message;
    use rcgen::{CertificateParams, KeyPair};
    use rustls::pki_types::PrivateKeyDer;

    use super::*;
    use crate::cursor_proto::agent::v1::{ModelDetails, ThinkingDetails};
    use crate::cursor_wire::{CursorModelParameter, CursorParameterizedVariant};

    #[derive(Clone, Debug)]
    struct CapturedRequest {
        path: String,
        authorization: Option<String>,
        content_type: Option<String>,
        cursor_client_type: Option<String>,
        has_grok_header: bool,
        version: Version,
        body: Vec<u8>,
    }

    #[derive(Default)]
    struct TestServerState {
        requests: Mutex<Vec<CapturedRequest>>,
        available_models_received: Option<Arc<tokio::sync::Notify>>,
        release_available_models: Option<Arc<tokio::sync::Notify>>,
    }

    async fn capture_catalog_request(
        State(state): State<Arc<TestServerState>>,
        request: Request<Body>,
    ) -> Response<Body> {
        let path = request.uri().path().to_owned();
        if path == AVAILABLE_MODELS_RPC {
            if let Some(received) = &state.available_models_received {
                received.notify_one();
            }
            if let Some(release) = &state.release_available_models {
                release.notified().await;
            }
        }
        let version = request.version();
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
        let cursor_client_type = request
            .headers()
            .get("x-cursor-client-type")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let has_grok_header = request
            .headers()
            .keys()
            .any(|name| name.as_str().starts_with("x-grok-"));
        let body = to_bytes(request.into_body(), MAX_AVAILABLE_MODELS_RESPONSE_BYTES)
            .await
            .expect("read request body")
            .to_vec();
        state
            .requests
            .lock()
            .expect("request capture lock")
            .push(CapturedRequest {
                path: path.clone(),
                authorization,
                content_type,
                cursor_client_type,
                has_grok_header,
                version,
                body,
            });

        let response_body = match path.as_str() {
            USABLE_MODELS_RPC => GetUsableModelsResponse {
                models: vec![ModelDetails {
                    model_id: "gpt-5.6".to_string(),
                    display_name: "GPT-5.6".to_string(),
                    display_model_id: "gpt-5.6".to_string(),
                    thinking_details: Some(ThinkingDetails {}),
                    ..Default::default()
                }],
            }
            .encode_to_vec(),
            AVAILABLE_MODELS_RPC => parameterized_models_fixture(),
            _ => {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::empty())
                    .expect("404 response");
            }
        };
        Response::builder()
            .header(CONTENT_TYPE, "application/proto")
            .body(Body::from(response_body))
            .expect("protobuf response")
    }

    fn parameterized_models_fixture() -> Vec<u8> {
        fn varint(mut value: u64, output: &mut Vec<u8>) {
            while value >= 0x80 {
                output.push((value as u8 & 0x7f) | 0x80);
                value >>= 7;
            }
            output.push(value as u8);
        }
        fn bytes_field(field: u64, value: &[u8], output: &mut Vec<u8>) {
            varint((field << 3) | 2, output);
            varint(value.len() as u64, output);
            output.extend_from_slice(value);
        }
        fn varint_field(field: u64, value: u64, output: &mut Vec<u8>) {
            varint(field << 3, output);
            varint(value, output);
        }

        let mut model = Vec::new();
        bytes_field(1, b"gpt-5.6", &mut model);
        varint_field(10, 1, &mut model);
        varint_field(14, 1, &mut model);
        varint_field(15, 300_000, &mut model);
        let mut response = Vec::new();
        bytes_field(2, &model, &mut response);
        response
    }

    async fn start_http2_test_server(
        state: Arc<TestServerState>,
    ) -> (String, reqwest::Certificate, tokio::task::JoinHandle<()>) {
        let key_pair = KeyPair::generate().expect("generate test key");
        let certificate = CertificateParams::new(vec!["localhost".to_owned()])
            .expect("certificate params")
            .self_signed(&key_pair)
            .expect("self-signed localhost certificate");
        let certificate_der = certificate.der().clone();
        let private_key = PrivateKeyDer::Pkcs8(key_pair.serialize_der().into());
        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate_der.clone()], private_key)
            .expect("server certificate");
        server_config.alpn_protocols = vec![b"h2".to_vec()];
        let tls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));
        let app = Router::new()
            .route(USABLE_MODELS_RPC, post(capture_catalog_request))
            .route(AVAILABLE_MODELS_RPC, post(capture_catalog_request))
            .with_state(state);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
        listener
            .set_nonblocking(true)
            .expect("set listener nonblocking");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            axum_server::from_tcp_rustls(listener, tls_config)
                .expect("build TLS server")
                .serve(app.into_make_service())
                .await
                .expect("serve Cursor fake server");
        });
        let certificate = reqwest::Certificate::from_der(certificate_der.as_ref())
            .expect("reqwest test certificate");
        (format!("https://localhost:{port}"), certificate, server)
    }

    fn test_client(origin: &str, certificate: reqwest::Certificate) -> CursorCatalogClient {
        let http = Client::builder()
            .add_root_certificate(certificate)
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .no_proxy()
            .timeout(UNARY_TIMEOUT)
            .build()
            .expect("test HTTP/2 client");
        CursorCatalogClient {
            http,
            agent_origin: Url::parse(origin).expect("test origin"),
            ai_origin: Url::parse(origin).expect("test origin"),
        }
    }

    #[tokio::test]
    async fn discovers_both_model_catalogs_over_http2_without_grok_headers() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let state = Arc::new(TestServerState::default());
        let (origin, certificate, server) = start_http2_test_server(state.clone()).await;
        let client = test_client(&origin, certificate);
        let cancellation = CancellationToken::new();
        let catalog = client
            .discover_models("fixture-cursor-token", &cancellation)
            .await
            .expect("discover Cursor catalog");
        server.abort();

        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].id, "gpt-5.6");
        assert_eq!(catalog.models[0].context_window, 300_000);
        assert!(!catalog.models[0].context_window_is_inferred);
        assert_eq!(catalog.models[0].supports_images, Some(true));

        let mut requests = state.requests.lock().expect("capture lock").clone();
        requests.sort_by(|left, right| left.path.cmp(&right.path));
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].path, USABLE_MODELS_RPC);
        assert!(requests[0].body.is_empty());
        assert_eq!(requests[1].path, AVAILABLE_MODELS_RPC);
        assert_eq!(requests[1].body, encode_available_models_request());
        for request in requests {
            assert_eq!(request.version, Version::HTTP_2);
            assert_eq!(
                request.authorization.as_deref(),
                Some("Bearer fixture-cursor-token")
            );
            assert_eq!(request.content_type.as_deref(), Some("application/proto"));
            assert_eq!(request.cursor_client_type.as_deref(), Some("cli"));
            assert!(!request.has_grok_header);
        }
    }

    #[tokio::test]
    async fn cancellation_wins_when_second_required_catalog_call_is_pending() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let state = Arc::new(TestServerState {
            available_models_received: Some(Arc::new(tokio::sync::Notify::new())),
            release_available_models: Some(Arc::new(tokio::sync::Notify::new())),
            ..Default::default()
        });
        let (origin, certificate, server) = start_http2_test_server(state.clone()).await;
        let client = test_client(&origin, certificate);
        let cancellation = CancellationToken::new();
        let lookup = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                client
                    .discover_models("fixture-cursor-token", &cancellation)
                    .await
            }
        });

        state
            .available_models_received
            .as_ref()
            .expect("gate")
            .notified()
            .await;
        cancellation.cancel();

        assert_eq!(
            lookup.await.expect("lookup task"),
            Err(CursorCatalogError::Cancelled)
        );
        server.abort();
    }

    #[test]
    fn validates_only_https_cursor_owned_origins() {
        for allowed in [
            "https://agentn.us.api5.cursor.sh",
            "https://api2.cursor.sh",
            "https://region.cursor.com",
        ] {
            validate_cursor_agent_url(&Url::parse(allowed).expect("allowed URL"))
                .expect("Cursor-owned HTTPS origin");
        }
        for rejected in [
            "http://agentn.us.api5.cursor.sh",
            "https://cursor.sh.attacker.invalid",
            "https://agentn.us.api5.cursor.sh.attacker.invalid",
            "https://user:pass@cursor.sh",
            "https://cursor.sh:444",
            "https://cursor.sh/path",
        ] {
            assert_eq!(
                validate_cursor_agent_url(&Url::parse(rejected).expect("parse URL")),
                Err(CursorCatalogError::UnsafeEndpoint),
                "must reject {rejected}"
            );
        }
    }

    #[test]
    fn reads_only_validated_regional_endpoint_from_cli_config() {
        let contents = br#"{"serverConfigCache":{"agentUrlConfig":{"agentnUrl":"https://agentn.eu.api5.cursor.sh"}}}"#;
        let url = parse_cursor_cli_agent_url(contents)
            .expect("validate CLI config")
            .expect("configured endpoint");
        assert_eq!(url.host_str(), Some("agentn.eu.api5.cursor.sh"));
        assert_eq!(
            parse_cursor_cli_agent_url(
                br#"{"serverConfigCache":{"agentUrlConfig":{"agentnUrl":"https://evil.invalid"}}}"#
            ),
            Err(CursorCatalogError::UnsafeEndpoint)
        );
    }

    #[test]
    fn merges_authoritative_parameter_metadata_and_marks_inferred_limits() {
        let models = vec![
            ModelDetails {
                model_id: "gpt-5.6".to_string(),
                display_name: "GPT-5.6".to_string(),
                display_model_id: "gpt-5.6".to_string(),
                thinking_details: Some(ThinkingDetails {}),
                ..Default::default()
            },
            ModelDetails {
                model_id: "claude-4.6-opus".to_string(),
                display_name_short: "Opus 4.6".to_string(),
                ..Default::default()
            },
        ];
        let parameterized = vec![CursorParameterizedModel {
            name: "gpt-5.6".to_string(),
            context_token_limit: Some(300_000),
            supports_images: Some(true),
            supports_max_mode: Some(true),
            variants: vec![CursorParameterizedVariant {
                parameters: vec![CursorModelParameter {
                    id: "reasoning".to_string(),
                    value: "high".to_string(),
                }],
                is_max_mode: true,
                ..Default::default()
            }],
            ..Default::default()
        }];
        let merged = merge_model_catalog(models, &parameterized);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "claude-4.6-opus");
        assert_eq!(merged[0].context_window, 200_000);
        assert!(merged[0].context_window_is_inferred);
        assert_eq!(merged[0].max_output_tokens, 128_000);
        assert_eq!(merged[1].context_window, 300_000);
        assert!(!merged[1].context_window_is_inferred);
        assert_eq!(merged[1].supports_images, Some(true));
        assert!(merged[1].reasoning);
        assert_eq!(merged[1].variants[0].parameters[0].value, "high");
    }

    #[test]
    fn preserves_catalog_max_mode_and_deduplicates_model_ids() {
        let model = ModelDetails {
            model_id: "fable".to_owned(),
            max_mode: Some(true),
            ..Default::default()
        };
        let parameterized = vec![CursorParameterizedModel {
            name: "fable".to_owned(),
            context_token_limit: Some(200_000),
            context_token_limit_for_max_mode: Some(500_000),
            supports_max_mode: Some(true),
            ..Default::default()
        }];

        let merged = merge_model_catalog(vec![model.clone(), model], &parameterized);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].max_mode);
        assert_eq!(merged[0].context_window, 500_000);
        assert_eq!(merged[0].max_mode_context_window, Some(500_000));
        assert!(!merged[0].context_window_is_inferred);
    }
}
