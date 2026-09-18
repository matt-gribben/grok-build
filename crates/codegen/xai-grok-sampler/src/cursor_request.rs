//! Translate Grok's normalized conversation request into Cursor's Run RPC.
//!
//! This module builds only the initial client message and its content-addressed
//! prompt blobs. It deliberately has no credential or network access; the
//! transport owns those boundaries.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use prost::Message;
use prost_types::{ListValue, NullValue, Struct, Value, value::Kind};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;
use xai_grok_sampling_types::{
    ChatCompletionRequest, ChatContentBlock, ChatRequestMessage, ConversationItem,
    ConversationRequest, ConversationToolChoice, MessageContent, Role,
};

use crate::cursor_connect::{ConnectFrameError, encode_connect_frame};
use crate::cursor_proto::agent::v1::{
    AgentClientMessage, AgentRunRequest, ConversationAction, ConversationStateStructure,
    McpToolDefinition, McpTools, RequestedModel, RequestedModelModelParameterbytes,
    SelectedContext, UserMessage, UserMessageAction, agent_client_message, conversation_action,
};

const CONNECT_DATA_FRAME: u8 = 0;
const MAX_BLOB_BYTES: usize = 16 * 1024 * 1024;
const MAX_BLOB_STORE_BYTES: usize = 48 * 1024 * 1024;
const MAX_BLOB_COUNT: usize = 10_000;
const MAX_TOOL_COUNT: usize = 512;
const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
const MAX_RUN_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_PREFLIGHT_CONVERSATION_BYTES: usize = 16 * 1024 * 1024;
const MAX_PREFLIGHT_CONVERSATION_ITEMS: usize = 16_384;
const MAX_TOOL_SCHEMA_DEPTH: usize = 64;
const MAX_TOOL_SCHEMA_NODES: usize = 65_536;
const MCP_PROVIDER: &str = "grok";
const MODEL_ROUTE_PREFIX: &str = "cursor-route-v1:";

/// Exact account-selected Cursor model and parameters carried through Grok's
/// generic model string until the dedicated sampler builds `RequestedModel`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CursorModelRoute {
    pub model_id: String,
    #[serde(default)]
    pub parameters: Vec<crate::cursor_wire::CursorModelParameter>,
    #[serde(default)]
    pub max_mode: bool,
}

impl CursorModelRoute {
    pub fn to_model_string(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("Cursor model route is JSON serializable");
        format!("{MODEL_ROUTE_PREFIX}{}", URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn from_model_string(value: &str) -> Result<Option<Self>, CursorRequestError> {
        let Some(encoded) = value.strip_prefix(MODEL_ROUTE_PREFIX) else {
            return Ok(None);
        };
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| CursorRequestError::InvalidModelRoute)?;
        let route: Self =
            serde_json::from_slice(&bytes).map_err(|_| CursorRequestError::InvalidModelRoute)?;
        if route.model_id.is_empty()
            || route.model_id.len() > 256
            || route.model_id.chars().any(char::is_control)
            || route.parameters.len() > 64
            || route.parameters.iter().any(|parameter| {
                parameter.id.is_empty()
                    || parameter.id.len() > 128
                    || parameter.value.len() > 256
                    || parameter.id.chars().any(char::is_control)
                    || parameter.value.chars().any(char::is_control)
            })
        {
            return Err(CursorRequestError::InvalidModelRoute);
        }
        Ok(Some(route))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CursorRequestError {
    #[error("Cursor requests need a stable Grok session identifier.")]
    MissingSessionId,
    #[error("Cursor requests need a current user message.")]
    MissingCurrentUserMessage,
    #[error("This Cursor request contains unsupported image content.")]
    UnsupportedImage,
    #[error("Cursor does not support this sampling option.")]
    UnsupportedSamplingOption,
    #[error("Cursor supports automatic or disabled tool choice only.")]
    UnsupportedToolChoice,
    #[error("Cursor does not support hosted search tools through this adapter.")]
    UnsupportedHostedTools,
    #[error("Cursor does not support this structured-output request.")]
    UnsupportedStructuredOutput,
    #[error("The conversation history contains an uncorrelated tool call.")]
    MissingToolCallId,
    #[error("The Cursor tool schema is invalid or too large.")]
    InvalidToolSchema,
    #[error("The selected Cursor model route is invalid.")]
    InvalidModelRoute,
    #[error("The Cursor request exceeds a supported size limit.")]
    RequestTooLarge,
    #[error("The Cursor request could not be framed.")]
    InvalidFrame,
}

impl From<ConnectFrameError> for CursorRequestError {
    fn from(_: ConnectFrameError) -> Self {
        Self::InvalidFrame
    }
}

/// Initial Connect frame and the content-addressed blobs Cursor may request.
/// Its Debug implementation reports only sizes, not prompt contents.
pub(crate) struct CursorRunPayload {
    pub initial_frame: Vec<u8>,
    pub blobs: HashMap<Vec<u8>, Vec<u8>>,
    pub conversation_id: String,
    pub model: String,
    /// Grok tool names explicitly advertised on this Run, used to reject any
    /// Cursor MCP request that tries to invoke an unadvertised host tool.
    pub advertised_tools: HashSet<String>,
    pub advertised_tool_definitions: Vec<McpToolDefinition>,
}

impl std::fmt::Debug for CursorRunPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorRunPayload")
            .field("initial_frame_bytes", &self.initial_frame.len())
            .field("blob_count", &self.blobs.len())
            .field(
                "blob_bytes",
                &self.blobs.values().map(Vec::len).sum::<usize>(),
            )
            .field("conversation_id", &self.conversation_id)
            .field("model", &self.model)
            .field("advertised_tool_count", &self.advertised_tools.len())
            .finish()
    }
}

/// Build a Cursor `AgentClientMessage.run_request` from one normalized Grok
/// conversation. Current-turn user input travels in the action; earlier
/// transcript messages and caller instructions are stored as bounded blobs.
pub(crate) fn build_cursor_run_payload(
    request: ConversationRequest,
) -> Result<CursorRunPayload, CursorRequestError> {
    preflight_cursor_run_request(&request)?;
    let session_id = request
        .x_grok_session_id
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or(CursorRequestError::MissingSessionId)?;
    if !request.hosted_tools.is_empty() {
        return Err(CursorRequestError::UnsupportedHostedTools);
    }
    if request.temperature.is_some()
        || request.top_p.is_some()
        || request.reasoning_effort.is_some()
    {
        return Err(CursorRequestError::UnsupportedSamplingOption);
    }
    if request.json_schema.is_some() {
        return Err(CursorRequestError::UnsupportedStructuredOutput);
    }
    match request.tool_choice.as_ref() {
        Some(ConversationToolChoice::Required | ConversationToolChoice::Function(_)) => {
            return Err(CursorRequestError::UnsupportedToolChoice);
        }
        Some(ConversationToolChoice::Auto | ConversationToolChoice::None) | None => {}
    }

    let failed_tool_call_ids: HashSet<String> = request
        .items
        .iter()
        .filter_map(|item| match item {
            ConversationItem::ToolResult(result) if result.is_error => {
                Some(result.tool_call_id.clone())
            }
            _ => None,
        })
        .collect();
    let model = request.model.clone().unwrap_or_default();
    let model_route = CursorModelRoute::from_model_string(&model)?;
    let model_id = model_route
        .as_ref()
        .map_or_else(|| model.clone(), |route| route.model_id.clone());
    if model_id.is_empty() {
        return Err(CursorRequestError::UnsupportedSamplingOption);
    }
    let conversation_id = request
        .x_grok_conv_id
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| session_id.clone());
    let chat_request: ChatCompletionRequest = request.into();
    let current_user_index = chat_request
        .messages
        .iter()
        .rposition(|message| message.role == Role::User)
        .ok_or(CursorRequestError::MissingCurrentUserMessage)?;
    if chat_request.messages[current_user_index + 1..]
        .iter()
        .any(|message| message.role != Role::System)
    {
        // Tool-result continuation must use the existing parked Cursor stream.
        // Rebuilding it as a fresh Run would lose the pending exec correlation.
        return Err(CursorRequestError::UnsupportedSamplingOption);
    }
    if chat_request.messages[current_user_index]
        .content
        .blocks()
        .iter()
        .any(|part| matches!(part, ChatContentBlock::ImageUrl { .. }))
    {
        return Err(CursorRequestError::UnsupportedImage);
    }

    let current_user = &chat_request.messages[current_user_index];
    let user_text = text_content(current_user)?;
    let system_prompt = chat_request
        .messages
        .iter()
        .filter(|message| message.role == Role::System)
        .map(text_content)
        .collect::<Result<Vec<_>, _>>()?
        .join("\n\n");

    let mut blob_store = BlobStore::default();
    let system_blob_id = blob_store.insert(
        serde_json::to_vec(&json!({ "role": "system", "content": system_prompt }))
            .map_err(|_| CursorRequestError::RequestTooLarge)?,
    )?;
    let rules_blob_id = if system_prompt.trim().is_empty() {
        None
    } else {
        Some(
            blob_store.insert(
                serde_json::to_vec(&json!({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": format!("<rules>\n{system_prompt}\n</rules>")
                    }]
                }))
                .map_err(|_| CursorRequestError::RequestTooLarge)?,
            )?,
        )
    };

    let mut mcp_tools = Vec::new();
    let mut advertised_tools = HashSet::new();
    if chat_request.tool_choice.as_ref().is_none_or(|choice| {
        !matches!(choice, xai_grok_sampling_types::ToolChoice::Preset(value) if value == "none")
    }) {
        let definitions = chat_request.tools.as_deref().unwrap_or_default();
        if definitions.len() > MAX_TOOL_COUNT {
            return Err(CursorRequestError::RequestTooLarge);
        }
        let mut seen_tool_names = HashSet::new();
        for definition in definitions {
            let name = definition.function.name.trim();
            if name.is_empty() || !seen_tool_names.insert(name.to_owned()) {
                return Err(CursorRequestError::InvalidToolSchema);
            }
            let schema = json_to_proto_value(&definition.function.parameters).encode_to_vec();
            if schema.len() > MAX_TOOL_SCHEMA_BYTES {
                return Err(CursorRequestError::RequestTooLarge);
            }
            mcp_tools.push(McpToolDefinition {
                name: name.to_owned(),
                provider_identifier: MCP_PROVIDER.to_owned(),
                tool_name: name.to_owned(),
                description: definition.function.description.clone().unwrap_or_default(),
                input_schema: schema,
            });
            advertised_tools.insert(name.to_owned());
        }
    }

    let tool_names_by_call_id: HashMap<String, String> = chat_request
        .messages
        .iter()
        .take(current_user_index)
        .flat_map(|message| message.tool_calls.iter())
        .filter_map(|call| {
            call.id
                .as_ref()
                .map(|id| (id.clone(), call.function.name.clone()))
        })
        .collect();

    let mut root_prompt_ids = vec![system_blob_id.clone()];
    if let Some(rules_blob_id) = rules_blob_id {
        root_prompt_ids.push(rules_blob_id);
    }
    for message in chat_request.messages.iter().take(current_user_index) {
        if message.role == Role::System {
            continue;
        }
        let Some(root_message) =
            encode_history_message(message, &tool_names_by_call_id, &failed_tool_call_ids)?
        else {
            continue;
        };
        root_prompt_ids.push(blob_store.insert(
            serde_json::to_vec(&root_message).map_err(|_| CursorRequestError::RequestTooLarge)?,
        )?);
    }

    let state = ConversationStateStructure {
        root_prompt_messages_json: root_prompt_ids,
        client_name: "grok-build".to_owned(),
        ..Default::default()
    };
    // Cursor reads this compact state shape from UserMessage.selected_context_blob
    // to find the root prompt and identify the client adapter.
    let selected_context_blob = blob_store.insert(
        ConversationStateStructure {
            root_prompt_messages_json: vec![system_blob_id],
            client_name: "grok-build".to_owned(),
            ..Default::default()
        }
        .encode_to_vec(),
    )?;
    let message_id = Uuid::new_v4().to_string();
    let action = ConversationAction {
        action: Some(conversation_action::Action::UserMessageAction(
            UserMessageAction {
                user_message: Some(UserMessage {
                    text: user_text,
                    message_id: message_id.clone(),
                    selected_context: Some(SelectedContext::default()),
                    mode: 1,
                    selected_context_blob,
                    correlation_id: message_id,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
    };
    let advertised_tool_definitions = mcp_tools.clone();
    let run_request = AgentRunRequest {
        conversation_state: Some(state),
        action: Some(action),
        requested_model: Some(RequestedModel {
            model_id: model_id.clone(),
            max_mode: model_route.as_ref().is_some_and(|route| route.max_mode),
            parameters: model_route
                .as_ref()
                .map(|route| {
                    route
                        .parameters
                        .iter()
                        .map(|parameter| RequestedModelModelParameterbytes {
                            id: parameter.id.clone(),
                            value: parameter.value.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            ..Default::default()
        }),
        mcp_tools: Some(McpTools { mcp_tools }),
        conversation_id: Some(conversation_id.clone()),
        ..Default::default()
    };
    let initial_payload = AgentClientMessage {
        message: Some(agent_client_message::Message::RunRequest(run_request)),
    }
    .encode_to_vec();
    if initial_payload.len() > MAX_RUN_REQUEST_BYTES {
        return Err(CursorRequestError::RequestTooLarge);
    }
    let initial_frame = encode_connect_frame(CONNECT_DATA_FRAME, &initial_payload)?;
    Ok(CursorRunPayload {
        initial_frame,
        blobs: blob_store.entries,
        conversation_id,
        model: model_id,
        advertised_tools,
        advertised_tool_definitions,
    })
}

/// Reject inputs whose bounded conversion could still amplify unbounded source
/// text, item counts, or schema recursion into large synchronous allocations.
fn preflight_cursor_run_request(request: &ConversationRequest) -> Result<(), CursorRequestError> {
    if request.items.len() > MAX_PREFLIGHT_CONVERSATION_ITEMS
        || request.tools.len() > MAX_TOOL_COUNT
    {
        return Err(CursorRequestError::RequestTooLarge);
    }

    let mut schema_nodes = 0;
    let mut tool_wire_bytes = 0_usize;
    for tool in &request.tools {
        let schema_bytes = estimate_proto_json_value(&tool.parameters, 0, &mut schema_nodes)?;
        if schema_bytes > MAX_TOOL_SCHEMA_BYTES {
            return Err(CursorRequestError::RequestTooLarge);
        }
        tool_wire_bytes = tool_wire_bytes
            .saturating_add(tool.name.len())
            .saturating_add(tool.description.as_ref().map_or(0, String::len))
            .saturating_add(schema_bytes)
            .saturating_add(64);
        if tool_wire_bytes > MAX_RUN_REQUEST_BYTES {
            return Err(CursorRequestError::RequestTooLarge);
        }
    }

    let mut writer = BoundedByteCounter {
        bytes: 0,
        limit: MAX_PREFLIGHT_CONVERSATION_BYTES,
    };
    for item in &request.items {
        serde_json::to_writer(&mut writer, item)
            .map_err(|_| CursorRequestError::RequestTooLarge)?;
        writer
            .write_all(b" ")
            .map_err(|_| CursorRequestError::RequestTooLarge)?;
    }
    for tool in &request.tools {
        serde_json::to_writer(&mut writer, tool)
            .map_err(|_| CursorRequestError::RequestTooLarge)?;
        writer
            .write_all(b" ")
            .map_err(|_| CursorRequestError::RequestTooLarge)?;
    }
    Ok(())
}

fn estimate_proto_json_value(
    value: &JsonValue,
    depth: usize,
    nodes: &mut usize,
) -> Result<usize, CursorRequestError> {
    if depth > MAX_TOOL_SCHEMA_DEPTH {
        return Err(CursorRequestError::RequestTooLarge);
    }
    *nodes = nodes.saturating_add(1);
    if *nodes > MAX_TOOL_SCHEMA_NODES {
        return Err(CursorRequestError::RequestTooLarge);
    }

    // Covers a Value field's protobuf tag/length and the map/list entry
    // wrappers added by json_to_proto_value, with payload strings counted by
    // their raw UTF-8 byte lengths. This is intentionally conservative.
    let mut bytes = 16_usize;
    match value {
        JsonValue::String(text) => bytes = bytes.saturating_add(text.len()),
        JsonValue::Array(values) => {
            for value in values {
                bytes = bytes.saturating_add(estimate_proto_json_value(value, depth + 1, nodes)?);
                if bytes > MAX_TOOL_SCHEMA_BYTES {
                    return Err(CursorRequestError::RequestTooLarge);
                }
            }
        }
        JsonValue::Object(values) => {
            for (key, value) in values {
                bytes = bytes.saturating_add(key.len()).saturating_add(12);
                bytes = bytes.saturating_add(estimate_proto_json_value(value, depth + 1, nodes)?);
                if bytes > MAX_TOOL_SCHEMA_BYTES {
                    return Err(CursorRequestError::RequestTooLarge);
                }
            }
        }
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) => {}
    }
    (bytes <= MAX_TOOL_SCHEMA_BYTES)
        .then_some(bytes)
        .ok_or(CursorRequestError::RequestTooLarge)
}

struct BoundedByteCounter {
    bytes: usize,
    limit: usize,
}

impl std::io::Write for BoundedByteCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let Some(bytes) = self.bytes.checked_add(buffer.len()) else {
            return Err(std::io::Error::other("serialized request too large"));
        };
        if bytes > self.limit {
            return Err(std::io::Error::other("serialized request too large"));
        }
        self.bytes = bytes;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn text_content(message: &ChatRequestMessage) -> Result<String, CursorRequestError> {
    match &message.content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Blocks(parts) => {
            let mut text = Vec::new();
            for part in parts {
                match part {
                    ChatContentBlock::Text { text: part } => text.push(part.as_str()),
                    ChatContentBlock::ImageUrl { .. } => {
                        return Err(CursorRequestError::UnsupportedImage);
                    }
                }
            }
            Ok(text.join("\n"))
        }
    }
}

fn encode_history_message(
    message: &ChatRequestMessage,
    tool_names_by_call_id: &HashMap<String, String>,
    failed_tool_call_ids: &HashSet<String>,
) -> Result<Option<JsonValue>, CursorRequestError> {
    match message.role {
        Role::System => Ok(None),
        Role::User => Ok(Some(json!({
            "role": "user",
            "content": [{ "type": "text", "text": format!("<user_query>\n{}\n</user_query>", text_content(message)?) }]
        }))),
        Role::Assistant => {
            let mut content = Vec::new();
            let text = text_content(message)?;
            if !text.is_empty() {
                content.push(json!({ "type": "text", "text": text }));
            }
            for call in &message.tool_calls {
                let call_id = call
                    .id
                    .as_deref()
                    .ok_or(CursorRequestError::MissingToolCallId)?;
                let name = cursor_history_tool_name(&call.function.name);
                let args: JsonValue = serde_json::from_str(&call.function.arguments)
                    .map_err(|_| CursorRequestError::InvalidToolSchema)?;
                content.push(json!({
                    "type": "tool-call",
                    "toolCallId": call_id,
                    "toolName": name,
                    "args": args,
                }));
            }
            if content.is_empty() {
                Ok(None)
            } else {
                Ok(Some(json!({ "role": "assistant", "content": content })))
            }
        }
        Role::Tool => {
            let call_id = message
                .tool_call_id
                .as_deref()
                .ok_or(CursorRequestError::MissingToolCallId)?;
            let tool_name = tool_names_by_call_id
                .get(call_id)
                .cloned()
                .unwrap_or_else(|| "tool".to_owned());
            Ok(Some(json!({
                "role": "tool",
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "toolName": cursor_history_tool_name(&tool_name),
                    "result": text_content(message)?,
                    "isError": failed_tool_call_ids.contains(call_id),
                }]
            })))
        }
    }
}

fn cursor_history_tool_name(name: &str) -> String {
    let name = name.trim();
    if name.starts_with("mcp_grok_") {
        name.to_owned()
    } else {
        format!("mcp_grok_{name}")
    }
}

fn json_to_proto_value(value: &JsonValue) -> Value {
    let kind = match value {
        JsonValue::Null => Kind::NullValue(NullValue::NullValue as i32),
        JsonValue::Bool(value) => Kind::BoolValue(*value),
        JsonValue::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        JsonValue::String(value) => Kind::StringValue(value.clone()),
        JsonValue::Array(values) => Kind::ListValue(ListValue {
            values: values.iter().map(json_to_proto_value).collect(),
        }),
        JsonValue::Object(values) => Kind::StructValue(Struct {
            fields: values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                .collect(),
        }),
    };
    Value { kind: Some(kind) }
}

#[derive(Default)]
struct BlobStore {
    entries: HashMap<Vec<u8>, Vec<u8>>,
    total_bytes: usize,
}

impl BlobStore {
    fn insert(&mut self, bytes: Vec<u8>) -> Result<Vec<u8>, CursorRequestError> {
        if bytes.len() > MAX_BLOB_BYTES
            || self.entries.len() >= MAX_BLOB_COUNT
            || self.total_bytes.saturating_add(bytes.len()) > MAX_BLOB_STORE_BYTES
        {
            return Err(CursorRequestError::RequestTooLarge);
        }
        let id = Sha256::digest(&bytes).to_vec();
        self.total_bytes += bytes.len();
        self.entries.entry(id.clone()).or_insert(bytes);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor_connect::ConnectFrameDecoder;
    fn request() -> ConversationRequest {
        ConversationRequest {
            items: vec![
                xai_grok_sampling_types::ConversationItem::System(
                    xai_grok_sampling_types::conversation::SystemItem {
                        content: "Follow the repo rules".into(),
                        synthetic_reason:
                            xai_grok_sampling_types::conversation::SyntheticReason::Primary,
                    },
                ),
                xai_grok_sampling_types::ConversationItem::User(
                    xai_grok_sampling_types::conversation::UserItem {
                        content: vec![xai_grok_sampling_types::conversation::ContentPart::Text {
                            text: "Earlier question".into(),
                        }],
                        ..Default::default()
                    },
                ),
                xai_grok_sampling_types::ConversationItem::Assistant(
                    xai_grok_sampling_types::conversation::AssistantItem {
                        content: "Earlier answer".into(),
                        tool_calls: vec![],
                        model_id: None,
                        model_fingerprint: None,
                        reasoning_effort: None,
                    },
                ),
                xai_grok_sampling_types::ConversationItem::User(
                    xai_grok_sampling_types::conversation::UserItem {
                        content: vec![xai_grok_sampling_types::conversation::ContentPart::Text {
                            text: "Current question".into(),
                        }],
                        ..Default::default()
                    },
                ),
            ],
            model: Some("cursor:gpt-5.6".to_owned()),
            x_grok_session_id: Some("session-1".to_owned()),
            ..Default::default()
        }
    }

    #[test]
    fn preflight_rejects_tool_schema_depth_before_proto_conversion() {
        let mut parameters = JsonValue::Null;
        for _ in 0..=MAX_TOOL_SCHEMA_DEPTH {
            parameters = JsonValue::Array(vec![parameters]);
        }
        let mut request = request();
        request.tools = vec![xai_grok_sampling_types::conversation::ToolSpec {
            name: "deep_schema".to_owned(),
            description: None,
            parameters,
        }];
        assert_eq!(
            preflight_cursor_run_request(&request),
            Err(CursorRequestError::RequestTooLarge)
        );
    }

    #[test]
    fn preflight_rejects_oversized_conversation_without_buffering_json() {
        let mut request = request();
        request.items.push(ConversationItem::System(
            xai_grok_sampling_types::conversation::SystemItem {
                content: std::sync::Arc::<str>::from(
                    "x".repeat(MAX_PREFLIGHT_CONVERSATION_BYTES + 1),
                ),
                synthetic_reason: xai_grok_sampling_types::conversation::SyntheticReason::Primary,
            },
        ));
        assert_eq!(
            preflight_cursor_run_request(&request),
            Err(CursorRequestError::RequestTooLarge)
        );
    }

    fn decode_run(payload: &CursorRunPayload) -> AgentRunRequest {
        let mut decoder = ConnectFrameDecoder::default();
        let frame = decoder
            .push(&payload.initial_frame)
            .expect("frame")
            .remove(0);
        decoder.finish().expect("complete frame");
        let client_message = AgentClientMessage::decode(frame.payload.as_slice()).expect("message");
        let Some(agent_client_message::Message::RunRequest(request)) = client_message.message
        else {
            panic!("expected RunRequest");
        };
        request
    }

    #[test]
    fn builds_connect_run_with_rules_history_and_current_user_action() {
        let payload = build_cursor_run_payload(request()).expect("build request");
        let run = decode_run(&payload);
        assert_eq!(run.conversation_id.as_deref(), Some("session-1"));
        assert_eq!(
            run.requested_model.as_ref().unwrap().model_id,
            "cursor:gpt-5.6"
        );
        assert_eq!(run.mcp_tools.as_ref().unwrap().mcp_tools.len(), 0);
        let state = run.conversation_state.as_ref().unwrap();
        assert_eq!(state.root_prompt_messages_json.len(), 4);
        assert_eq!(state.client_name, "grok-build");
        let Some(conversation_action::Action::UserMessageAction(action)) =
            run.action.unwrap().action
        else {
            panic!("expected user action");
        };
        let user = action.user_message.unwrap();
        assert_eq!(user.text, "Current question");
        let selected_context_state = payload
            .blobs
            .get(&user.selected_context_blob)
            .expect("selected context state blob is stored by its content hash");
        assert_eq!(
            Sha256::digest(selected_context_state).as_slice(),
            user.selected_context_blob
        );
        let selected_context_state =
            ConversationStateStructure::decode(selected_context_state.as_slice())
                .expect("selected context state");
        assert_eq!(selected_context_state.client_name, "grok-build");
        assert_eq!(selected_context_state.root_prompt_messages_json.len(), 1);
        let system_id = &state.root_prompt_messages_json[0];
        let system = payload.blobs.get(system_id).expect("system blob");
        assert!(
            std::str::from_utf8(system)
                .unwrap()
                .contains("\"role\":\"system\"")
        );
        let rules = payload
            .blobs
            .get(&state.root_prompt_messages_json[1])
            .unwrap();
        assert!(std::str::from_utf8(rules).unwrap().contains("<rules>"));
        assert!(payload.blobs.values().any(|blob| {
            std::str::from_utf8(blob)
                .is_ok_and(|text| text.contains("Earlier question") && text.contains("user_query"))
        }));
    }

    #[test]
    fn encodes_only_advertised_tools_as_cursor_mcp_tools() {
        let mut request = request();
        request
            .tools
            .push(xai_grok_sampling_types::conversation::ToolSpec {
                name: "read_file".to_owned(),
                description: Some("Read a file".to_owned()),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            });
        let payload = build_cursor_run_payload(request).expect("build request");
        let run = decode_run(&payload);
        let tools = &run.mcp_tools.unwrap().mcp_tools;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].provider_identifier, "grok");
        assert_eq!(
            payload.advertised_tools,
            HashSet::from(["read_file".to_owned()])
        );
        assert_eq!(tools[0].tool_name, "read_file");
        let schema = Value::decode(tools[0].input_schema.as_slice()).expect("schema Value");
        assert!(matches!(schema.kind, Some(Kind::StructValue(_))));
    }

    #[test]
    fn sends_the_exact_catalog_selected_model_parameter_route() {
        let route = CursorModelRoute {
            model_id: "gpt-5.6".to_owned(),
            parameters: vec![crate::cursor_wire::CursorModelParameter {
                id: "reasoning".to_owned(),
                value: "high".to_owned(),
            }],
            max_mode: true,
        };
        let mut request = request();
        request.model = Some(route.to_model_string());
        let payload = build_cursor_run_payload(request).expect("build request");
        let run = decode_run(&payload);
        let requested = run.requested_model.expect("requested model");
        assert_eq!(requested.model_id, "gpt-5.6");
        assert!(requested.max_mode);
        assert_eq!(
            requested.parameters,
            vec![RequestedModelModelParameterbytes {
                id: "reasoning".to_owned(),
                value: "high".to_owned(),
            }]
        );
        assert_eq!(payload.model, "gpt-5.6");
    }

    #[test]
    fn history_preserves_grok_mcp_names_and_tool_failure_status() {
        let mut request = request();
        let call = xai_grok_sampling_types::ToolCall {
            id: std::sync::Arc::<str>::from("read-call-1"),
            name: "read_file".to_owned(),
            arguments: std::sync::Arc::<str>::from(r#"{"path":"README.md"}"#),
        };
        request
            .items
            .insert(3, ConversationItem::assistant_tool_calls(vec![call]));
        request.items.insert(
            4,
            ConversationItem::tool_result_error("read-call-1", "denied"),
        );

        let payload = build_cursor_run_payload(request).expect("build request");
        let history = payload
            .blobs
            .values()
            .filter_map(|blob| serde_json::from_slice::<JsonValue>(blob).ok())
            .find(|message| {
                message.to_string().contains("read-call-1")
                    && message.to_string().contains("tool-result")
            })
            .expect("historical tool result blob");
        assert!(history.to_string().contains("mcp_grok_read_file"));
        assert!(history.to_string().contains("\"isError\":true"));
    }

    #[test]
    fn rejects_missing_session_and_unsupported_options() {
        let mut missing_session = request();
        missing_session.x_grok_session_id = None;
        assert_eq!(
            build_cursor_run_payload(missing_session).unwrap_err(),
            CursorRequestError::MissingSessionId
        );

        let mut temperature = request();
        temperature.temperature = Some(0.2);
        assert_eq!(
            build_cursor_run_payload(temperature).unwrap_err(),
            CursorRequestError::UnsupportedSamplingOption
        );
    }

    #[test]
    fn distinct_conv_id_becomes_the_cursor_wire_conversation() {
        let mut request = request();
        request.x_grok_conv_id = Some("oneshot-compact-1".to_owned());
        let payload = build_cursor_run_payload(request).expect("build request");
        let run = decode_run(&payload);
        assert_eq!(run.conversation_id.as_deref(), Some("oneshot-compact-1"));
        assert_eq!(payload.conversation_id, "oneshot-compact-1");
    }
}
