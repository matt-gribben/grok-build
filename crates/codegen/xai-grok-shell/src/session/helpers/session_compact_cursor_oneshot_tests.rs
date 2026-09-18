use super::*;
use xai_grok_sampling_types::LengthPolicy;

#[test]
fn cursor_oneshot_compact_request_is_tool_free_with_distinct_conv_id() {
    let request = cursor_oneshot_compact_request(
        vec![ConversationItem::user("summarize")],
        "session-1",
        "composer-2",
    );

    assert!(
        request.tools.is_empty(),
        "Cursor compact must not advertise tools"
    );
    assert!(request.hosted_tools.is_empty());
    assert!(matches!(
        request.tool_choice,
        Some(ConversationToolChoice::None)
    ));
    assert_eq!(request.temperature, None);
    assert_eq!(request.top_p, None);
    assert_eq!(request.reasoning_effort, None);
    assert_eq!(request.json_schema, None);
    assert_eq!(request.model.as_deref(), Some("composer-2"));
    assert_eq!(request.x_grok_session_id.as_deref(), Some("session-1"));
    assert_eq!(request.length_policy, LengthPolicy::Fail);

    let conv_id = request
        .x_grok_conv_id
        .expect("distinct Cursor conversation");
    assert!(conv_id.starts_with("compact-"), "{conv_id}");
    assert_ne!(conv_id, "session-1");
}
