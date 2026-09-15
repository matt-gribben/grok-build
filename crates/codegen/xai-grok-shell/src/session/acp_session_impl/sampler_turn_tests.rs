use xai_grok_sampling_types::{SearchDateBound, ToolOverrides, WebSearchOptions, XSearchOptions};

use super::{
    CLASSIFIER_REQUEST_TOKEN_RESERVE, CursorDesktopBearerResolver, LengthSalvageAction,
    LengthSalvageStreak, MAX_OUTPUT_TOKEN_LIMIT_RETRIES, classifier_request_fits_context,
    resolve_configured_cutoff,
};

fn x_cut(to: &str) -> XSearchOptions {
    XSearchOptions {
        date_bound: Some(SearchDateBound::new(None, Some(to.into())).unwrap()),
    }
}

/// When every sample ends with finish reason Length while tools are active, the turn salvages up to the cap and then fails.
/// The reminder is injected only on the first salvage.
#[test]
fn length_salvage_streak_proceeds_to_the_cap_then_exhausts() {
    let mut streak = LengthSalvageStreak::default();
    for n in 1..=MAX_OUTPUT_TOKEN_LIMIT_RETRIES {
        match streak.on_sample(true) {
            LengthSalvageAction::Proceed { inject_reminder } => {
                assert_eq!(
                    inject_reminder,
                    n == 1,
                    "reminder only on the first salvage"
                );
            }
            other => panic!("salvage {n} within the cap must proceed, got {other:?}"),
        }
    }
    assert!(matches!(
        streak.on_sample(true),
        LengthSalvageAction::Exhausted
    ));
}

/// A non-Length (or tool-less) sample resets the streak: a later salvage starts a fresh streak and re-injects the reminder.
#[test]
fn length_salvage_streak_resets_on_non_length_sample() {
    let mut streak = LengthSalvageStreak::default();
    for _ in 0..MAX_OUTPUT_TOKEN_LIMIT_RETRIES {
        let _ = streak.on_sample(true);
    }
    assert!(matches!(
        streak.on_sample(false),
        LengthSalvageAction::NotSalvage
    ));
    assert!(matches!(
        streak.on_sample(true),
        LengthSalvageAction::Proceed {
            inject_reminder: true
        }
    ));
}

#[test]
fn classifier_request_bound_enforces_its_reserve_with_saturating_arithmetic() {
    let window = 12_000 + CLASSIFIER_REQUEST_TOKEN_RESERVE;
    for (input, context_window, expected) in [
        (12_000, window, true),
        (12_001, window, false),
        (u64::MAX, u64::MAX, false),
    ] {
        assert_eq!(
            classifier_request_fits_context(input, context_window),
            expected
        );
    }
}

#[test]
fn seed_cutoff_is_inherited_without_a_per_turn_update() {
    let seed = ToolOverrides {
        x_search: Some(x_cut("2020-01-01")),
        web_search: None,
    };
    assert_eq!(resolve_configured_cutoff(Some(seed.clone()), None), seed);
}

#[test]
fn non_empty_base_cutoff_wins_per_tool_and_an_empty_one_reverts_to_the_seed() {
    let seed = ToolOverrides {
        x_search: Some(x_cut("2020-01-01")),
        web_search: Some(WebSearchOptions {
            allowed_domains: Some(vec!["x.com".into()]),
            excluded_domains: None,
        }),
    };
    let base = ToolOverrides {
        x_search: Some(x_cut("2019-06-01")),
        web_search: Some(WebSearchOptions {
            allowed_domains: Some(vec![]),
            excluded_domains: None,
        }),
    };
    let got = resolve_configured_cutoff(Some(seed.clone()), Some(&base));
    assert_eq!(got.x_search, Some(x_cut("2019-06-01")));
    assert_eq!(got.web_search, seed.web_search);
}

/// Explicit live acceptance path for the active Cursor Desktop session.
/// This reads the local access token only when invoked with `--ignored`, sends
/// one short prompt through Cursor's Run stream, and never logs the credential.
#[tokio::test]
#[ignore = "contacts Cursor inference using the signed-in Desktop session"]
async fn live_cursor_subscription_smoke_uses_stored_desktop_session() {
    use std::time::{Duration, Instant};

    use tokio_util::sync::CancellationToken;
    use xai_grok_login::cursor_credentials::resolve_cursor_desktop_access_token;
    use xai_grok_sampler::{
        ApiBackend, RetryPolicy, SamplerActor, SamplerConfig, cursor_catalog::CursorCatalogClient,
        cursor_request::CursorModelRoute,
    };
    use xai_grok_sampling_types::conversation::UserItem;
    use xai_grok_sampling_types::{ContentPart, ConversationItem, ConversationRequest};

    let credential = resolve_cursor_desktop_access_token()
        .expect("a signed-in Cursor Desktop access token is available");
    let catalog = CursorCatalogClient::new()
        .expect("Cursor's configured regional endpoint is trusted")
        .discover_models(credential.expose_secret(), &CancellationToken::new())
        .await
        .expect("Cursor returns a usable account model catalog");
    let selected = catalog
        .models
        .iter()
        .find(|model| model.id.to_ascii_lowercase().contains("composer-2"))
        .or_else(|| {
            catalog
                .models
                .iter()
                .find(|model| !model.id.to_ascii_lowercase().contains("auto"))
        })
        .or_else(|| catalog.models.first())
        .expect("Cursor's account catalog includes a usable model");
    let variant = selected
        .variants
        .iter()
        .find(|variant| variant.is_default_non_max_config == Some(true))
        .or_else(|| {
            selected
                .variants
                .iter()
                .find(|variant| !variant.is_max_mode)
        });
    let route = CursorModelRoute {
        model_id: selected.id.clone(),
        parameters: variant.map_or_else(Vec::new, |variant| variant.parameters.clone()),
        max_mode: false,
    }
    .to_model_string();
    let config = SamplerConfig {
        model: route.clone(),
        api_backend: ApiBackend::Cursor,
        max_completion_tokens: Some(24),
        max_retries: Some(0),
        bearer_resolver: Some(std::sync::Arc::new(CursorDesktopBearerResolver::default())),
        ..SamplerConfig::default()
    };
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let sampler = SamplerActor::spawn(config, RetryPolicy::default(), event_tx);
    let request = ConversationRequest {
        items: vec![ConversationItem::User(UserItem {
            content: vec![ContentPart::Text {
                text: "Reply with only the word OK. Do not call tools.".into(),
            }],
            ..UserItem::default()
        })],
        model: Some(route),
        max_output_tokens: Some(24),
        x_grok_session_id: Some(uuid::Uuid::new_v4().to_string()),
        ..ConversationRequest::default()
    };

    let started = Instant::now();
    let collected = tokio::time::timeout(
        Duration::from_secs(90),
        sampler.submit_and_collect(xai_grok_sampler::RequestId::random(), request),
    )
    .await
    .expect("Cursor inference completes within 90 seconds");
    let (response, _) = collected.unwrap_or_else(|_| {
        panic!("Cursor inference failed; provider error details are intentionally suppressed")
    });
    let assistant = response
        .assistant()
        .expect("Cursor returns an assistant response");
    assert!(
        !assistant.content.trim().is_empty(),
        "Cursor returns non-empty text"
    );
    eprintln!(
        "Cursor live smoke passed: model={}, elapsed_ms={}",
        selected.id,
        started.elapsed().as_millis()
    );
}

#[test]
fn inherited_cutoff_agrees_with_the_wire_echo_so_the_two_implementations_cannot_drift() {
    use xai_grok_sampling_types::{HostedTool, apply_tool_overrides};
    let web = WebSearchOptions {
        allowed_domains: Some(vec!["x.com".into()]),
        excluded_domains: None,
    };
    let cases = [
        (
            Some(ToolOverrides {
                x_search: Some(x_cut("2020-01-01")),
                web_search: None,
            }),
            None,
        ),
        (
            Some(ToolOverrides {
                x_search: Some(x_cut("2020-01-01")),
                web_search: Some(web.clone()),
            }),
            Some(ToolOverrides {
                x_search: Some(x_cut("2019-06-01")),
                web_search: None,
            }),
        ),
        (
            None,
            Some(ToolOverrides {
                x_search: Some(x_cut("2018-01-01")),
                web_search: Some(web.clone()),
            }),
        ),
    ];
    for (seed, base) in cases {
        let mut tools = vec![
            HostedTool::WebSearch { options: None },
            HostedTool::XSearch { options: None },
        ];
        apply_tool_overrides(&mut tools, seed.as_ref());
        let wire_echo = apply_tool_overrides(&mut tools, base.as_ref());
        let inherited = resolve_configured_cutoff(seed.clone(), base.as_ref());
        assert_eq!(wire_echo, inherited, "seed={seed:?} base={base:?}");
    }
}

#[cfg(test)]
mod subagent_sampling_gate_tests {
    use super::super::super::support::create_test_actor;
    use super::super::acquire_subagent_sampling_permit;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Semaphore;

    #[derive(Default)]
    struct ConcurrencyProbe {
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
    }

    impl ConcurrencyProbe {
        fn enter(&self) {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        }
        fn leave(&self) {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn subagent_submits_never_exceed_cap_and_excess_queues() {
        const CAP: usize = 3;
        const TURNS: usize = 12;
        let semaphore = Arc::new(Semaphore::new(CAP));
        let probe = Arc::new(ConcurrencyProbe::default());
        let ran = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..TURNS {
            let gate = Some(semaphore.clone());
            let probe = probe.clone();
            let ran = ran.clone();
            handles.push(tokio::spawn(async move {
                let permit = acquire_subagent_sampling_permit(&gate).await;
                assert!(permit.is_some(), "a subagent turn must receive a permit");
                probe.enter();
                tokio::time::sleep(Duration::from_millis(20)).await;
                probe.leave();
                ran.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(
            ran.load(Ordering::SeqCst),
            TURNS,
            "every queued turn ran (queued, not errored)"
        );
        assert!(
            probe.max_in_flight.load(Ordering::SeqCst) <= CAP,
            "in-flight subagent submits exceeded the cap: {} > {CAP}",
            probe.max_in_flight.load(Ordering::SeqCst),
        );
    }

    #[tokio::test]
    async fn cancelled_waiter_releases_without_deadlock() {
        let semaphore = Arc::new(Semaphore::new(1));
        let gate = Some(semaphore.clone());
        let held = acquire_subagent_sampling_permit(&gate).await;
        assert!(held.is_some());

        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move {
                let _permit = acquire_subagent_sampling_permit(&gate).await;
                std::future::pending::<()>().await;
            }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            semaphore.available_permits(),
            0,
            "slot stays held while the second turn queues"
        );
        waiter.abort();
        let _ = waiter.await;

        drop(held);
        let next = tokio::time::timeout(
            Duration::from_millis(200),
            acquire_subagent_sampling_permit(&gate),
        )
        .await
        .expect("a permit must be free once the held one is released");
        assert!(next.is_some(), "the cancelled waiter did not leak the slot");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn submit_holds_permit_for_subagent_not_main() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let saturated = Arc::new(Semaphore::new(1));
                let _held = saturated.clone().acquire_owned().await.unwrap();

                let (gw_tx, _gw_rx) = tokio::sync::mpsc::unbounded_channel();
                let (p_tx, _p_rx) = tokio::sync::mpsc::unbounded_channel();
                let mut subagent = create_test_actor(0, 200_000, 80, gw_tx, p_tx).await;
                subagent.sampling_gate = Some(saturated.clone());
                let subagent = Arc::new(subagent);

                let queued = tokio::time::timeout(
                    Duration::from_millis(150),
                    subagent.submit_turn_request(Default::default()),
                )
                .await;
                assert!(
                    queued.is_err(),
                    "a subagent submit must queue behind the drained gate, never reaching the sampler"
                );

                let (gw_tx, _gw_rx) = tokio::sync::mpsc::unbounded_channel();
                let (p_tx, _p_rx) = tokio::sync::mpsc::unbounded_channel();
                let main = create_test_actor(0, 200_000, 80, gw_tx, p_tx).await;
                assert!(main.sampling_gate.is_none());
                let main = Arc::new(main);

                let ran = tokio::time::timeout(
                    Duration::from_millis(150),
                    main.submit_turn_request(Default::default()),
                )
                .await;
                assert!(
                    ran.is_ok(),
                    "the main session must reach the sampler even while the gate is drained"
                );
                assert!(
                    main.turn_stream_drained.lock().is_empty(),
                    "a result with no queued terminal event must release request ownership before returning"
                );
            })
            .await;
    }
}
