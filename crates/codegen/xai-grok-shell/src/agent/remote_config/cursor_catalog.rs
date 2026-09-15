//! Opt-in discovery for models available through a signed-in Cursor Desktop session.

use indexmap::IndexMap;
use xai_grok_sampler::{
    cursor_catalog::{CursorCatalogClient, CursorDiscoveredModel},
    cursor_request::CursorModelRoute,
};
use xai_grok_sampling_types::ApiBackend as SamplingApiBackend;

use crate::agent::config::{Config, EndpointsConfig, ModelEntry};

const MAX_CURSOR_VARIANT_ENTRIES: usize = 8_192;

/// Cursor's local credential is consulted only when a trusted model/provider
/// entry explicitly selects the Cursor backend.
pub(super) fn is_cursor_provider_enabled() -> bool {
    let raw = match crate::util::config::load_effective_config() {
        Ok(raw) => raw,
        Err(_) => return false,
    };
    let config = match Config::new_from_toml_cfg(&raw) {
        Ok(config) => config,
        Err(_) => return false,
    };
    cursor_backend_enabled(&config)
}

/// Fetch Cursor's account-specific model catalog without writing it to the
/// shared Grok model cache. The caller runs this on the existing prefetch
/// worker thread, so the small current-thread runtime owns the HTTP connection.
pub(super) fn fetch_cursor_models(
    endpoints: &EndpointsConfig,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Option<IndexMap<String, ModelEntry>> {
    let access_token =
        match xai_grok_login::cursor_credentials::resolve_cursor_desktop_access_token() {
            Ok(token) => token,
            Err(error) => {
                tracing::info!(%error, "Cursor model discovery skipped: no usable Desktop session");
                return None;
            }
        };
    let client = match CursorCatalogClient::new() {
        Ok(client) => client,
        Err(error) => {
            tracing::warn!(%error, "Cursor model discovery could not initialize its trusted endpoint");
            return None;
        }
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::warn!(%error, "Cursor model discovery runtime could not be initialized");
            return None;
        }
    };
    let catalog = match runtime
        .block_on(client.discover_models(access_token.expose_secret(), cancellation))
    {
        Ok(catalog) => catalog,
        Err(error) => {
            tracing::info!(%error, "Cursor model discovery did not return an account catalog");
            return None;
        }
    };

    let models = build_cursor_model_entries(catalog.models, endpoints);
    (!models.is_empty()).then_some(models)
}

fn build_cursor_model_entries(
    discovered: Vec<CursorDiscoveredModel>,
    endpoints: &EndpointsConfig,
) -> IndexMap<String, ModelEntry> {
    let mut models = IndexMap::with_capacity(discovered.len().saturating_add(64));
    let mut variant_entries = 0;
    for model in discovered {
        if model.id.len() > 256 || model.id.chars().any(char::is_control) {
            tracing::warn!("Cursor model discovery skipped an invalid model identifier");
            continue;
        }
        let key = format!("cursor/{}", model.id);
        let mut entry = ModelEntry::fallback(&model.id, endpoints);
        entry.info.id = Some(key.clone());
        entry.info.model = CursorModelRoute {
            model_id: model.id.clone(),
            parameters: Vec::new(),
            max_mode: model.max_mode,
        }
        .to_model_string();
        entry.info.model_family = Some("cursor".to_owned());
        // This backend routes through Cursor's validated RPC client. It has no
        // REST base URL and must never inherit one as a credential destination.
        entry.info.base_url.clear();
        entry.info.api_backend = SamplingApiBackend::Cursor;
        entry.info.name = Some(format!("Cursor: {}", sanitize_display_name(&model.name)));
        entry.info.description = Some(if model.context_window_is_inferred {
            "Cursor subscription model. Context window is estimated; output limit is a local budget."
                .to_owned()
        } else {
            "Cursor subscription model. Output limit is a local budget.".to_owned()
        });
        entry.info.context_window = std::num::NonZeroU64::new(model.context_window)
            .unwrap_or_else(|| std::num::NonZeroU64::new(200_000).expect("nonzero fallback"));
        entry.info.max_completion_tokens = Some(model.max_output_tokens);
        entry.info.reasoning_effort = None;
        entry.info.supports_reasoning_effort = false;
        entry.info.reasoning_efforts.clear();
        entry.info.variants.clear();
        entry.info.supports_backend_search = false;
        entry.info.supported_in_api = true;
        entry.api_key = None;
        entry.env_key = None;
        entry.auth_provider = None;
        entry.api_base_url = None;
        models.insert(key.clone(), entry.clone());

        for (index, variant) in model.variants.iter().enumerate() {
            if variant_entries >= MAX_CURSOR_VARIANT_ENTRIES {
                break;
            }
            if variant.parameters.is_empty() && !variant.is_max_mode {
                continue;
            }
            let mut variant_entry = entry.clone();
            let variant_key = format!("{key}/variant/{index}");
            let label = variant
                .display_name_outside_picker
                .as_deref()
                .or(variant.display_name.as_deref())
                .filter(|name| !name.trim().is_empty())
                .map(sanitize_display_name)
                .unwrap_or_else(|| format!("Variant {}", index + 1));
            let mode_suffix = if variant.is_max_mode {
                " · Max Mode"
            } else {
                ""
            };
            variant_entry.info.id = Some(variant_key.clone());
            variant_entry.info.model = CursorModelRoute {
                model_id: model.id.clone(),
                parameters: variant.parameters.clone(),
                max_mode: variant.is_max_mode,
            }
            .to_model_string();
            variant_entry.info.name = Some(format!(
                "Cursor: {} ({label}{mode_suffix})",
                sanitize_display_name(&model.name)
            ));
            variant_entry.info.context_window = std::num::NonZeroU64::new(variant_context_window(
                model.context_window,
                model
                    .max_mode_context_window
                    .unwrap_or(model.context_window),
                &variant.parameters,
                variant.is_max_mode,
            ))
            .unwrap_or_else(|| std::num::NonZeroU64::new(200_000).expect("nonzero fallback"));
            models.insert(variant_key, variant_entry);
            variant_entries += 1;
        }
    }
    models
}

fn sanitize_display_name(name: &str) -> String {
    name.chars()
        .filter(|character| !character.is_control())
        .take(160)
        .collect::<String>()
        .trim()
        .to_owned()
}

fn variant_context_window(
    fallback: u64,
    max_mode_fallback: u64,
    parameters: &[xai_grok_sampler::cursor_wire::CursorModelParameter],
    max_mode: bool,
) -> u64 {
    let Some(context) = parameters
        .iter()
        .find(|parameter| parameter.id.eq_ignore_ascii_case("context"))
        .map(|parameter| parameter.value.trim().to_ascii_lowercase())
    else {
        return if max_mode {
            max_mode_fallback
        } else {
            fallback
        };
    };
    for (suffix, multiplier) in [("m", 1_000_000_u64), ("k", 1_000_u64)] {
        if let Some(number) = context.strip_suffix(suffix)
            && let Ok(number) = number.parse::<u64>()
        {
            return number.saturating_mul(multiplier);
        }
    }
    if max_mode {
        max_mode_fallback
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_catalog_requires_the_app_settings_opt_in() {
        let app_setting: toml::Value = toml::from_str(
            r#"
            [ui]
            cursor_provider_enabled = true
            "#,
        )
        .unwrap();
        let app_setting_config = Config::new_from_toml_cfg(&app_setting).unwrap();
        assert!(cursor_backend_enabled(&app_setting_config));

        let disabled_setting: toml::Value = toml::from_str(
            r#"
            [ui]
            cursor_provider_enabled = false
            "#,
        )
        .unwrap();
        let disabled_setting_config = Config::new_from_toml_cfg(&disabled_setting).unwrap();
        assert!(!cursor_backend_enabled(&disabled_setting_config));

        let provider: toml::Value = toml::from_str(
            r#"
            [model_providers.cursor]
            api_backend = "cursor"
            "#,
        )
        .unwrap();
        let provider_config = Config::new_from_toml_cfg(&provider).unwrap();
        assert!(!cursor_backend_enabled(&provider_config));

        let model: toml::Value = toml::from_str(
            r#"
            [model.my-cursor-model]
            model = "some-cursor-model"
            api_backend = "cursor"
            "#,
        )
        .unwrap();
        let model_config = Config::new_from_toml_cfg(&model).unwrap();
        assert!(!cursor_backend_enabled(&model_config));

        let unrelated: toml::Value = toml::from_str(
            r#"
            [model_providers.openai]
            api_backend = "responses"
            "#,
        )
        .unwrap();
        let unrelated_config = Config::new_from_toml_cfg(&unrelated).unwrap();
        assert!(!cursor_backend_enabled(&unrelated_config));
    }

    #[test]
    fn display_name_removes_controls_and_has_a_bounded_length() {
        assert_eq!(sanitize_display_name(" Model\n\u{1b}[31m "), "Model[31m");
        assert_eq!(sanitize_display_name(&"x".repeat(200)).len(), 160);
    }

    #[test]
    fn max_mode_variant_uses_catalog_max_context_when_no_context_parameter_exists() {
        assert_eq!(variant_context_window(256_000, 500_000, &[], true), 500_000);
        assert_eq!(
            variant_context_window(256_000, 500_000, &[], false),
            256_000
        );
        assert_eq!(
            variant_context_window(
                256_000,
                500_000,
                &[xai_grok_sampler::cursor_wire::CursorModelParameter {
                    id: "context".to_owned(),
                    value: "1m".to_owned(),
                }],
                true,
            ),
            1_000_000,
        );
    }

    #[test]
    fn picker_entries_preserve_base_and_variant_cursor_routes() {
        let endpoints = EndpointsConfig::default();
        let discovered = CursorDiscoveredModel {
            id: "fable".to_owned(),
            name: "Fable".to_owned(),
            aliases: Vec::new(),
            reasoning: true,
            supports_images: Some(true),
            supports_max_mode: Some(true),
            max_mode: true,
            context_window: 500_000,
            max_mode_context_window: Some(500_000),
            context_window_is_inferred: false,
            max_output_tokens: 16_000,
            variants: vec![xai_grok_sampler::cursor_wire::CursorParameterizedVariant {
                parameters: vec![xai_grok_sampler::cursor_wire::CursorModelParameter {
                    id: "reasoning".to_owned(),
                    value: "high".to_owned(),
                }],
                is_max_mode: true,
                ..Default::default()
            }],
        };
        let entries = build_cursor_model_entries(vec![discovered], &endpoints);
        let base = entries
            .get("cursor/fable")
            .expect("base model is selectable");
        assert_eq!(base.info.api_backend, SamplingApiBackend::Cursor);
        assert!(base.info.supported_in_api);
        let base_route = CursorModelRoute::from_model_string(&base.info.model)
            .expect("parse base route")
            .expect("Cursor route");
        assert!(base_route.max_mode);
        assert_eq!(base.info.context_window.get(), 500_000);

        let variant = entries
            .get("cursor/fable/variant/0")
            .expect("parameter variant is selectable");
        let variant_route = CursorModelRoute::from_model_string(&variant.info.model)
            .expect("parse variant route")
            .expect("Cursor variant route");
        assert!(variant_route.max_mode);
        assert_eq!(variant_route.parameters[0].value, "high");
        assert_eq!(variant.info.context_window.get(), 500_000);
    }
}

fn cursor_backend_enabled(config: &Config) -> bool {
    config.ui.cursor_provider_enabled == Some(true)
}
