//! For a token that is not a JWT, `parse_jwt_expiration` returns `None` and `is_jwt_expired_or_near` returns `false`.

use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;

#[derive(Deserialize)]
struct Claims {
    exp: Option<serde_json::Number>,
}

pub fn parse_jwt_expiration(token: &str) -> Option<DateTime<Utc>> {
    jsonwebtoken::dangerous::insecure_decode::<Claims>(token)
        .ok()
        .and_then(|data| data.claims.exp)
        .and_then(|exp| {
            if let Some(seconds) = exp.as_i64() {
                return DateTime::from_timestamp(seconds, 0);
            }

            let timestamp = exp.as_f64()?;
            if !timestamp.is_finite() {
                return None;
            }
            let whole_seconds = timestamp.floor();
            if whole_seconds < i64::MIN as f64 || whole_seconds > i64::MAX as f64 {
                return None;
            }
            let mut seconds = whole_seconds as i64;
            let mut nanos = ((timestamp - whole_seconds) * 1_000_000_000.0).round() as u32;
            if nanos == 1_000_000_000 {
                seconds = seconds.checked_add(1)?;
                nanos = 0;
            }
            DateTime::from_timestamp(seconds, nanos)
        })
}

pub fn is_jwt_expired_or_near(token: &str, threshold: Duration) -> bool {
    parse_jwt_expiration(token)
        .map(|exp| exp <= Utc::now() + threshold)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens with an `aud` claim must parse successfully.
    /// `jsonwebtoken::Validation::default()` enables audience validation which silently rejects these tokens unless `validate_aud = false` is set.
    #[test]
    fn parses_jwt_with_aud_claim() {
        let token = build_test_jwt(r#"{"aud":["some-audience"],"exp":1772575524}"#);
        let exp = parse_jwt_expiration(&token);
        assert_eq!(exp.unwrap().timestamp(), 1772575524);
    }

    #[test]
    fn parses_fractional_numeric_date_expiry() {
        let token = build_test_jwt(r#"{"exp":1772575524.5}"#);
        let expiration = parse_jwt_expiration(&token).expect("parse fractional exp");
        assert_eq!(expiration.timestamp(), 1772575524);
        assert_eq!(expiration.timestamp_subsec_nanos(), 500_000_000);
    }

    fn build_test_jwt(payload_json: &str) -> String {
        use base64::Engine;
        let enc = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = enc.encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let payload = enc.encode(payload_json);
        format!("{header}.{payload}.fake-signature")
    }
}
