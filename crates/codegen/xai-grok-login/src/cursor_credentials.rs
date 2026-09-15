//! Read-only access to the active Cursor Desktop session for Cursor inference.
//!
//! This deliberately reads only `cursorAuth/accessToken` from Cursor's
//! `state.vscdb`. It does not inspect the refresh token, invoke a browser, or
//! make a network request. The private Cursor inference protocol is separate
//! from this credential source.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration as StdDuration;

use chrono::{Duration, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use thiserror::Error;
use zeroize::Zeroizing;

const ACCESS_TOKEN_KEY: &str = "cursorAuth/accessToken";
const EXPIRY_SKEW: Duration = Duration::minutes(5);
const DATABASE_BUSY_TIMEOUT: StdDuration = StdDuration::from_millis(250);

/// Cursor Desktop's current access token.
///
/// The token is intentionally neither `Debug` nor serializable. Callers should
/// borrow it only while constructing an authenticated Cursor request.
pub struct CursorAccessToken(Zeroizing<String>);

impl CursorAccessToken {
    /// Borrow the token for the authenticated request that needs it.
    pub fn expose_secret(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for CursorAccessToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorAccessToken([REDACTED])")
    }
}

/// Sanitized failures from the local Cursor Desktop credential source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CursorCredentialError {
    #[error("Cursor Desktop credentials are not supported on this platform yet.")]
    UnsupportedPlatform,
    #[error("Could not locate the Cursor Desktop session database path for this user.")]
    DesktopPathUnavailable,
    #[error("Could not read the Cursor Desktop session database.")]
    DatabaseUnavailable,
    #[error("No signed-in Cursor Desktop access token was found.")]
    NotSignedIn,
    #[error("The Cursor Desktop access token has expired; restart Cursor Desktop and try again.")]
    Expired,
}

/// Read Cursor Desktop's local access token without opening the database for
/// writing or creating a missing database.
pub fn resolve_cursor_desktop_access_token() -> Result<CursorAccessToken, CursorCredentialError> {
    let path = cursor_state_database_path()?;
    read_access_token_from_database(&path)
}

fn cursor_state_database_path() -> Result<PathBuf, CursorCredentialError> {
    let home = std::env::var_os("HOME");
    let appdata = std::env::var_os("APPDATA");
    cursor_state_database_path_for(std::env::consts::OS, home.as_deref(), appdata.as_deref())
}

fn cursor_state_database_path_for(
    os: &str,
    home: Option<&OsStr>,
    appdata: Option<&OsStr>,
) -> Result<PathBuf, CursorCredentialError> {
    match os {
        "macos" => home
            .filter(|home| is_absolute_profile_root(os, home))
            .map(|home| {
                PathBuf::from(home)
                    .join("Library/Application Support/Cursor/User/globalStorage/state.vscdb")
            })
            .ok_or(CursorCredentialError::DesktopPathUnavailable),
        "linux" => home
            .filter(|home| is_absolute_profile_root(os, home))
            .map(|home| PathBuf::from(home).join(".config/Cursor/User/globalStorage/state.vscdb"))
            .ok_or(CursorCredentialError::DesktopPathUnavailable),
        "windows" => appdata
            .filter(|appdata| is_absolute_profile_root(os, appdata))
            .map(|appdata| PathBuf::from(appdata).join("Cursor/User/globalStorage/state.vscdb"))
            .ok_or(CursorCredentialError::DesktopPathUnavailable),
        _ => Err(CursorCredentialError::UnsupportedPlatform),
    }
}

fn is_absolute_profile_root(os: &str, root: &OsStr) -> bool {
    if root.is_empty() {
        return false;
    }
    if os != "windows" {
        // This helper also validates simulated OS paths in cross-platform tests,
        // so host `Path::is_absolute` would apply the wrong OS's path rules.
        return root.to_string_lossy().starts_with('/');
    }

    let root = root.to_string_lossy();
    let bytes = root.as_bytes();
    root.starts_with("\\\\")
        || root.starts_with("//")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\'))
}

fn read_access_token_from_database(
    path: &Path,
) -> Result<CursorAccessToken, CursorCredentialError> {
    // READ_ONLY prevents writes to the database and WAL. SQLite may create or
    // update `-shm` for WAL locking/index coordination; this is SQLite-managed
    // shared state, not a credential write. Do not alter journal mode or run a
    // checkpoint, and never fall back to writable flags.
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|_| CursorCredentialError::DatabaseUnavailable)?;
    connection
        .busy_timeout(DATABASE_BUSY_TIMEOUT)
        .map_err(|_| CursorCredentialError::DatabaseUnavailable)?;
    let stored: Option<String> = connection
        .query_row(
            "SELECT value FROM ItemTable WHERE key = ?1",
            [ACCESS_TOKEN_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|_| CursorCredentialError::DatabaseUnavailable)?;
    drop(connection);

    let mut token = Zeroizing::new(stored.ok_or(CursorCredentialError::NotSignedIn)?);
    let leading_whitespace = token.len() - token.trim_start().len();
    token.replace_range(..leading_whitespace, "");
    let trimmed_length = token.trim_end().len();
    token.truncate(trimmed_length);
    if token.is_empty() {
        return Err(CursorCredentialError::NotSignedIn);
    }

    // Cursor Desktop has used both JWT-shaped and opaque session tokens. When
    // an expiry claim is present, match Pi's five-minute early-expiry margin;
    // opaque or unparseable tokens follow Pi's behavior and are left to Cursor
    // to validate. This provider does not cache the token beyond the request.
    if crate::parse_jwt_expiration(&token).is_some_and(|expiry| expiry <= Utc::now() + EXPIRY_SKEW)
    {
        return Err(CursorCredentialError::Expired);
    }

    Ok(CursorAccessToken(token))
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut path = path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use base64::Engine;
    use rusqlite::Connection;
    use tempfile::TempDir;

    use super::*;

    fn create_database(path: &Path) -> Connection {
        let connection = Connection::open(path).expect("create fixture database");
        connection
            .execute(
                "CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)",
                [],
            )
            .expect("create ItemTable");
        connection
    }

    fn put_value(connection: &Connection, key: &str, value: &str) {
        connection
            .execute(
                "INSERT INTO ItemTable (key, value) VALUES (?1, ?2)",
                (key, value),
            )
            .expect("insert fixture value");
    }

    fn token_with_expiration(exp: i64) -> String {
        let encoder = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = encoder.encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let payload = encoder.encode(format!(r#"{{"exp":{exp}}}"#));
        format!("{header}.{payload}.fixture-signature")
    }

    fn token_without_expiration() -> String {
        let encoder = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = encoder.encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let payload = encoder.encode(r#"{"sub":"fixture"}"#);
        format!("{header}.{payload}.fixture-signature")
    }

    fn database_with_token(temp: &TempDir, name: &str, token: &str) -> PathBuf {
        let path = temp.path().join(name).join("state.vscdb");
        fs::create_dir_all(path.parent().expect("database parent")).expect("database directory");
        let connection = create_database(&path);
        put_value(&connection, ACCESS_TOKEN_KEY, token);
        drop(connection);
        path
    }

    #[test]
    fn resolves_supported_platform_paths_and_reports_missing_environment() {
        let home = OsStr::new("/home/example");
        let appdata = OsStr::new("C:/Users/example/AppData/Roaming");
        assert_eq!(
            cursor_state_database_path_for("macos", Some(home), None),
            Ok(PathBuf::from(
                "/home/example/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
            ))
        );
        assert_eq!(
            cursor_state_database_path_for("linux", Some(home), None),
            Ok(PathBuf::from(
                "/home/example/.config/Cursor/User/globalStorage/state.vscdb"
            ))
        );
        assert_eq!(
            cursor_state_database_path_for("windows", None, Some(appdata)),
            Ok(PathBuf::from(
                "C:/Users/example/AppData/Roaming/Cursor/User/globalStorage/state.vscdb"
            ))
        );
        assert_eq!(
            cursor_state_database_path_for("macos", None, None),
            Err(CursorCredentialError::DesktopPathUnavailable)
        );
        assert_eq!(
            cursor_state_database_path_for("linux", Some(OsStr::new("")), None),
            Err(CursorCredentialError::DesktopPathUnavailable)
        );
        assert_eq!(
            cursor_state_database_path_for("linux", Some(OsStr::new("relative/home")), None),
            Err(CursorCredentialError::DesktopPathUnavailable)
        );
        assert_eq!(
            cursor_state_database_path_for("windows", None, Some(OsStr::new("relative/appdata"))),
            Err(CursorCredentialError::DesktopPathUnavailable)
        );
        assert_eq!(
            cursor_state_database_path_for("plan9", None, None),
            Err(CursorCredentialError::UnsupportedPlatform)
        );
    }

    #[test]
    fn reads_only_access_token_and_redacts_debug() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let connection = create_database(&path);
        put_value(&connection, ACCESS_TOKEN_KEY, "opaque-cursor-access-token");
        put_value(&connection, "cursorAuth/refreshToken", "must-not-be-used");
        drop(connection);

        let token = read_access_token_from_database(&path).expect("read access token");
        assert_eq!(token.expose_secret(), "opaque-cursor-access-token");
        assert_eq!(format!("{token:?}"), "CursorAccessToken([REDACTED])");
    }

    #[test]
    fn rejects_expired_access_token() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let connection = create_database(&path);
        put_value(&connection, ACCESS_TOKEN_KEY, &token_with_expiration(1));
        drop(connection);

        assert_eq!(
            read_access_token_from_database(&path).unwrap_err(),
            CursorCredentialError::Expired
        );
    }

    #[test]
    fn rejects_token_inside_pi_expiry_skew_and_accepts_future_expiry() {
        let temp = TempDir::new().expect("temp dir");
        let now = Utc::now().timestamp();
        let near_path = database_with_token(&temp, "near", &token_with_expiration(now + 120));
        let future_path = database_with_token(&temp, "future", &token_with_expiration(now + 600));

        assert_eq!(
            read_access_token_from_database(&near_path).unwrap_err(),
            CursorCredentialError::Expired
        );
        assert!(read_access_token_from_database(&future_path).is_ok());
    }

    #[test]
    fn opaque_malformed_and_missing_exp_tokens_follow_pi_policy() {
        let temp = TempDir::new().expect("temp dir");
        let rows = vec![
            ("opaque", "opaque-cursor-token".to_string()),
            ("malformed", "not.a.valid.jwt".to_string()),
            ("no-exp", token_without_expiration()),
        ];

        for (name, token) in rows {
            let path = database_with_token(&temp, name, &token);
            assert!(
                read_access_token_from_database(&path).is_ok(),
                "Pi treats tokens without a readable exp claim as opaque"
            );
        }
    }

    #[test]
    fn missing_database_is_not_created() {
        let temp = TempDir::new().expect("temp dir");
        let parent = temp.path().join("existing-parent");
        fs::create_dir_all(&parent).expect("existing parent directory");
        let path = parent.join("state.vscdb");

        assert_eq!(
            read_access_token_from_database(&path).unwrap_err(),
            CursorCredentialError::DatabaseUnavailable
        );
        assert!(!path.exists());
        assert!(!sidecar_path(&path, "-wal").exists());
        assert!(!sidecar_path(&path, "-shm").exists());
    }

    #[test]
    fn corrupt_database_returns_sanitized_error() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        fs::write(&path, b"not a sqlite database").expect("write corrupt fixture");

        assert_eq!(
            read_access_token_from_database(&path).unwrap_err(),
            CursorCredentialError::DatabaseUnavailable
        );
    }

    #[test]
    fn locked_database_returns_sanitized_error_without_waiting_indefinitely() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let writer = create_database(&path);
        put_value(&writer, ACCESS_TOKEN_KEY, "opaque-cursor-access-token");
        writer
            .execute_batch("BEGIN EXCLUSIVE")
            .expect("lock fixture exclusively");

        let started = std::time::Instant::now();
        let result = read_access_token_from_database(&path);
        assert_eq!(
            result.unwrap_err(),
            CursorCredentialError::DatabaseUnavailable
        );
        assert!(started.elapsed() < StdDuration::from_secs(2));
        writer
            .execute_batch("ROLLBACK")
            .expect("release fixture lock");
    }

    #[test]
    fn refresh_token_alone_is_not_a_credential_fallback() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let connection = create_database(&path);
        put_value(&connection, "cursorAuth/refreshToken", "must-not-be-used");
        drop(connection);

        assert_eq!(
            read_access_token_from_database(&path).unwrap_err(),
            CursorCredentialError::NotSignedIn
        );
    }

    #[test]
    fn read_only_wal_lookup_preserves_database_and_wal() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let writer = Connection::open(&path).expect("create WAL fixture");
        let journal_mode: String = writer
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .expect("enable WAL");
        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
        writer
            .execute(
                "CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)",
                [],
            )
            .expect("create ItemTable");
        put_value(&writer, ACCESS_TOKEN_KEY, "opaque-wal-session-token");
        let wal_path = sidecar_path(&path, "-wal");
        let shm_path = sidecar_path(&path, "-shm");
        assert!(wal_path.exists());
        assert!(shm_path.exists());
        let database_before = fs::read(&path).expect("snapshot database");
        let wal_before = fs::read(&wal_path).expect("snapshot WAL");
        let shm_size_before = fs::metadata(&shm_path).expect("stat SHM").len();

        let token = read_access_token_from_database(&path).expect("read WAL database");
        assert_eq!(token.expose_secret(), "opaque-wal-session-token");

        assert_eq!(
            fs::read(&path).expect("read database after lookup"),
            database_before,
            "main database must remain unchanged"
        );
        assert_eq!(
            fs::read(&wal_path).expect("read WAL after lookup"),
            wal_before,
            "WAL must remain unchanged"
        );
        assert!(shm_path.exists(), "existing SHM must not be removed");
        assert_eq!(
            fs::metadata(&shm_path).expect("stat SHM after read").len(),
            shm_size_before,
            "existing SHM size must not change"
        );

        drop(writer);
    }

    #[test]
    fn read_only_wal_lookup_reads_a_wal_without_an_existing_shared_memory_file() {
        let temp = TempDir::new().expect("temp dir");
        let path = temp.path().join("state.vscdb");
        let writer = Connection::open(&path).expect("create WAL fixture");
        let _: String = writer
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .expect("enable WAL");
        writer
            .execute(
                "CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)",
                [],
            )
            .expect("create ItemTable");
        put_value(&writer, ACCESS_TOKEN_KEY, "opaque-wal-session-token");
        let clone_path = temp.path().join("clone/state.vscdb");
        fs::create_dir_all(clone_path.parent().expect("clone parent")).expect("clone directory");
        fs::copy(&path, &clone_path).expect("copy WAL database");
        let clone_wal_path = sidecar_path(&clone_path, "-wal");
        fs::copy(sidecar_path(&path, "-wal"), &clone_wal_path).expect("copy WAL file");
        let clone_shm_path = sidecar_path(&clone_path, "-shm");
        assert!(!clone_shm_path.exists());
        let clone_database_before = fs::read(&clone_path).expect("snapshot cloned database");
        let clone_wal_before = fs::read(&clone_wal_path).expect("snapshot cloned WAL");
        drop(writer);

        let token = read_access_token_from_database(&clone_path).expect("read WAL without SHM");
        assert_eq!(token.expose_secret(), "opaque-wal-session-token");
        assert_eq!(
            fs::read(&clone_path).expect("read cloned database after lookup"),
            clone_database_before,
            "main database must remain unchanged"
        );
        assert_eq!(
            fs::read(&clone_wal_path).expect("read cloned WAL after lookup"),
            clone_wal_before,
            "WAL must remain unchanged"
        );
        assert!(
            clone_shm_path.exists(),
            "the bundled SQLite reader should create its WAL coordination SHM file"
        );
    }
}
