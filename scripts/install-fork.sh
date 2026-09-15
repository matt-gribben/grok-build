#!/usr/bin/env bash
# Install a prebuilt grok-cursor binary from matt-gribben/grok-build GitHub Releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/matt-gribben/grok-build/main/scripts/install-fork.sh | bash
#   VERSION=nightly bash scripts/install-fork.sh
#   VERSION=v1.0.24 bash scripts/install-fork.sh
#
# Installs as `grok-cursor` (never `grok`) under ~/.grok-cursor. Official Grok
# (`~/.grok`, `grok` on PATH) and source-build `~/.grok-build` are left alone.

set -euo pipefail

REPO="${GROK_BUILD_REPO:-matt-gribben/grok-build}"
VERSION="${VERSION:-}"
CMD_NAME="grok-cursor"
PREFIX="${GROK_CURSOR_HOME:-${HOME}/.grok-cursor}"
BIN_DIR="${PREFIX}/bin"
LIBEXEC_DIR="${PREFIX}/libexec"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "${uname_s}" in
  Darwin) os="macos" ;;
  Linux) os="linux" ;;
  *)
    echo "Unsupported OS: ${uname_s}" >&2
    exit 1
    ;;
esac

case "${uname_m}" in
  x86_64 | amd64) arch="x86_64" ;;
  arm64 | aarch64) arch="aarch64" ;;
  *)
    echo "Unsupported architecture: ${uname_m}" >&2
    exit 1
    ;;
esac

asset="grok-cursor-${os}-${arch}"
api="https://api.github.com/repos/${REPO}/releases"

if [ -z "${VERSION}" ]; then
  VERSION="$(curl -fsSL "${api}/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
  if [ -z "${VERSION}" ]; then
    VERSION="nightly"
  fi
fi

url="https://github.com/${REPO}/releases/download/${VERSION}/${asset}"
mkdir -p "${BIN_DIR}" "${LIBEXEC_DIR}"
tmp="$(mktemp "${TMPDIR:-/tmp}/grok-cursor-XXXXXX")"
trap 'rm -f "${tmp}"' EXIT

echo "Downloading ${url}"
if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
  echo "Failed to download ${asset} from ${VERSION}." >&2
  echo "Check https://github.com/${REPO}/releases" >&2
  exit 1
fi
chmod +x "${tmp}"

payload="${LIBEXEC_DIR}/${CMD_NAME}"
mv "${tmp}" "${payload}"
trap - EXIT

wrapper="${BIN_DIR}/${CMD_NAME}"
cat >"${wrapper}" <<EOF
#!/usr/bin/env bash
# Keep fork state out of official ~/.grok and source-build ~/.grok-build.
export GROK_HOME="\${GROK_HOME:-${PREFIX}}"
exec "${payload}" "\$@"
EOF
chmod +x "${wrapper}"

# Drop a PATH-friendly copy when ~/.local/bin already exists, without naming it grok.
if [ -d "${HOME}/.local/bin" ]; then
  ln -sfn "${wrapper}" "${HOME}/.local/bin/${CMD_NAME}"
fi

config="${PREFIX}/config.toml"
if [ ! -f "${config}" ]; then
  cat >"${config}" <<'EOF'
[cli]
# Fork builds should not auto-update from the official x.ai channel.
auto_update = false
EOF
elif ! grep -q '^[[:space:]]*auto_update' "${config}"; then
  if grep -q '^\[cli\]' "${config}"; then
    awk '
      BEGIN { done=0 }
      /^\[cli\]$/ { print; print "auto_update = false"; done=1; next }
      { print }
      END { if (!done) print "\n[cli]\nauto_update = false" }
    ' "${config}" >"${config}.tmp" && mv "${config}.tmp" "${config}"
  else
    printf '\n[cli]\nauto_update = false\n' >>"${config}"
  fi
fi

case ":${PATH}:" in
  *":${BIN_DIR}:"* | *":${HOME}/.local/bin:"*) on_path=1 ;;
  *) on_path=0 ;;
esac

echo
echo "Installed ${wrapper}"
GROK_HOME="${PREFIX}" "${payload}" --version || true
echo
if [ "${on_path}" -eq 0 ]; then
  echo "Add this to your shell profile:"
  echo "  export PATH=\"${BIN_DIR}:\$PATH\""
  echo
fi
echo "Run it as: ${CMD_NAME}"
echo "Profile: ${PREFIX} (official grok keeps ~/.grok; source builds keep ~/.grok-build)."
echo "macOS may quarantine unsigned downloads; if launch is blocked:"
echo "  xattr -d com.apple.quarantine \"${payload}\""
