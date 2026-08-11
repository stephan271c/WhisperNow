#!/usr/bin/env bash
# Make WhisperNow's self-signed macOS code-signing certificate.
#
# Without a paid Apple Developer ID, release builds are signed with this certificate
# instead. macOS does not trust it, so users approve the first launch once (System
# Settings > Privacy & Security > Open Anyway), but every release carries the same
# designated requirement (the certificate's hash), so Microphone and Accessibility
# grants survive updates. Replacing the certificate resets those
# grants for every user: make it once and keep the private key.
#
# Usage: scripts/make_macos_signing_certificate.sh <private-output-directory>
#
# Writes the public certificate to release/macos-signing-certificate.pem (commit it;
# CI checks every macOS build against it), and to the output directory, which must be
# outside the repository: signing.p12, signing.p12.base64 and password.txt. Store the
# base64 file and the password as the release environment's MACOS_CERTIFICATE and
# MACOS_CERTIFICATE_PASSWORD secrets, and keep a backup of both.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <private-output-directory>" >&2
  exit 2
fi
here="$(cd "$(dirname "$0")/.." && pwd)"
pinned="$here/release/macos-signing-certificate.pem"
if [ -e "$pinned" ]; then
  echo "$pinned exists; a new certificate resets every user's macOS permissions" >&2
  exit 1
fi
mkdir -p "$1"
output="$(cd "$1" && pwd)"
if git -C "$output" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "$output is inside a git work tree; keep the private key out of repositories" >&2
  rmdir "$output" 2>/dev/null || true
  exit 1
fi
for name in signing.p12 signing.p12.base64 password.txt; do
  if [ -e "$output/$name" ]; then
    echo "$output/$name exists" >&2
    exit 1
  fi
done

umask 077
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
openssl req -x509 -newkey rsa:3072 -nodes -days 7300 -sha256 \
  -subj "/CN=WhisperNow self-signed code signing" \
  -addext "basicConstraints=critical,CA:false" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=critical,codeSigning" \
  -keyout "$work/key.pem" -out "$work/certificate.pem" 2>/dev/null
# No trailing newline: the file is also the GitHub secret's value.
openssl rand -hex 24 | tr -d '\n' > "$output/password.txt"
# macOS's `security import` reads PKCS#12 files protected with 3DES and SHA-1 MACs;
# OpenSSL 3's default (AES-256, PBKDF2) is not readable on every macOS release.
openssl pkcs12 -export -name "WhisperNow self-signed code signing" \
  -inkey "$work/key.pem" -in "$work/certificate.pem" \
  -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -macalg sha1 \
  -passout "file:$output/password.txt" -out "$output/signing.p12"
openssl base64 -A -in "$output/signing.p12" -out "$output/signing.p12.base64"
cp "$work/certificate.pem" "$pinned"
chmod 644 "$pinned"

echo "public certificate: $pinned (commit it)"
openssl x509 -in "$pinned" -noout -fingerprint -sha1
echo "private files: $output (back them up; do not commit them)"
