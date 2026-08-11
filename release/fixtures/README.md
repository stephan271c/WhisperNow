# Release test fixtures

`disposable-updater.key` is an intentionally public test key, protected with the password
`disposable`. It exists only to make local signature tests deterministic. Production tag jobs use
the protected `TAURI_SIGNING_PRIVATE_KEY` secret and must never use this key.

The `signatures/` payloads exercise exact-byte and swapped-signature rejection in the Rust
verification boundary (`tools/update-signature`).
