use std::{env, fs, path::PathBuf, process::ExitCode};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use minisign_verify::{PublicKey, Signature};

struct Arguments {
    payload: PathBuf,
    signature: PathBuf,
    public_key: String,
}

fn main() -> ExitCode {
    let result = parse_arguments().and_then(|arguments| verify(&arguments));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("update signature verification failed: {message}");
            ExitCode::FAILURE
        }
    }
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut payload = None;
    let mut signature = None;
    let mut public_key = None;
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--payload" => payload = Some(PathBuf::from(value)),
            "--signature" => signature = Some(PathBuf::from(value)),
            "--public-key" => public_key = Some(value),
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    Ok(Arguments {
        payload: payload.ok_or_else(|| "--payload is required".to_owned())?,
        signature: signature.ok_or_else(|| "--signature is required".to_owned())?,
        public_key: public_key.ok_or_else(|| "--public-key is required".to_owned())?,
    })
}

fn decode_envelope(value: &str, kind: &str) -> Result<String, String> {
    let decoded = STANDARD
        .decode(value.trim())
        .map_err(|_| format!("{kind} is not a valid Tauri base64 envelope"))?;
    String::from_utf8(decoded).map_err(|_| format!("{kind} envelope is not UTF-8"))
}

fn verify(arguments: &Arguments) -> Result<(), String> {
    let payload =
        fs::read(&arguments.payload).map_err(|error| format!("could not read payload: {error}"))?;
    let signature_envelope = fs::read_to_string(&arguments.signature)
        .map_err(|error| format!("could not read signature: {error}"))?;
    verify_bytes(&payload, &signature_envelope, &arguments.public_key)
}

fn verify_bytes(payload: &[u8], signature_envelope: &str, public_key: &str) -> Result<(), String> {
    let public_key = PublicKey::decode(&decode_envelope(public_key, "public key")?)
        .map_err(|_| "public key envelope is malformed".to_owned())?;
    let signature = Signature::decode(&decode_envelope(signature_envelope, "signature")?)
        .map_err(|_| "signature envelope is malformed".to_owned())?;

    public_key
        .verify(payload, &signature, true)
        .map_err(|_| "signature does not match the payload and public key".to_owned())
}

#[cfg(test)]
mod tests {
    use super::verify_bytes;

    const PUBLIC_KEY: &str = include_str!("../../../release/fixtures/disposable-updater.key.pub");
    const ALPHA: &[u8] = include_bytes!("../../../release/fixtures/signatures/alpha.bin");
    const ALPHA_SIGNATURE: &str =
        include_str!("../../../release/fixtures/signatures/alpha.bin.sig");
    const BETA_SIGNATURE: &str = include_str!("../../../release/fixtures/signatures/beta.bin.sig");

    #[test]
    fn verifies_the_exact_fixture_bytes() {
        assert!(verify_bytes(ALPHA, ALPHA_SIGNATURE, PUBLIC_KEY).is_ok());
    }

    #[test]
    fn rejects_changed_bytes_and_a_swapped_valid_signature() {
        assert!(verify_bytes(b"changed", ALPHA_SIGNATURE, PUBLIC_KEY).is_err());
        assert!(verify_bytes(ALPHA, BETA_SIGNATURE, PUBLIC_KEY).is_err());
    }
}
