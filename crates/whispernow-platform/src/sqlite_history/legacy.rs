use std::{fs, io::ErrorKind};
use whispernow_domain::AppError;

use rusqlite::{Connection, Transaction, params};
use serde::Deserialize;
use time::{OffsetDateTime, PrimitiveDateTime, format_description::well_known::Rfc3339};
use whispernow_domain::{EntryStatus, ErrorCategory, UtcMillis};

use super::{history_error, status_text};
use crate::AppPaths;

/// One record of the Python release's `history.json`; its other fields are not kept.
#[derive(Deserialize)]
struct LegacyRecord {
    timestamp: String,
    /// `[ERROR]` when recognition failed.
    raw_text: String,
    #[serde(default)]
    enhanced_text: Option<String>,
}

/// Imports `history.json` once: it is renamed to `history.json.migrated` afterwards.
pub(super) fn import_legacy(
    connection: &mut Connection,
    paths: &AppPaths,
) -> Result<Option<AppError>, AppError> {
    let document = match fs::read_to_string(paths.history_file()) {
        Ok(document) => document,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(history_error("history_import_read")),
    };
    let Ok(records) = serde_json::from_str::<Vec<LegacyRecord>>(&document) else {
        quarantine_legacy(paths)?;
        tracing::warn!(
            code = "history_import_invalid",
            "legacy transcription history was quarantined"
        );
        return Ok(Some(AppError::new(
            ErrorCategory::General,
            "history_import_invalid",
            "Legacy transcription history could not be imported",
        )));
    };
    let transaction = connection
        .transaction()
        .map_err(|_| history_error("history_import_begin"))?;
    for record in records {
        import_record(&transaction, record)?;
    }
    transaction
        .commit()
        .map_err(|_| history_error("history_import_commit"))?;
    rename_legacy(paths);
    Ok(None)
}

/// Imports one Python record as an entry. A record whose recognition failed keeps no transcript.
fn import_record(transaction: &Transaction<'_>, record: LegacyRecord) -> Result<(), AppError> {
    let created_at = parse_legacy_timestamp(&record.timestamp).unwrap_or(UtcMillis(0));
    let failed = record.raw_text == "[ERROR]";
    let (status, raw_text, error) = if failed {
        (
            EntryStatus::Failed,
            None,
            Some("Speech recognition failed".to_owned()),
        )
    } else {
        (EntryStatus::Ok, Some(record.raw_text), None)
    };
    transaction
        .execute(
            "INSERT INTO entries (created_at_ms, raw_text, cleaned_text, status, error)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                created_at.get(),
                raw_text,
                record.enhanced_text.filter(|_| !failed),
                status_text(status),
                error,
            ],
        )
        .map_err(|_| history_error("history_import_record"))?;
    Ok(())
}

fn parse_legacy_timestamp(value: &str) -> Option<UtcMillis> {
    if let Ok(numeric) = value.parse::<i64>() {
        return Some(UtcMillis(if numeric.unsigned_abs() < 1_000_000_000_000 {
            numeric.saturating_mul(1_000)
        } else {
            numeric
        }));
    }
    let timestamp = if let Ok(timestamp) = OffsetDateTime::parse(value, &Rfc3339) {
        timestamp
    } else {
        let separator = if value.contains('T') { 'T' } else { ' ' };
        let description = if value.contains('.') {
            format!("[year]-[month]-[day]{separator}[hour]:[minute]:[second].[subsecond]")
        } else {
            format!("[year]-[month]-[day]{separator}[hour]:[minute]:[second]")
        };
        let description = time::format_description::parse(&description).ok()?;
        PrimitiveDateTime::parse(value, &description)
            .ok()?
            .assume_utc()
    };
    i64::try_from(timestamp.unix_timestamp_nanos().div_euclid(1_000_000))
        .ok()
        .map(UtcMillis)
}

fn rename_legacy(paths: &AppPaths) {
    if fs::rename(paths.history_file(), paths.migrated_history_file()).is_err() {
        tracing::warn!(
            code = "history_import_rename",
            "legacy transcription history could not be renamed; it is imported again at the next start"
        );
    }
}

fn quarantine_legacy(paths: &AppPaths) -> Result<(), AppError> {
    let quarantine = paths.history_quarantine_dir();
    fs::create_dir_all(&quarantine).map_err(|_| history_error("history_import_quarantine"))?;
    let initial = quarantine.join("history.json.invalid");
    let destination = if initial.exists() {
        (1_u32..=1_000)
            .map(|index| quarantine.join(format!("history.json.invalid.{index}")))
            .find(|candidate| !candidate.exists())
            .ok_or_else(|| history_error("history_import_quarantine"))?
    } else {
        initial
    };
    fs::rename(paths.history_file(), destination)
        .map_err(|_| history_error("history_import_quarantine"))
}
