//! `SQLite` transcription history: one row per dictation, and its audio as `<id>.wav`.

use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
};
use whispernow_domain::AppError;

use hound::{SampleFormat, WavSpec, WavWriter};
use rusqlite::{
    Connection, OptionalExtension, Row, params, params_from_iter,
    types::{Type, Value},
};
use whispernow_core::audio::TARGET_SAMPLE_RATE;
use whispernow_domain::{
    EntryId, EntryStatus, ErrorCategory, HistoryEntry, HistoryLimit, HistoryPage,
    HistoryPageRequest, ModelId, NewHistoryEntry, UtcMillis,
};

use crate::AppPaths;

mod legacy;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_ms INTEGER NOT NULL,
    duration_ms INTEGER,
    raw_text TEXT,
    cleaned_text TEXT,
    status TEXT NOT NULL,
    error TEXT,
    asr_model_id TEXT
);
CREATE INDEX IF NOT EXISTS entries_newest_first ON entries (created_at_ms DESC, id DESC);
";

const COLUMNS: &str = "id, created_at_ms, duration_ms, raw_text, cleaned_text, status, error";

/// Callers serialize its use and keep it away from latency-sensitive runtime and UI threads.
pub struct SqliteHistoryRepository {
    connection: Connection,
    audio_dir: PathBuf,
}

impl SqliteHistoryRepository {
    /// Opens the history database, imports the Python release's `history.json` when present,
    /// and removes unowned numeric WAV files and unfinished audio writes. Returns any non-fatal diagnostic, such
    /// as a malformed `history.json` that was moved aside rather than imported.
    pub fn open(paths: &AppPaths) -> Result<(Self, Option<AppError>), AppError> {
        let open_error = |code: &str| {
            tracing::warn!(code, "transcription history did not open");
            AppError::new(
                ErrorCategory::HistoryUnavailable,
                "history_unavailable",
                "Transcription history is unavailable",
            )
        };
        let database = paths.history_database_file();
        if let Some(parent) = database.parent() {
            fs::create_dir_all(parent).map_err(|_| open_error("history_open"))?;
        }
        let mut connection = Connection::open(&database).map_err(|_| open_error("history_open"))?;
        connection
            .execute_batch(SCHEMA)
            .map_err(|_| open_error("history_schema"))?;
        let diagnostic = legacy::import_legacy(&mut connection, paths)
            .map_err(|error| open_error(&error.code))?;
        let audio_dir = paths.recordings_dir();
        remove_orphan_audio(&connection, &audio_dir);
        Ok((
            Self {
                connection,
                audio_dir,
            },
            diagnostic,
        ))
    }

    /// Stores one dictation. Its audio is kept when `keep_audio` is on or the dictation failed
    /// or was cancelled; entries beyond `limit` are then deleted with their audio. A failure to
    /// write the audio still stores the entry, without audio.
    pub fn insert(
        &mut self,
        entry: NewHistoryEntry,
        audio: &[f32],
        keep_audio: bool,
        limit: HistoryLimit,
    ) -> Result<EntryId, AppError> {
        let NewHistoryEntry {
            created_at,
            duration_millis,
            raw_text,
            cleaned_text,
            status,
            error,
            asr_model_id,
        } = entry;
        let transaction = self
            .connection
            .transaction()
            .map_err(|_| history_error("history_insert"))?;
        transaction
            .execute(
                "INSERT INTO entries (created_at_ms, duration_ms, raw_text, cleaned_text, status,
                     error, asr_model_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    created_at.get(),
                    duration_millis.map(sql_u64),
                    raw_text,
                    cleaned_text,
                    status_text(status),
                    error,
                    asr_model_id.as_ref().map(ModelId::as_str),
                ],
            )
            .map_err(|_| history_error("history_insert"))?;
        let id = EntryId(transaction.last_insert_rowid());
        // Cleanup of a prior rolled-back insert may have failed for this reused id.
        let pending_audio = audio_path(&self.audio_dir, id);
        match fs::symlink_metadata(&pending_audio) {
            Ok(_) => fs::remove_file(&pending_audio)
                .map_err(|_| history_error("history_audio_delete"))?,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                ) => {}
            Err(_) => return Err(history_error("history_audio_delete")),
        }
        let wrote_audio = if (keep_audio || status.always_keeps_audio()) && !audio.is_empty() {
            if write_wav(&self.audio_dir, id, audio).is_ok() {
                true
            } else {
                tracing::warn!(
                    code = "history_audio_write",
                    "a dictation's audio could not be kept"
                );
                false
            }
        } else {
            false
        };
        let result = prune(&transaction, limit).and_then(|pruned| {
            transaction
                .commit()
                .map_err(|_| history_error("history_insert"))?;
            Ok(pruned)
        });
        let pruned = match result {
            Ok(pruned) => pruned,
            Err(error) => {
                // A rolled-back id can be reused by the next insert.
                if wrote_audio {
                    remove_audio(&self.audio_dir, id);
                }
                return Err(error);
            }
        };
        for id in pruned {
            remove_audio(&self.audio_dir, id);
        }
        Ok(id)
    }

    pub fn mark_not_delivered(&self, id: EntryId, error: &str) -> Result<(), AppError> {
        self.connection
            .execute(
                "UPDATE entries SET status = ?1, error = ?2 WHERE id = ?3",
                params![status_text(EntryStatus::NotDelivered), error, id.0],
            )
            .map(drop)
            .map_err(|_| history_error("history_update"))
    }

    /// Lists one newest-first page.
    pub fn list(&self, request: &HistoryPageRequest) -> Result<HistoryPage, AppError> {
        let limit = request.limit.max(1);
        let mut clauses = Vec::new();
        let mut values: Vec<Value> = Vec::new();
        if let Some(text) = request.filter.search_text() {
            clauses.push("(raw_text LIKE ? ESCAPE '\\' OR cleaned_text LIKE ? ESCAPE '\\')");
            let pattern = format!("%{}%", escape_like(text));
            values.push(Value::Text(pattern.clone()));
            values.push(Value::Text(pattern));
        }
        if request.filter.failed_only {
            clauses.push("status = 'failed'");
        }
        let filter = if clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", clauses.join(" AND "))
        };
        // One extra row tells whether another page follows.
        values.push(Value::Integer(i64::from(limit) + 1));
        let mut statement = self
            .connection
            .prepare(&format!(
                "SELECT {COLUMNS} FROM entries {filter}
                 ORDER BY created_at_ms DESC, id DESC LIMIT ?"
            ))
            .map_err(|_| history_error("history_list"))?;
        let mut entries = statement
            .query_map(params_from_iter(values), |row| {
                entry_from_row(row, &self.audio_dir)
            })
            .and_then(Iterator::collect::<Result<Vec<_>, _>>)
            .map_err(|_| history_error("history_list"))?;
        let has_more = entries.len() > limit as usize;
        entries.truncate(limit as usize);
        Ok(HistoryPage { entries, has_more })
    }

    pub fn get(&self, id: EntryId) -> Result<Option<HistoryEntry>, AppError> {
        self.connection
            .query_row(
                &format!("SELECT {COLUMNS} FROM entries WHERE id = ?1"),
                [id.0],
                |row| entry_from_row(row, &self.audio_dir),
            )
            .optional()
            .map_err(|_| history_error("history_read"))
    }

    /// Reads an entry's audio as a playable WAV file, or `None` when it has none.
    pub fn audio(&self, id: EntryId) -> Result<Option<Vec<u8>>, AppError> {
        match fs::read(audio_path(&self.audio_dir, id)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(history_error("history_audio_read")),
        }
    }

    /// Deletes one entry and its audio; returns whether it existed.
    pub fn delete(&self, id: EntryId) -> Result<bool, AppError> {
        let deleted = self
            .connection
            .execute("DELETE FROM entries WHERE id = ?1", [id.0])
            .map_err(|_| history_error("history_delete"))?;
        remove_audio(&self.audio_dir, id);
        Ok(deleted > 0)
    }

    /// Removes the recordings before deleting rows, preserving their owners if removal fails.
    pub fn clear(&self) -> Result<bool, AppError> {
        let recordings_removed = match crate::clear_data::remove_path(&self.audio_dir) {
            Ok(()) => true,
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(_) => return Err(history_error("history_clear_audio")),
        };
        let deleted = self
            .connection
            .execute("DELETE FROM entries", [])
            .map_err(|_| history_error("history_clear"))?;
        self.connection
            .execute_batch("VACUUM")
            .map_err(|_| history_error("history_clear"))?;
        Ok(deleted > 0 || recordings_removed)
    }

    /// Deletes the entries beyond `limit`, with their audio; returns how many were deleted.
    pub fn prune(&self, limit: HistoryLimit) -> Result<usize, AppError> {
        let pruned = prune(&self.connection, limit)?;
        for id in &pruned {
            remove_audio(&self.audio_dir, *id);
        }
        Ok(pruned.len())
    }
}

fn prune(connection: &Connection, limit: HistoryLimit) -> Result<Vec<EntryId>, AppError> {
    let Some(kept) = limit.max_entries() else {
        return Ok(Vec::new());
    };
    connection
        .prepare(
            "DELETE FROM entries WHERE id IN (
            SELECT id FROM entries ORDER BY created_at_ms DESC, id DESC LIMIT -1 OFFSET ?1
        ) RETURNING id",
        )
        .and_then(|mut statement| {
            statement
                .query_map([i64::from(kept)], |row| row.get(0).map(EntryId))?
                .collect()
        })
        .map_err(|_| history_error("history_prune"))
}

/// Removes unowned numeric WAVs and unfinished writes before the first insert.
fn remove_orphan_audio(connection: &Connection, audio_dir: &Path) -> usize {
    let owned: Result<HashSet<i64>, _> = connection
        .prepare("SELECT id FROM entries")
        .and_then(|mut statement| statement.query_map([], |row| row.get(0))?.collect());
    let (Ok(owned), Ok(files)) = (owned, fs::read_dir(audio_dir)) else {
        return 0;
    };
    let mut removed = 0;
    for file in files.flatten() {
        let path = file.path();
        let wav_id = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".wav"))
            .filter(|stem| !stem.is_empty() && stem.bytes().all(|byte| byte.is_ascii_digit()));
        let orphan_wav = wav_id.is_some_and(|stem| {
            stem.parse::<i64>()
                .ok()
                .is_none_or(|id| !owned.contains(&id))
        });
        let temporary = path.extension().is_some_and(|extension| extension == "tmp")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".history-audio-"));
        if (orphan_wav || temporary)
            && file.file_type().is_ok_and(|kind| kind.is_file())
            && fs::remove_file(&path).is_ok()
        {
            removed += 1;
        }
    }
    removed
}

fn entry_from_row(row: &Row<'_>, audio_dir: &Path) -> rusqlite::Result<HistoryEntry> {
    let status: String = row.get(5)?;
    Ok(HistoryEntry {
        id: EntryId(row.get(0)?),
        created_at: UtcMillis(row.get(1)?),
        duration_millis: row.get::<_, Option<i64>>(2)?.map(i64::unsigned_abs),
        raw_text: row.get(3)?,
        cleaned_text: row.get(4)?,
        status: parse_status(&status).ok_or_else(|| {
            rusqlite::Error::FromSqlConversionFailure(5, Type::Text, "unknown entry status".into())
        })?,
        error: row.get(6)?,
        audio_kept: audio_path(audio_dir, EntryId(row.get(0)?)).is_file(),
    })
}

const fn status_text(status: EntryStatus) -> &'static str {
    match status {
        EntryStatus::Ok => "ok",
        EntryStatus::Failed => "failed",
        EntryStatus::Cancelled => "cancelled",
        EntryStatus::NotDelivered => "not_delivered",
    }
}

fn parse_status(text: &str) -> Option<EntryStatus> {
    match text {
        "ok" => Some(EntryStatus::Ok),
        "failed" => Some(EntryStatus::Failed),
        "cancelled" => Some(EntryStatus::Cancelled),
        "not_delivered" => Some(EntryStatus::NotDelivered),
        _ => None,
    }
}

/// Writes normalized mono 16-kHz samples as a 16-bit WAV, through a temporary file renamed into
/// place so a crash never leaves a partial file under an entry's name.
fn write_wav(audio_dir: &Path, id: EntryId, samples: &[f32]) -> io::Result<()> {
    fs::create_dir_all(audio_dir)?;
    let temporary = tempfile::Builder::new()
        .prefix(".history-audio-")
        .suffix(".tmp")
        .tempfile_in(audio_dir)?;
    {
        let mut writer = WavWriter::new(
            io::BufWriter::new(temporary.as_file()),
            WavSpec {
                channels: 1,
                sample_rate: TARGET_SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )
        .map_err(io::Error::other)?;
        for sample in samples {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the clamped product is within the i16 range"
            )]
            let quantized = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            writer.write_sample(quantized).map_err(io::Error::other)?;
        }
        writer.finalize().map_err(io::Error::other)?;
    }
    temporary.as_file().sync_all()?;
    temporary
        .persist(audio_path(audio_dir, id))
        .map(drop)
        .map_err(|error| error.error)
}

fn remove_audio(audio_dir: &Path, id: EntryId) {
    match fs::remove_file(audio_path(audio_dir, id)) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(_) => tracing::warn!(
            code = "history_audio_delete",
            "a deleted entry's audio could not be removed; it is removed at the next start"
        ),
    }
}

fn audio_path(audio_dir: &Path, id: EntryId) -> PathBuf {
    audio_dir.join(format!("{}.wav", id.0))
}

fn escape_like(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(character, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn sql_u64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn history_error(code: &'static str) -> AppError {
    AppError::new(
        ErrorCategory::General,
        code,
        "Transcription history is unavailable",
    )
}

#[cfg(test)]
mod tests;
