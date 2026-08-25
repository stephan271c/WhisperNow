//! `SQLite` transcription history: one row per dictation, and its audio as `<id>.wav`.

use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
};

use hound::{SampleFormat, WavSpec, WavWriter};
use rusqlite::{
    Connection, OptionalExtension, Row, params, params_from_iter,
    types::{Type, Value},
};
use whispernow_domain::{
    AppError, EntryId, EntryStatus, HistoryCursor, HistoryEntry, HistoryLimit, HistoryPage,
    HistoryPageRequest, ModelId, NewHistoryEntry, UtcMillis,
};

use crate::AppPaths;

mod legacy;

const SAMPLE_RATE_HZ: u32 = 16_000;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_ms INTEGER NOT NULL,
    duration_ms INTEGER,
    raw_text TEXT,
    cleaned_text TEXT,
    status TEXT NOT NULL,
    error TEXT,
    has_audio INTEGER NOT NULL DEFAULT 0,
    asr_model_id TEXT,
    llm_model TEXT
);
CREATE INDEX IF NOT EXISTS entries_newest_first ON entries (created_at_ms DESC, id DESC);
";

const COLUMNS: &str = "id, created_at_ms, duration_ms, raw_text, cleaned_text, status, error, \
                       has_audio, asr_model_id, llm_model";

/// `SQLite` transcription-history store. Callers serialize its use and keep it away from
/// latency-sensitive runtime and UI threads.
pub struct SqliteHistoryRepository {
    connection: Connection,
    audio_dir: PathBuf,
}

impl SqliteHistoryRepository {
    /// Opens the history database, imports the Python release's `history.json` when present,
    /// and deletes audio files that no entry owns. Also returns any non-fatal diagnostic, such
    /// as a malformed `history.json` that was quarantined rather than imported.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the database cannot be opened.
    pub fn open(paths: &AppPaths) -> Result<(Self, Option<AppError>), AppError> {
        let database = paths.history_database_file();
        if let Some(parent) = database.parent() {
            fs::create_dir_all(parent).map_err(|_| history_error("history_open"))?;
        }
        let mut connection =
            Connection::open(&database).map_err(|_| history_error("history_open"))?;
        connection
            .execute_batch(SCHEMA)
            .map_err(|_| history_error("history_schema"))?;
        let diagnostic = legacy::import_legacy(&mut connection, &paths.history_file())?;
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
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the entry cannot be stored.
    pub fn insert(
        &self,
        entry: NewHistoryEntry,
        audio: &[f32],
        keep_audio: bool,
        limit: HistoryLimit,
    ) -> Result<HistoryEntry, AppError> {
        let NewHistoryEntry {
            created_at,
            duration_millis,
            raw_text,
            cleaned_text,
            status,
            error,
            asr_model_id,
            llm_model,
        } = entry;
        let connection = &self.connection;
        connection
            .execute(
                "INSERT INTO entries (created_at_ms, duration_ms, raw_text, cleaned_text, status,
                     error, asr_model_id, llm_model)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    created_at.get(),
                    duration_millis.map(sql_u64),
                    raw_text,
                    cleaned_text,
                    status_text(status),
                    error,
                    asr_model_id.as_ref().map(ModelId::as_str),
                    llm_model,
                ],
            )
            .map_err(|_| history_error("history_insert"))?;
        let id = entry_id(connection.last_insert_rowid())?;
        if (keep_audio || status.always_keeps_audio()) && !audio.is_empty() {
            if write_wav(&self.audio_dir, id, audio).is_ok() {
                connection
                    .execute(
                        "UPDATE entries SET has_audio = 1 WHERE id = ?1",
                        [sql_id(id)?],
                    )
                    .map_err(|_| history_error("history_insert"))?;
            } else {
                tracing::warn!(
                    code = "history_audio_write",
                    "a dictation's audio could not be kept"
                );
            }
        }
        prune(connection, &self.audio_dir, limit)?;
        get(connection, id)?.ok_or_else(|| history_error("history_insert"))
    }

    /// Marks a stored transcript that never reached the focused application.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the entry cannot be updated.
    pub fn mark_not_delivered(&self, id: EntryId, error: &str) -> Result<(), AppError> {
        self.connection
            .execute(
                "UPDATE entries SET status = ?1, error = ?2 WHERE id = ?3",
                params![status_text(EntryStatus::NotDelivered), error, sql_id(id)?],
            )
            .map(drop)
            .map_err(|_| history_error("history_update"))
    }

    /// Lists one newest-first page.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the page cannot be read.
    pub fn list(&self, request: &HistoryPageRequest) -> Result<HistoryPage, AppError> {
        let limit = request.limit.max(1);
        let mut clauses = Vec::new();
        let mut values: Vec<Value> = Vec::new();
        if let Some(cursor) = request.before {
            clauses.push("(created_at_ms < ? OR (created_at_ms = ? AND id < ?))");
            values.push(Value::Integer(cursor.created_at.get()));
            values.push(Value::Integer(cursor.created_at.get()));
            values.push(Value::Integer(sql_id(cursor.id)?));
        }
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
            .query_map(params_from_iter(values), entry_from_row)
            .and_then(Iterator::collect::<Result<Vec<_>, _>>)
            .map_err(|_| history_error("history_list"))?;
        let next_cursor = if entries.len() > limit as usize {
            entries.truncate(limit as usize);
            entries.last().map(|last| HistoryCursor {
                created_at: last.created_at,
                id: last.id,
            })
        } else {
            None
        };
        Ok(HistoryPage {
            entries,
            next_cursor,
        })
    }

    /// Loads one entry.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the lookup fails.
    pub fn get(&self, id: EntryId) -> Result<Option<HistoryEntry>, AppError> {
        get(&self.connection, id)
    }

    /// Reads an entry's audio as a playable WAV file, or `None` when it has none.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the entry's audio file cannot be read.
    pub fn audio(&self, id: EntryId) -> Result<Option<Vec<u8>>, AppError> {
        let Some(entry) = self.get(id)? else {
            return Ok(None);
        };
        if !entry.has_audio {
            return Ok(None);
        }
        match fs::read(audio_path(&self.audio_dir, id)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(history_error("history_audio_read")),
        }
    }

    /// Deletes one entry and its audio; returns whether it existed.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the entry cannot be deleted.
    pub fn delete(&self, id: EntryId) -> Result<bool, AppError> {
        let deleted = self
            .connection
            .execute("DELETE FROM entries WHERE id = ?1", [sql_id(id)?])
            .map_err(|_| history_error("history_delete"))?;
        remove_audio(&self.audio_dir, id);
        Ok(deleted > 0)
    }

    /// Deletes every entry and every audio file; returns whether anything was removed.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when history cannot be cleared.
    pub fn clear(&self) -> Result<bool, AppError> {
        let deleted = self
            .connection
            .execute("DELETE FROM entries", [])
            .map_err(|_| history_error("history_clear"))?;
        let removed = remove_orphan_audio(&self.connection, &self.audio_dir);
        Ok(deleted > 0 || removed > 0)
    }

    /// Deletes the entries beyond `limit`, with their audio; returns how many were deleted.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when pruning fails.
    pub fn prune(&self, limit: HistoryLimit) -> Result<usize, AppError> {
        prune(&self.connection, &self.audio_dir, limit)
    }

    /// Closes the database before its files are deleted or the application exits.
    ///
    /// # Errors
    ///
    /// Returns a content-free error when the database does not close cleanly.
    pub fn close(self) -> Result<(), AppError> {
        self.connection
            .close()
            .map_err(|_| history_error("history_close"))
    }
}

fn get(connection: &Connection, id: EntryId) -> Result<Option<HistoryEntry>, AppError> {
    connection
        .query_row(
            &format!("SELECT {COLUMNS} FROM entries WHERE id = ?1"),
            [sql_id(id)?],
            entry_from_row,
        )
        .optional()
        .map_err(|_| history_error("history_read"))
}

fn prune(
    connection: &Connection,
    audio_dir: &Path,
    limit: HistoryLimit,
) -> Result<usize, AppError> {
    let Some(kept) = limit.max_entries() else {
        return Ok(0);
    };
    let pruned: Vec<i64> = connection
        .prepare("SELECT id FROM entries ORDER BY created_at_ms DESC, id DESC LIMIT -1 OFFSET ?1")
        .and_then(|mut statement| {
            statement
                .query_map([i64::from(kept)], |row| row.get(0))?
                .collect()
        })
        .map_err(|_| history_error("history_prune"))?;
    for id in &pruned {
        connection
            .execute("DELETE FROM entries WHERE id = ?1", [id])
            .map_err(|_| history_error("history_prune"))?;
        remove_audio(audio_dir, EntryId(id.unsigned_abs()));
    }
    Ok(pruned.len())
}

/// Deletes files in the audio directory that no entry owns, such as one left by a crash between
/// writing a file and its row; returns how many were deleted. Callers serialize their use of
/// the store, so no insert is writing a file meanwhile.
fn remove_orphan_audio(connection: &Connection, audio_dir: &Path) -> usize {
    let owned: Result<HashSet<i64>, _> = connection
        .prepare("SELECT id FROM entries WHERE has_audio = 1")
        .and_then(|mut statement| statement.query_map([], |row| row.get(0))?.collect());
    let (Ok(owned), Ok(files)) = (owned, fs::read_dir(audio_dir)) else {
        return 0;
    };
    let mut removed = 0;
    for file in files.flatten() {
        let path = file.path();
        let owned_by_entry = path.extension().is_some_and(|extension| extension == "wav")
            && path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<i64>().ok())
                .is_some_and(|id| owned.contains(&id));
        if !owned_by_entry
            && file.file_type().is_ok_and(|kind| kind.is_file())
            && fs::remove_file(&path).is_ok()
        {
            removed += 1;
        }
    }
    removed
}

fn entry_from_row(row: &Row<'_>) -> rusqlite::Result<HistoryEntry> {
    let status: String = row.get(5)?;
    let asr_model_id = row
        .get::<_, Option<String>>(8)?
        .map(ModelId::new)
        .transpose()
        .map_err(|error| rusqlite::Error::FromSqlConversionFailure(8, Type::Text, error.into()))?;
    Ok(HistoryEntry {
        id: EntryId(row.get::<_, i64>(0)?.unsigned_abs()),
        created_at: UtcMillis(row.get(1)?),
        duration_millis: row.get::<_, Option<i64>>(2)?.map(i64::unsigned_abs),
        raw_text: row.get(3)?,
        cleaned_text: row.get(4)?,
        status: parse_status(&status).ok_or_else(|| {
            rusqlite::Error::FromSqlConversionFailure(5, Type::Text, "unknown entry status".into())
        })?,
        error: row.get(6)?,
        has_audio: row.get(7)?,
        asr_model_id,
        llm_model: row.get(9)?,
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
    let temporary = tempfile::NamedTempFile::new_in(audio_dir)?;
    {
        let mut writer = WavWriter::new(
            io::BufWriter::new(temporary.as_file()),
            WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE_HZ,
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

/// Escapes `LIKE` wildcards so the search text matches literally.
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

fn sql_id(id: EntryId) -> Result<i64, AppError> {
    i64::try_from(id.0).map_err(|_| history_error("history_id"))
}

fn entry_id(row_id: i64) -> Result<EntryId, AppError> {
    u64::try_from(row_id)
        .map(EntryId)
        .map_err(|_| history_error("history_id"))
}

fn sql_u64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn history_error(code: &'static str) -> AppError {
    AppError::redacted(code, "Transcription history is unavailable")
}

#[cfg(test)]
mod tests;
