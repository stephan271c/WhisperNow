use std::fs;

use rusqlite::Connection;
use serde_json::json;
use whispernow_domain::{
    EntryStatus, ErrorCategory, HistoryFilter, HistoryLimit, HistoryPageRequest, ModelId,
    NewHistoryEntry, UtcMillis,
};

use super::SqliteHistoryRepository;
use crate::AppPaths;

struct Fixture {
    _root: tempfile::TempDir,
    paths: AppPaths,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temporary directory");
        let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
        Self { _root: root, paths }
    }

    fn open(&self) -> SqliteHistoryRepository {
        SqliteHistoryRepository::open(&self.paths)
            .expect("open history")
            .0
    }

    fn wav_names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(self.paths.recordings_dir())
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }
}

fn entry(created_at: i64, status: EntryStatus, text: &str) -> NewHistoryEntry {
    NewHistoryEntry {
        created_at: UtcMillis(created_at),
        duration_millis: Some(1_000),
        raw_text: (status != EntryStatus::Failed).then(|| text.to_owned()),
        cleaned_text: None,
        status,
        error: (status == EntryStatus::Failed).then(|| "Speech recognition failed".to_owned()),
        asr_model_id: Some(ModelId::new("parakeet").expect("model id")),
    }
}

const SPEECH: [f32; 4] = [0.0, 0.5, -0.5, 0.25];

fn all(repository: &SqliteHistoryRepository) -> Vec<whispernow_domain::HistoryEntry> {
    repository
        .list(&HistoryPageRequest {
            limit: 100,
            filter: HistoryFilter::default(),
        })
        .expect("list")
        .entries
}

#[test]
fn an_entry_round_trips_and_only_kept_audio_is_written() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let spoken = history
        .insert(
            NewHistoryEntry {
                cleaned_text: Some("Hello.".to_owned()),
                ..entry(1, EntryStatus::Ok, "hello")
            },
            &SPEECH,
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    let spoken = history.get(spoken).expect("get").expect("entry");
    assert_eq!(spoken.raw_text.as_deref(), Some("hello"));
    assert_eq!(spoken.text(), Some("Hello."));
    assert!(!spoken.audio_kept);
    assert_eq!(history.audio(spoken.id).expect("audio"), None);

    let kept = history
        .insert(
            entry(2, EntryStatus::Ok, "kept"),
            &SPEECH,
            true,
            HistoryLimit::Last20,
        )
        .expect("insert");
    let kept = history.get(kept).expect("get").expect("entry");
    assert!(kept.audio_kept);
    let wav = history.audio(kept.id).expect("audio").expect("kept audio");
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(fixture.wav_names(), vec![format!("{}.wav", kept.id.0)]);
    assert_eq!(history.get(kept.id).expect("get"), Some(kept));
}

#[test]
fn history_reads_do_not_parse_the_asr_model_id_saved_for_training() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let id = history
        .insert(
            entry(1, EntryStatus::Ok, "dictation"),
            &[],
            false,
            HistoryLimit::Unlimited,
        )
        .expect("insert");
    let stored: String = history
        .connection
        .query_row(
            "SELECT asr_model_id FROM entries WHERE id = ?1",
            [id.0],
            |row| row.get(0),
        )
        .expect("training model id");
    assert_eq!(stored, "parakeet");
    history
        .connection
        .execute(
            "UPDATE entries SET asr_model_id = '../retired/model' WHERE id = ?1",
            [id.0],
        )
        .expect("unrecognized stored model id");

    let saved = history.get(id).expect("read history").expect("entry");
    assert_eq!(saved.text(), Some("dictation"));
    assert_eq!(all(&history), vec![saved]);
}

#[test]
fn failed_and_cancelled_entries_keep_their_audio_when_keep_audio_is_off() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let failed = history
        .insert(
            entry(1, EntryStatus::Failed, ""),
            &SPEECH,
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    let cancelled = history
        .insert(
            entry(2, EntryStatus::Cancelled, "partial"),
            &SPEECH,
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    let failed = history.get(failed).expect("get").expect("entry");
    let cancelled = history.get(cancelled).expect("get").expect("entry");
    assert!(failed.audio_kept && cancelled.audio_kept);
    assert_eq!(failed.error.as_deref(), Some("Speech recognition failed"));
    assert_eq!(fixture.wav_names().len(), 2);
}

#[test]
fn inserting_beyond_the_limit_deletes_the_oldest_entries_with_their_audio() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let mut ids = Vec::new();
    for index in 0..25 {
        ids.push(
            history
                .insert(
                    entry(index, EntryStatus::Ok, &format!("entry {index}")),
                    &SPEECH,
                    true,
                    HistoryLimit::Last20,
                )
                .expect("insert"),
        );
    }
    let kept = all(&history);
    assert_eq!(kept.len(), 20);
    assert_eq!(kept[0].raw_text.as_deref(), Some("entry 24"));
    assert_eq!(kept[19].raw_text.as_deref(), Some("entry 5"));
    assert_eq!(fixture.wav_names().len(), 20);
    assert!(!fixture.wav_names().contains(&format!("{}.wav", ids[4].0)));

    assert_eq!(history.prune(HistoryLimit::Unlimited).expect("prune"), 0);
    assert_eq!(history.prune(HistoryLimit::Last20).expect("prune"), 0);
}

#[test]
fn listing_pages_newest_first_and_filters_by_text_and_failure() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let mut notes = Vec::new();
    for index in 0..5 {
        let note = history
            .insert(
                entry(index.min(1), EntryStatus::Ok, &format!("note {index} 100%")),
                &[],
                false,
                HistoryLimit::Unlimited,
            )
            .expect("insert");
        notes.push(note);
    }
    let failed = history
        .insert(
            entry(9, EntryStatus::Failed, ""),
            &[],
            false,
            HistoryLimit::Unlimited,
        )
        .expect("insert");

    let first = history
        .list(&HistoryPageRequest {
            limit: 4,
            filter: HistoryFilter::default(),
        })
        .expect("first page");
    assert_eq!(
        first
            .entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [failed, notes[4], notes[3], notes[2]]
    );
    assert!(first.has_more);
    let second = history
        .list(&HistoryPageRequest {
            limit: 8,
            filter: HistoryFilter::default(),
        })
        .expect("second page");
    assert_eq!(
        second
            .entries
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        [failed, notes[4], notes[3], notes[2], notes[1], notes[0]]
    );
    assert_eq!(second.entries[..first.entries.len()], first.entries);
    assert!(!second.has_more);

    let search = |query: &str, failed_only| {
        history
            .list(&HistoryPageRequest::first_page(HistoryFilter {
                query: query.to_owned(),
                failed_only,
            }))
            .expect("search")
            .entries
            .len()
    };
    assert_eq!(search("NOTE 3", false), 1);
    assert_eq!(search("100%", false), 5);
    assert_eq!(search("1_0", false), 0);
    assert_eq!(search("  ", true), 1);
}

#[test]
fn delivery_failure_delete_and_clear_update_entries_and_audio() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let first = history
        .insert(
            entry(1, EntryStatus::Ok, "one"),
            &SPEECH,
            true,
            HistoryLimit::Last20,
        )
        .expect("insert");
    let second = history
        .insert(
            entry(2, EntryStatus::Ok, "two"),
            &SPEECH,
            true,
            HistoryLimit::Last20,
        )
        .expect("insert");
    history
        .mark_not_delivered(first, "The paste failed")
        .expect("mark");
    let first = history.get(first).expect("get").expect("entry");
    assert_eq!(first.status, EntryStatus::NotDelivered);
    assert_eq!(first.error.as_deref(), Some("The paste failed"));

    assert!(history.delete(first.id).expect("delete"));
    assert!(!history.delete(first.id).expect("delete again"));
    assert_eq!(fixture.wav_names(), vec![format!("{}.wav", second.0)]);
    assert!(history.clear().expect("clear"));
    assert!(!history.clear().expect("clear again"));
    assert!(all(&history).is_empty());
    assert!(fixture.wav_names().is_empty());
}

#[test]
fn ids_are_never_reused_after_a_delete() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let first = history
        .insert(
            entry(1, EntryStatus::Ok, "one"),
            &[],
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    history.delete(first).expect("delete");
    let second = history
        .insert(
            entry(2, EntryStatus::Ok, "two"),
            &[],
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    assert!(second > first);
}

#[test]
fn clearing_vacuums_transcript_pages_without_reusing_ids() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let private_text = "private-transcript-to-remove".repeat(1_000);
    let first = history
        .insert(
            entry(1, EntryStatus::Ok, &private_text),
            &SPEECH,
            true,
            HistoryLimit::Unlimited,
        )
        .expect("insert");
    let database = fixture.paths.history_database_file();
    let before = fs::metadata(&database).expect("database").len();

    assert!(history.clear().expect("clear"));

    assert!(all(&history).is_empty());
    assert!(!fixture.paths.recordings_dir().exists());
    assert!(fs::metadata(&database).expect("vacuumed database").len() < before);
    let bytes = fs::read(&database).expect("database bytes");
    assert!(
        !bytes
            .windows(b"private-transcript-to-remove".len())
            .any(|window| window == b"private-transcript-to-remove")
    );
    let later = history
        .insert(
            entry(2, EntryStatus::Ok, "after clear"),
            &SPEECH,
            true,
            HistoryLimit::Unlimited,
        )
        .expect("insert after clear");
    assert!(later > first);
    assert!(history.audio(later).expect("audio").is_some());
}

#[cfg(unix)]
#[test]
fn a_recordings_removal_failure_keeps_history_rows() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = Fixture::new();
    let mut history = fixture.open();
    let id = history
        .insert(
            entry(1, EntryStatus::Ok, "recoverable"),
            &SPEECH,
            true,
            HistoryLimit::Unlimited,
        )
        .expect("insert");
    let recordings = fixture.paths.recordings_dir();
    let permissions = fs::metadata(&recordings).expect("recordings").permissions();
    fs::set_permissions(&recordings, fs::Permissions::from_mode(0o500))
        .expect("read-only recordings");

    let result = history.clear();

    fs::set_permissions(&recordings, permissions).expect("restore permissions");
    assert_eq!(
        result.expect_err("recordings kept").code,
        "history_clear_audio"
    );
    assert_eq!(
        history
            .get(id)
            .expect("entry")
            .expect("retained")
            .raw_text
            .as_deref(),
        Some("recoverable")
    );
    assert!(history.audio(id).expect("audio").is_some());
}

#[test]
fn startup_removes_audio_files_that_no_entry_owns() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let kept = history
        .insert(
            entry(1, EntryStatus::Failed, ""),
            &SPEECH,
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    drop(history);
    let audio_dir = fixture.paths.recordings_dir();
    fs::write(audio_dir.join("9999.wav"), b"orphan").expect("orphan");
    fs::write(audio_dir.join(".history-audio-crash.tmp"), b"partial").expect("temporary");
    fs::write(audio_dir.join("20260101T000000.000000Z.wav"), b"old").expect("old name");
    fs::write(audio_dir.join("notes.txt"), b"keep").expect("user file");

    let reopened = fixture.open();
    let mut expected = vec![
        format!("{}.wav", kept.0),
        "20260101T000000.000000Z.wav".to_owned(),
        "notes.txt".to_owned(),
    ];
    expected.sort();
    assert_eq!(fixture.wav_names(), expected);
    assert!(reopened.audio(kept).expect("audio").is_some());
}

#[test]
fn an_entry_with_an_unknown_status_fails_to_load() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let stored = history
        .insert(
            entry(1, EntryStatus::Ok, "one"),
            &[],
            false,
            HistoryLimit::Last20,
        )
        .expect("insert");
    drop(history);
    Connection::open(fixture.paths.history_database_file())
        .expect("database")
        .execute("UPDATE entries SET status = 'lost'", [])
        .expect("unknown status");
    let error = fixture.open().get(stored).expect_err("unknown status");
    assert_eq!(error.category, ErrorCategory::General);
}

#[test]
fn a_failed_open_reports_history_unavailable() {
    let fixture = Fixture::new();
    let database = fixture.paths.history_database_file();
    fs::create_dir_all(database.parent().expect("database directory")).expect("directory");
    fs::write(database, "not a SQLite database").expect("invalid database");
    let Err(error) = SqliteHistoryRepository::open(&fixture.paths) else {
        panic!("the database cannot be opened")
    };
    assert_eq!(error.category, ErrorCategory::HistoryUnavailable);
    assert_eq!(error.code, "history_unavailable");
}

#[test]
fn a_python_history_document_imports_once_as_entries() {
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.paths.config_dir()).expect("config directory");
    let records = json!([
        {
            "timestamp": "1720000000",
            "raw_text": "seconds",
            "enhanced_text": "Seconds.",
            "enhancement_name": "Fix Grammar & Spelling",
            "cost_usd": 0.00042
        },
        {
            "timestamp": "2026-08-25T12:34:56.789Z",
            "raw_text": "[ERROR]",
            "enhanced_text": null,
            "enhancement_name": null,
            "cost_usd": null
        },
        { "timestamp": "2026-08-25T12:34:56.123456", "raw_text": "naive UTC" },
        { "timestamp": "legacy-unparseable", "raw_text": "stable fallback" }
    ]);
    fs::write(fixture.paths.history_file(), records.to_string()).expect("write legacy JSON");

    let history = fixture.open();
    let imported = all(&history);
    assert_eq!(imported.len(), 4);
    let find = |text: &str| {
        imported
            .iter()
            .find(|entry| entry.raw_text.as_deref() == Some(text))
            .expect("imported entry")
    };
    assert_eq!(find("seconds").created_at, UtcMillis(1_720_000_000_000));
    assert_eq!(find("seconds").cleaned_text.as_deref(), Some("Seconds."));
    assert_eq!(find("naive UTC").created_at, UtcMillis(1_787_661_296_123));
    assert_eq!(find("stable fallback").created_at, UtcMillis(0));
    let failed = imported
        .iter()
        .find(|entry| entry.status == EntryStatus::Failed)
        .expect("failed record");
    assert_eq!(failed.raw_text, None);
    assert!(!fixture.paths.history_file().exists());
    assert!(fixture.paths.migrated_history_file().exists());

    drop(history);
    assert_eq!(all(&fixture.open()).len(), 4, "imported once");
}

#[test]
fn a_malformed_python_history_document_is_moved_aside_with_a_diagnostic() {
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.paths.config_dir()).expect("config directory");
    fs::write(fixture.paths.history_file(), b"{not json").expect("malformed");
    let (history, diagnostic) = SqliteHistoryRepository::open(&fixture.paths).expect("open");
    assert_eq!(
        diagnostic.map(|diagnostic| diagnostic.code).as_deref(),
        Some("history_import_invalid")
    );
    assert!(all(&history).is_empty());
    let backups = fixture
        .paths
        .history_backup_files()
        .expect("history backups");
    assert_eq!(backups.len(), 1);
    assert!(
        backups[0]
            .file_name()
            .expect("backup name")
            .to_string_lossy()
            .starts_with("history.json.bad-")
    );
    assert_eq!(fs::read(&backups[0]).expect("backup content"), b"{not json");
    assert!(!fixture.paths.history_file().exists());
}

mod transactions;
