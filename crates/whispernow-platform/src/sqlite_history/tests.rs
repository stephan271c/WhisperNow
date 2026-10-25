use std::fs;

use rusqlite::Connection;
use serde_json::json;
use whispernow_domain::{
    AppError, EntryId, EntryStatus, ErrorCategory, HistoryEntry, HistoryFilter, HistoryLimit,
    HistoryPageRequest, HistoryRetention, ModelId, NewHistoryEntry, UtcMillis,
};

use super::SqliteHistoryRepository;
use crate::{AppPaths, atomic_file::TEMPORARY_PREFIX};

struct Fixture {
    _root: tempfile::TempDir,
    paths: AppPaths,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temporary directory");
        let paths = AppPaths::under(root.path());
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
        duration_millis: 1_000,
        raw_text: (status != EntryStatus::Failed).then(|| text.to_owned()),
        cleaned_text: None,
        status,
        error: (status == EntryStatus::Failed).then(|| "Speech recognition failed".to_owned()),
        asr_model_id: ModelId::new("parakeet").expect("model id"),
    }
}

const SPEECH: &[f32] = &[0.0, 0.5, -0.5, 0.25];

fn store(
    history: &mut SqliteHistoryRepository,
    entry: NewHistoryEntry,
    audio: Option<&[f32]>,
) -> Result<EntryId, AppError> {
    history.insert(
        entry,
        audio,
        HistoryLimit::Unlimited,
        HistoryRetention::Forever,
    )
}

fn all(repository: &SqliteHistoryRepository) -> Vec<HistoryEntry> {
    repository
        .list(&HistoryPageRequest {
            limit: 100,
            filter: HistoryFilter::default(),
        })
        .expect("list")
        .entries
}

fn find(repository: &SqliteHistoryRepository, id: EntryId) -> HistoryEntry {
    all(repository)
        .into_iter()
        .find(|entry| entry.id == id)
        .expect("entry")
}

#[test]
fn an_entry_round_trips_and_only_kept_audio_is_written() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let spoken = store(
        &mut history,
        NewHistoryEntry {
            cleaned_text: Some("Hello.".to_owned()),
            ..entry(1, EntryStatus::Ok, "hello")
        },
        None,
    )
    .expect("insert");
    let spoken = find(&history, spoken);
    assert_eq!(spoken.raw_text.as_deref(), Some("hello"));
    assert_eq!(spoken.text(), Some("Hello."));
    assert!(!spoken.audio_kept);
    assert_eq!(history.audio(spoken.id).expect("audio"), None);

    let kept = store(
        &mut history,
        entry(2, EntryStatus::Ok, "kept"),
        Some(SPEECH),
    )
    .expect("insert");
    let kept = find(&history, kept);
    assert!(kept.audio_kept);
    let wav = history.audio(kept.id).expect("audio").expect("kept audio");
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(fixture.wav_names(), vec![format!("{}.wav", kept.id.0)]);
    assert_eq!(all(&history).first(), Some(&kept));
}

#[test]
fn every_entry_status_reads_back() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    for status in EntryStatus::ALL {
        let id = store(&mut history, entry(1, status, "text"), None).expect("insert");
        assert_eq!(find(&history, id).status, status);
    }
}

#[test]
fn history_reads_do_not_parse_the_asr_model_id_saved_for_training() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let id = store(&mut history, entry(1, EntryStatus::Ok, "dictation"), None).expect("insert");
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

    let saved = all(&history);
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].id, id);
    assert_eq!(saved[0].text(), Some("dictation"));
}

#[test]
fn failed_and_cancelled_entries_keep_the_audio_they_are_given() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let mut insert = |status, keep_audio: bool| {
        let id = store(
            &mut history,
            entry(1, status, "partial"),
            keep_audio.then_some(SPEECH),
        )
        .expect("insert");
        find(&history, id)
    };
    for status in [EntryStatus::Failed, EntryStatus::Cancelled] {
        assert!(!insert(status, false).audio_kept, "{status:?}");
        assert!(insert(status, true).audio_kept, "{status:?}");
    }
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
                    Some(SPEECH),
                    HistoryLimit::Last20,
                    HistoryRetention::Forever,
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

    assert_eq!(
        history
            .prune(HistoryLimit::Unlimited, HistoryRetention::Forever)
            .expect("prune"),
        0
    );
    assert_eq!(
        history
            .prune(HistoryLimit::Last20, HistoryRetention::Forever)
            .expect("prune"),
        0
    );
}

#[test]
fn an_entry_older_than_the_retention_is_deleted_with_its_audio() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let now = UtcMillis::now().get();
    let day = 86_400_000;
    let old = history
        .insert(
            entry(now - 10 * day, EntryStatus::Ok, "old"),
            Some(SPEECH),
            HistoryLimit::Unlimited,
            HistoryRetention::Days7,
        )
        .expect("insert");
    let fresh = history
        .insert(
            entry(now, EntryStatus::Ok, "fresh"),
            Some(SPEECH),
            HistoryLimit::Unlimited,
            HistoryRetention::Days7,
        )
        .expect("insert");
    assert_eq!(fixture.wav_names(), vec![format!("{}.wav", fresh.0)]);

    let kept = all(&history);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].id, fresh);
    assert_ne!(kept[0].id, old);
}

#[test]
fn a_forever_retention_keeps_entries_of_any_age() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let ancient = store(
        &mut history,
        entry(
            UtcMillis::now().get() - 365 * 86_400_000,
            EntryStatus::Ok,
            "ancient",
        ),
        Some(SPEECH),
    )
    .expect("insert");
    assert_eq!(all(&history).len(), 1);
    assert_eq!(fixture.wav_names(), vec![format!("{}.wav", ancient.0)]);
}

#[test]
fn pruning_deletes_entries_that_aged_out() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let now = UtcMillis::now().get();
    let day = 86_400_000;
    let mut ids = Vec::new();
    for age_days in [10, 8, 2, 0] {
        ids.push(
            store(
                &mut history,
                entry(
                    now - age_days * day,
                    EntryStatus::Ok,
                    &format!("entry {age_days}"),
                ),
                Some(SPEECH),
            )
            .expect("insert"),
        );
    }
    assert_eq!(
        history
            .prune(HistoryLimit::Unlimited, HistoryRetention::Days7)
            .expect("prune"),
        2
    );
    let kept = all(&history);
    assert_eq!(
        kept.iter().map(|entry| entry.id).collect::<Vec<_>>(),
        vec![ids[3], ids[2]]
    );
    assert_eq!(fixture.wav_names().len(), 2);
}

#[test]
fn listing_pages_newest_first_and_filters_by_text_and_failure() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let mut notes = Vec::new();
    for index in 0..5 {
        let note = store(
            &mut history,
            entry(index.min(1), EntryStatus::Ok, &format!("note {index} 100%")),
            None,
        )
        .expect("insert");
        notes.push(note);
    }
    let failed = store(&mut history, entry(9, EntryStatus::Failed, ""), None).expect("insert");

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
    let first =
        store(&mut history, entry(1, EntryStatus::Ok, "one"), Some(SPEECH)).expect("insert");
    let second =
        store(&mut history, entry(2, EntryStatus::Ok, "two"), Some(SPEECH)).expect("insert");
    history
        .mark_not_delivered(first, "The paste failed")
        .expect("mark");
    let first = find(&history, first);
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
    let first = store(&mut history, entry(1, EntryStatus::Ok, "one"), None).expect("insert");
    history.delete(first).expect("delete");
    let second = store(&mut history, entry(2, EntryStatus::Ok, "two"), None).expect("insert");
    assert!(second > first);
}

#[test]
fn clearing_vacuums_transcript_pages_without_reusing_ids() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let private_text = "private-transcript-to-remove".repeat(1_000);
    let first = store(
        &mut history,
        entry(1, EntryStatus::Ok, &private_text),
        Some(SPEECH),
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
    let later = store(
        &mut history,
        entry(2, EntryStatus::Ok, "after clear"),
        Some(SPEECH),
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
    let id = store(
        &mut history,
        entry(1, EntryStatus::Ok, "recoverable"),
        Some(SPEECH),
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
    assert_eq!(find(&history, id).raw_text.as_deref(), Some("recoverable"));
    assert!(history.audio(id).expect("audio").is_some());
}

#[test]
fn startup_removes_audio_files_that_no_entry_owns() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let kept = store(
        &mut history,
        entry(1, EntryStatus::Failed, ""),
        Some(SPEECH),
    )
    .expect("insert");
    drop(history);
    let audio_dir = fixture.paths.recordings_dir();
    fs::write(audio_dir.join("9999.wav"), b"orphan").expect("orphan");
    fs::write(
        audio_dir.join(format!("{TEMPORARY_PREFIX}crash")),
        b"partial",
    )
    .expect("temporary");
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
    store(&mut history, entry(1, EntryStatus::Ok, "one"), None).expect("insert");
    drop(history);
    Connection::open(fixture.paths.history_database_file())
        .expect("database")
        .execute("UPDATE entries SET status = 'lost'", [])
        .expect("unknown status");
    let error = fixture
        .open()
        .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
        .expect_err("unknown status");
    assert_eq!(error.category, ErrorCategory::General);
}

#[test]
fn constraint_failure_messages_name_columns_not_row_values() {
    let connection = Connection::open_in_memory().expect("database");
    connection.execute_batch(super::SCHEMA).expect("schema");
    connection
        .execute(
            "INSERT INTO entries (id, created_at_ms, raw_text, status) VALUES (1, 1, 'first words', 'ok')",
            [],
        )
        .expect("first row");
    for (statement, column) in [
        (
            "INSERT INTO entries (id, created_at_ms, raw_text, status) VALUES (1, 2, 'second words', 'ok')",
            "entries.id",
        ),
        (
            "INSERT INTO entries (created_at_ms, raw_text, status) VALUES (NULL, 'third words', 'ok')",
            "entries.created_at_ms",
        ),
    ] {
        let message = connection
            .execute(statement, [])
            .expect_err("constraint failure")
            .to_string();
        assert!(message.contains(column), "{message}");
        assert!(!message.contains("words"), "{message}");
    }
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
