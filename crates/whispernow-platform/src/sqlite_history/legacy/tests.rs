use super::*;
use crate::SqliteHistoryRepository;
use whispernow_domain::{HistoryFilter, HistoryPageRequest};

#[test]
fn a_failed_legacy_rename_rolls_back_and_retries_without_duplicates() {
    let root = tempfile::tempdir().expect("temporary directory");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    fs::create_dir_all(paths.config_dir()).expect("config directory");
    let document = br#"[{"timestamp":"1720000000","raw_text":"import once"}]"#;
    fs::write(paths.history_file(), document).expect("legacy history");
    fs::create_dir_all(paths.migrated_history_file()).expect("blocked destination");
    fs::write(paths.migrated_history_file().join("keep"), b"obstruction")
        .expect("nonempty destination directory");

    for _ in 0..2 {
        let (history, diagnostic) = SqliteHistoryRepository::open(&paths).expect("history opens");
        assert_eq!(
            diagnostic.expect("import diagnostic").code,
            "history_import_rename"
        );
        assert!(
            history
                .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
                .expect("history remains usable")
                .entries
                .is_empty()
        );
        assert_eq!(
            fs::read(paths.history_file()).expect("source preserved"),
            document
        );
    }

    fs::remove_dir_all(paths.migrated_history_file()).expect("unblock destination");
    for _ in 0..2 {
        let (history, diagnostic) = SqliteHistoryRepository::open(&paths).expect("history opens");
        assert!(diagnostic.is_none());
        let entries = history
            .list(&HistoryPageRequest::first_page(HistoryFilter::default()))
            .expect("imported history")
            .entries;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].raw_text.as_deref(), Some("import once"));
    }
    assert!(!paths.history_file().exists());
    assert_eq!(
        fs::read(paths.migrated_history_file()).expect("imported copy"),
        document
    );
}

#[test]
fn a_failed_import_commit_restores_the_legacy_document() {
    let root = tempfile::tempdir().expect("temporary directory");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    fs::create_dir_all(paths.config_dir()).expect("config directory");
    let document = br#"[{"timestamp":"1720000000","raw_text":"retry after commit failure"}]"#;
    fs::write(paths.history_file(), document).expect("legacy history");
    let mut connection = Connection::open_in_memory().expect("database");
    connection
        .execute_batch(super::super::SCHEMA)
        .expect("schema");
    connection
        .execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE import_parent (id INTEGER PRIMARY KEY);
             CREATE TABLE import_child (
                 parent_id INTEGER REFERENCES import_parent(id) DEFERRABLE INITIALLY DEFERRED
             );
             CREATE TRIGGER reject_import_commit AFTER INSERT ON entries
             BEGIN
                 INSERT INTO import_child (parent_id) VALUES (1);
             END;",
        )
        .expect("deferred commit failure");

    let error = import_legacy(&mut connection, &paths).expect_err("commit rejected");

    assert_eq!(error, "history_import_commit");
    assert_eq!(
        fs::read(paths.history_file()).expect("source restored"),
        document
    );
    assert!(!paths.migrated_history_file().exists());
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM entries", [], |row| row
                .get::<_, i64>(0))
            .expect("rolled back rows"),
        0
    );
    connection
        .execute_batch("DROP TRIGGER reject_import_commit")
        .expect("remove commit failure");
    assert!(
        import_legacy(&mut connection, &paths)
            .expect("retry import")
            .is_none()
    );
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM entries", [], |row| row
                .get::<_, i64>(0))
            .expect("imported rows"),
        1
    );
}
