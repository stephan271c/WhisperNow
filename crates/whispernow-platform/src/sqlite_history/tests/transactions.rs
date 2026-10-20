use super::*;

#[test]
fn deleting_retained_audio_updates_list_results_and_playback() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    let id = history
        .insert(
            entry(1, EntryStatus::Ok, "spoken"),
            &SPEECH,
            true,
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("insert");
    fs::remove_file(fixture.paths.recordings_dir().join(format!("{}.wav", id.0)))
        .expect("manual deletion");
    assert!(!all(&history)[0].audio_kept);
    assert_eq!(history.audio(id).expect("missing audio"), None);
}

#[test]
fn a_failed_commit_rolls_back_the_entry_and_removes_audio_before_id_reuse() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    history
        .connection
        .execute_batch(
            "
        PRAGMA foreign_keys = ON;
        CREATE TABLE parents (id INTEGER PRIMARY KEY);
        CREATE TABLE dependent (id INTEGER REFERENCES parents(id) DEFERRABLE INITIALLY DEFERRED);
        CREATE TRIGGER fail_commit AFTER INSERT ON entries BEGIN
            INSERT INTO dependent VALUES (NEW.id);
        END;
    ",
        )
        .expect("deferred constraint");
    assert!(
        history
            .insert(
                entry(1, EntryStatus::Ok, "rolled back"),
                &SPEECH,
                true,
                HistoryLimit::Unlimited,
                HistoryRetention::Forever,
            )
            .is_err()
    );
    assert!(all(&history).is_empty());
    assert!(fixture.wav_names().is_empty());
    history
        .connection
        .execute_batch("DROP TRIGGER fail_commit")
        .expect("allow commits");
    let id = history
        .insert(
            entry(2, EntryStatus::Ok, "next"),
            &[],
            false,
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("next insert");
    assert_eq!(id.0, 1);
    assert!(!all(&history)[0].audio_kept);
    assert_eq!(history.audio(id).expect("no stale audio"), None);
}

#[test]
fn a_failed_prune_rolls_back_the_new_entry_and_preserves_owned_audio() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    for created_at in 0..20 {
        history
            .insert(
                entry(created_at, EntryStatus::Ok, "kept"),
                &SPEECH,
                true,
                HistoryLimit::Unlimited,
                HistoryRetention::Forever,
            )
            .expect("existing entry");
    }
    history.connection.execute_batch("CREATE TRIGGER refuse_prune BEFORE DELETE ON entries BEGIN SELECT RAISE(ABORT, 'refused'); END;").expect("refuse deletion");
    assert!(
        history
            .insert(
                entry(20, EntryStatus::Ok, "rolled back"),
                &SPEECH,
                true,
                HistoryLimit::Last20,
                HistoryRetention::Forever,
            )
            .is_err()
    );
    assert_eq!(all(&history).len(), 20);
    assert_eq!(fixture.wav_names().len(), 20);
    assert!(all(&history).iter().all(|entry| entry.audio_kept));
}

#[cfg(unix)]
#[test]
fn a_reused_id_cannot_acquire_audio_that_could_not_be_cleaned_up() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = Fixture::new();
    let mut history = fixture.open();
    let recordings = fixture.paths.recordings_dir();
    fs::create_dir_all(&recordings).expect("recordings");
    fs::write(recordings.join("1.wav"), b"old audio").expect("leftover rollback audio");
    let permissions = fs::metadata(&recordings).expect("directory").permissions();
    fs::set_permissions(&recordings, fs::Permissions::from_mode(0o500)).expect("refuse cleanup");
    let result = history.insert(
        entry(1, EntryStatus::Ok, "new transcript"),
        &[],
        false,
        HistoryLimit::Unlimited,
        HistoryRetention::Forever,
    );
    fs::set_permissions(&recordings, permissions).expect("allow cleanup");
    assert_eq!(
        result.expect_err("stale audio cannot attach").code,
        "history_audio_delete"
    );
    assert!(all(&history).is_empty());
    let id = history
        .insert(
            entry(2, EntryStatus::Ok, "retry"),
            &[],
            false,
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("retry");
    assert_eq!(id.0, 1);
    assert_eq!(history.audio(id).expect("no stale audio"), None);
    assert!(!all(&history)[0].audio_kept);
}

#[test]
fn an_audio_write_failure_still_saves_the_transcript() {
    let fixture = Fixture::new();
    let mut history = fixture.open();
    fs::write(fixture.paths.recordings_dir(), b"obstructed").expect("block recordings directory");
    history
        .insert(
            entry(1, EntryStatus::Ok, "transcript kept"),
            &SPEECH,
            true,
            HistoryLimit::Unlimited,
            HistoryRetention::Forever,
        )
        .expect("transcript saved");
    let entries = all(&history);
    assert_eq!(entries[0].raw_text.as_deref(), Some("transcript kept"));
    assert!(!entries[0].audio_kept);
}
