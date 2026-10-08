use dioxus::prelude::*;
use whispernow_api::{
    AppStateDto, DeleteHistoryEntryRequest, EntryId, EntryStatus, HistoryCursor, HistoryEntry,
    HistoryFilter, HistoryPage, HistoryPageRequest, PhaseDto, SettingsView, UtcMillis,
};

use super::format::group_by_day;
use super::{HistoryScreen, LocalClock};
use crate::client::fake;
use crate::test_support::{Page, Probe, show_history, snapshot, tag, text};
use crate::{HistoryCopied, ManagementState, Route, SettingsRoute, UiState};

/// 2026-09-27 10:30 UTC.
const NOW: UtcMillis = UtcMillis(1_790_505_000_000);
const HOUR: i64 = 3_600_000;

/// The page with a fixed clock, wired to the fake desktop.
#[component]
fn Surface(management_state: ManagementState, snapshot: AppStateDto, probe: Probe) -> Element {
    let state = use_signal(move || {
        let mut state = UiState::default();
        state.hydrate(snapshot);
        state
    });
    let management = use_signal(move || management_state);
    probe.provide(state, management);
    use_context_provider(|| LocalClock::fixed(NOW, 0));
    rsx! { HistoryScreen {} }
}

fn history_page(management_state: ManagementState) -> Page {
    Page::new(|probe| {
        VirtualDom::new_with_props(
            Surface,
            SurfaceProps {
                management_state,
                snapshot: snapshot(PhaseDto::Idle),
                probe,
            },
        )
    })
}

/// A pasted dictation of 1:15 with kept audio, cleaned up from its transcript.
fn entry(id: u64, created_at: UtcMillis) -> HistoryEntry {
    HistoryEntry {
        created_at,
        duration_millis: Some(75_000),
        cleaned_text: Some(format!("cleaned {id}")),
        has_audio: true,
        ..crate::test_support::history_entry(id, &format!("raw {id}"))
    }
}

/// Settings keeping the last 20 entries, and the history window `entries`.
fn management(entries: Vec<HistoryEntry>) -> ManagementState {
    let mut management = ManagementState::default();
    management.install_settings(SettingsView {
        settings: crate::test_support::editable_settings(),
        warnings: Vec::new(),
    });
    show_history(&mut management, entries);
    management
}

/// Entry 7, an hour old.
fn one_entry() -> ManagementState {
    management(vec![entry(7, UtcMillis(NOW.0 - HOUR))])
}

/// The search text of every history read so far.
fn history_reads() -> Vec<String> {
    fake::requests::<HistoryPageRequest>("get_history")
        .into_iter()
        .map(|request| request.filter.query)
        .collect()
}

#[test]
fn rows_group_by_the_viewers_local_day() {
    let at = |hours_before_now: i64| UtcMillis(NOW.0 - hours_before_now * HOUR);
    // 09:30, 14:30 the day before, and 04:30 the day before that, all UTC.
    let entries = vec![entry(3, at(1)), entry(2, at(20)), entry(1, at(30))];
    let groups = |offset_minutes| {
        let clock = LocalClock::fixed(NOW, offset_minutes);
        group_by_day(clock, &entries)
            .into_iter()
            .map(|(day, members)| {
                (
                    clock.day_label(day),
                    members.iter().map(|entry| entry.id.0).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let named = |groups: &[(&str, &[u64])]| {
        groups
            .iter()
            .map(|(label, ids)| ((*label).to_owned(), ids.to_vec()))
            .collect::<Vec<_>>()
    };

    assert_eq!(groups(0), named(&[("Today", &[3]), ("Yesterday", &[2, 1])]));
    // Ten hours east, the evening before is already today.
    assert_eq!(
        groups(600),
        named(&[("Today", &[3, 2]), ("Yesterday", &[1])])
    );
    // Eight hours west, the earliest entry falls two days back.
    assert_eq!(
        groups(-480),
        named(&[
            ("Today", &[3]),
            ("Yesterday", &[2]),
            ("Friday, 25 September", &[1]),
        ])
    );

    let page = history_page(management(entries));
    let html = page.html();
    let today = html.find(">Today<").expect("today heading");
    let yesterday = html.find(">Yesterday<").expect("yesterday heading");
    let row = |id: &str| {
        html.find(&format!("id=\"history-entry-{id}\""))
            .expect("row")
    };
    assert!(today < row("3") && row("3") < yesterday && yesterday < row("2"));
}

#[test]
fn rows_show_local_time_the_pasted_text_the_duration_and_a_problem_status() {
    let mut not_pasted = entry(8, UtcMillis(NOW.0 - 5 * 60_000));
    not_pasted.status = EntryStatus::NotDelivered;
    not_pasted.error = Some("Pasting failed".to_owned());
    let mut raw = entry(9, UtcMillis(NOW.0 - 3 * HOUR));
    raw.cleaned_text = None;
    let page = history_page(management(vec![
        not_pasted,
        raw,
        entry(7, UtcMillis(NOW.0 - 4 * HOUR)),
    ]));
    let html = page.html();

    let row_8 = text(&html, "history-entry-8");
    assert!(row_8.contains("5 min ago") && row_8.contains("1:15") && row_8.contains("Not pasted"));
    assert!(row_8.contains("cleaned 8") && row_8.contains("Pasting failed"));
    let row_9 = text(&html, "history-entry-9");
    assert!(row_9.contains("07:30") && row_9.contains("raw 9"));
    let row_7 = text(&html, "history-entry-7");
    assert!(row_7.contains("cleaned 7") && !row_7.contains("raw 7"));
    assert!(!row_7.contains("Failed") && !row_7.contains("Not pasted"));
    assert!(!html.contains("UTC"), "no internal time zone shown");
}

#[test]
fn a_failed_entry_shows_its_error_and_its_audio_can_be_played() {
    let mut failed = entry(7, UtcMillis(NOW.0 - HOUR));
    failed.status = EntryStatus::Failed;
    failed.raw_text = None;
    failed.cleaned_text = None;
    failed.error = Some("The speech model could not turn this recording into text.".to_owned());
    let mut without_audio = entry(8, UtcMillis(NOW.0 - 2 * HOUR));
    without_audio.has_audio = false;
    let page = history_page(management(vec![failed, without_audio]));
    let html = page.html();

    let row = text(&html, "history-entry-7");
    assert!(row.contains("Failed") && row.contains("No transcript"));
    assert!(text(&html, "history-problem-7").contains("could not turn this recording"));
    assert!(tag(&html, "history-copy-7").is_some_and(|tag| tag.contains("disabled")));
    assert!(tag(&html, "history-play-7").is_some());
    assert!(tag(&html, "history-play-8").is_none(), "no audio was kept");
}

#[test]
fn search_is_debounced_and_paged_within_the_search() {
    let mut management = one_entry();
    management.history_next_cursor = Some(HistoryCursor {
        created_at: NOW,
        id: EntryId(7),
    });
    let mut page = history_page(management);
    page.input("history-search", "inv")
        .input("history-search", "invoice")
        .settle();
    assert!(
        fake::commands().is_empty(),
        "typing reads nothing until it pauses"
    );
    assert_eq!(page.context().management.read().history_search, "invoice");

    // The pause after the first keystroke was cut short; only the latest text is searched.
    page.wait();
    assert_eq!(history_reads(), ["invoice"]);
    assert_eq!(page.context().management.read().history_window_pages(), 1);

    page.click("history-load-more");
    assert_eq!(history_reads(), ["invoice", "invoice"]);
    assert_eq!(page.context().management.read().history_window_pages(), 2);
}

#[test]
fn the_failed_filter_and_an_empty_search_offer_a_way_back() {
    let mut page = history_page(one_entry());
    assert!(
        tag(&page.html(), "history-failed-filter")
            .is_some_and(|tag| tag.contains("aria-pressed=\"false\""))
    );
    page.click("history-failed-filter");
    assert!(page.context().management.read().history_filter.failed_only);
    assert_eq!(fake::commands(), ["get_history"]);

    let mut filtered = management(Vec::new());
    filtered.show_failed_history_only(true);
    let mut page = history_page(filtered);
    let html = page.html();
    assert!(
        tag(&html, "history-failed-filter")
            .is_some_and(|tag| tag.contains("aria-pressed=\"true\""))
    );
    assert!(!html.contains("to dictate"));
    page.click("history-clear-filter");
    assert_eq!(
        page.context().management.read().history_filter,
        HistoryFilter::default()
    );
    assert_eq!(fake::commands(), ["get_history"]);
}

#[test]
fn the_empty_list_shows_the_actual_shortcut() {
    let page = history_page(management(Vec::new()));
    let html = page.html();
    assert!(html.contains("Hold "));
    let keys: Vec<&str> = html
        .split("<kbd")
        .skip(1)
        .map(|rest| {
            &rest[rest.find('>').expect("kbd tag end") + 1..rest.find("</kbd>").expect("kbd end")]
        })
        .collect();
    assert_eq!(keys, ["Ctrl", "Space"]);
    assert!(tag(&html, "history-clear").is_none(), "nothing to clear");
}

#[test]
fn the_header_summarizes_retention_and_links_to_privacy() {
    let mut page = history_page(one_entry());
    assert!(page.html().contains(
        "Keeping the last 20 transcripts and audio only for failed and cancelled dictations"
    ));
    page.click("history-retention-settings")
        .click("history-open-folder");
    assert_eq!(
        page.context().navigator.route(),
        Route::Settings(SettingsRoute::PrivacyData)
    );
    assert_eq!(fake::commands(), ["open_recordings_folder"]);
}

#[test]
fn deleting_an_entry_or_the_history_requires_confirmation() {
    let mut page = history_page(one_entry());
    page.click("history-delete-7");
    assert!(fake::commands().is_empty(), "Delete only asks");
    page.click("history-delete-7-confirm");
    assert_eq!(fake::commands(), ["delete_history_entry"]);
    assert_eq!(
        fake::requests::<DeleteHistoryEntryRequest>("delete_history_entry"),
        [DeleteHistoryEntryRequest { id: EntryId(7) }]
    );
    fake::answer("delete_history_entry", Ok(()));
    page.settle();

    page.click("history-clear");
    assert_eq!(
        fake::commands(),
        ["delete_history_entry", "get_history"],
        "Clear history only asks"
    );
    page.click("history-clear-confirm");
    assert_eq!(fake::commands().last(), Some(&"clear_history"));
}

#[test]
fn history_changes_wait_for_the_change_in_flight() {
    let mut page = history_page(one_entry());
    page.click("history-delete-7")
        .click("history-delete-7-confirm");
    page.settle();
    let html = page.html();
    assert!(tag(&html, "history-delete-7").is_some_and(|tag| tag.contains("disabled")));
    assert!(tag(&html, "history-clear").is_some_and(|tag| tag.contains("disabled")));
    assert!(text(&html, "history-change-reason").contains("Wait for the current change"));
    assert_eq!(fake::commands(), ["delete_history_entry"]);
}

#[test]
fn copy_confirms_on_the_button_that_was_pressed() {
    let mut state = one_entry();
    state.history.push(entry(8, UtcMillis(NOW.0 - 2 * HOUR)));
    let mut page = history_page(state);
    page.click("history-copy-7");
    assert_eq!(fake::requests::<String>("write_clipboard"), ["cleaned 7"]);
    fake::answer("write_clipboard", Ok(()));
    page.settle();
    let html = page.html();
    assert_eq!(text(&html, "history-copy-7").trim(), "Copied");
    assert_eq!(text(&html, "history-copy-8").trim(), "Copy");
    assert!(
        page.context().management.read().notice.is_none(),
        "no page-wide notice"
    );

    page.wait();
    assert_eq!(
        text(&page.html(), "history-copy-7").trim(),
        "Copy",
        "the confirmation clears after a moment"
    );

    let mut failed = one_entry();
    failed.history_copied = Some(HistoryCopied {
        entry: EntryId(7),
        succeeded: false,
    });
    let page = history_page(failed);
    assert_eq!(text(&page.html(), "history-copy-7").trim(), "Copy failed");
}

#[test]
fn storage_unavailability_replaces_the_list_with_a_reload() {
    let mut management = ManagementState::default();
    management.fail_history_refresh(
        &HistoryFilter::default(),
        Some(&whispernow_api::ApiErrorDto {
            category: whispernow_api::ErrorCategoryDto::General,
            code: "history_unavailable".to_owned(),
            summary: "History storage is unavailable".to_owned(),
            recovery: None,
        }),
    );
    let mut page = history_page(management);
    let html = page.html();
    assert!(html.contains("History storage is unavailable") && html.contains("role=\"alert\""));
    for absent in ["history-search", "history-clear", "history-load-more"] {
        assert!(
            tag(&html, absent).is_none(),
            "{absent} shown while unavailable"
        );
    }
    page.click("history-reload");
    assert_eq!(fake::commands(), ["get_history"]);
}

#[test]
fn a_search_that_cannot_run_explains_itself_in_place_of_stale_rows() {
    let mut management = one_entry();
    management.history_search = "x".repeat(300);
    assert!(management.apply_history_search());
    let filter = management.history_filter.clone();
    management.fail_history_refresh(
        &filter,
        Some(&whispernow_api::ApiErrorDto {
            category: whispernow_api::ErrorCategoryDto::General,
            code: "text_too_large".to_owned(),
            summary: "This search is too long. Shorten it and try again.".to_owned(),
            recovery: None,
        }),
    );
    assert!(management.notice.is_none(), "no page-wide notice");

    let page = history_page(management.clone());
    let html = page.html();
    assert!(
        tag(&html, "history-entry-7").is_none(),
        "rows from the previous search stay hidden"
    );
    assert!(tag(&html, "history-search-error").is_some_and(|tag| tag.contains("role=\"alert\"")));
    let search = tag(&html, "history-search").expect("search input");
    assert!(search.contains("aria-invalid=\"true\""));
    assert!(search.contains("aria-describedby=\"history-search-error\""));
    assert!(
        search.contains(&format!("maxlength=\"{}\"", HistoryFilter::MAX_QUERY_BYTES)),
        "the field stops at the length a search accepts"
    );

    management.finish_history_refresh(
        1,
        &filter,
        HistoryPage {
            entries: one_entry().history,
            next_cursor: None,
        },
    );
    let page = history_page(management);
    let html = page.html();
    assert!(tag(&html, "history-search-error").is_none());
    assert!(tag(&html, "history-entry-7").is_some());
    assert!(
        !tag(&html, "history-search")
            .expect("search input")
            .contains("aria-invalid")
    );
}

#[test]
fn production_controls_send_typed_commands() {
    let mut state = one_entry();
    state.history_next_cursor = Some(HistoryCursor {
        created_at: NOW,
        id: EntryId(7),
    });
    let sent = |ids: &[&str]| {
        let mut page = history_page(state.clone());
        for id in ids {
            page.click(id);
        }
        fake::commands()
    };

    assert_eq!(
        sent(&["history-delete-7", "history-delete-7-confirm"]),
        ["delete_history_entry"]
    );
    assert_eq!(
        fake::requests::<DeleteHistoryEntryRequest>("delete_history_entry"),
        [DeleteHistoryEntryRequest { id: EntryId(7) }]
    );
    assert_eq!(
        sent(&["history-clear", "history-clear-confirm"]),
        ["clear_history"]
    );
    assert_eq!(sent(&["history-open-folder"]), ["open_recordings_folder"]);
    assert_eq!(sent(&["history-load-more"]), ["get_history"]);
    assert_eq!(sent(&["history-copy-7"]), ["write_clipboard"]);
    assert_eq!(fake::requests::<String>("write_clipboard"), ["cleaned 7"]);
}
