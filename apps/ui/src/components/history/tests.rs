use dioxus::prelude::*;
use whispernow_domain::ipc::{EventPayload, Phase};
use whispernow_domain::{
    AppError, EntryId, EntryStatus, ErrorCategory, HistoryEntry, HistoryFilter, HistoryPage,
    HistoryPageRequest, Shortcut, UtcMillis,
};

use super::format::{group_by_day, retention_summary};
use super::{HistoryScreen, LocalClock};
use crate::client::fake;
use crate::management::{HistoryRows, ManagementState};
use crate::operations::UiOperation;
use crate::route::{Route, SettingsRoute};
use crate::state::UiState;
use crate::test_support::{
    Page, editable_settings, management_with, show_history, snapshot, tag, text,
};

/// 2026-09-27 10:30 UTC.
const NOW: UtcMillis = UtcMillis(1_790_505_000_000);
const HOUR: i64 = 3_600_000;

fn fixed_clock_history() -> Element {
    use_context_provider(|| LocalClock::fixed(NOW, 0));
    rsx! { HistoryScreen {} }
}

fn history_page(management_state: ManagementState) -> Page {
    let mut state = UiState::default();
    state.hydrate(snapshot(Phase::Idle));
    Page::host(state, management_state, fixed_clock_history)
}

/// A pasted dictation of 1:15 with kept audio, cleaned up from its transcript.
fn entry(id: i64, created_at: UtcMillis) -> HistoryEntry {
    HistoryEntry {
        created_at,
        duration_millis: Some(75_000),
        cleaned_text: Some(format!("cleaned {id}")),
        audio_kept: true,
        ..crate::test_support::history_entry(id, &format!("raw {id}"))
    }
}

/// Settings keeping the last 20 entries, and the history window `entries`.
fn management(entries: Vec<HistoryEntry>) -> ManagementState {
    let mut management = management_with(editable_settings());
    show_history(&mut management, entries);
    management
}

fn one_entry() -> ManagementState {
    management(vec![entry(7, UtcMillis(NOW.0 - HOUR))])
}

fn more_to_load(management: &mut ManagementState) {
    let HistoryRows::Loaded { has_more, .. } = &mut management.history else {
        panic!("history is loaded");
    };
    *has_more = true;
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
    let named = |groups: &[(&str, &[i64])]| {
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
    assert!(row_7.contains("cleaned 7"));
    assert!(!row_7.contains("Failed") && !row_7.contains("Not pasted"));
}

#[test]
fn a_cleaned_row_opens_to_its_original_transcript() {
    let mut raw = entry(9, UtcMillis(NOW.0 - 3 * HOUR));
    raw.cleaned_text = None;
    let page = history_page(management(vec![entry(7, UtcMillis(NOW.0 - HOUR)), raw]));
    let html = page.html();

    assert_eq!(text(&html, "history-cleaned-7"), "Cleaned");
    assert!(tag(&html, "history-text-7").is_some_and(|tag| !tag.contains(" open")));
    assert_eq!(
        text(&html, "history-original-7"),
        "Original transcriptraw 7"
    );
    assert!(text(&html, "history-entry-9").contains("raw 9"));
    assert!(tag(&html, "history-text-9").is_none());
    assert!(tag(&html, "history-cleaned-9").is_none());
}

#[test]
fn a_failed_entry_shows_its_error_and_its_audio_can_be_played() {
    let mut failed = entry(7, UtcMillis(NOW.0 - HOUR));
    failed.status = EntryStatus::Failed;
    failed.raw_text = None;
    failed.cleaned_text = None;
    failed.error = Some("The speech model could not turn this recording into text.".to_owned());
    let mut without_audio = entry(8, UtcMillis(NOW.0 - 2 * HOUR));
    without_audio.audio_kept = false;
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
    more_to_load(&mut management);
    let mut page = history_page(management);
    page.input("history-search", "inv")
        .input("history-search", "invoice")
        .settle();
    assert_eq!(fake::count("get_history"), 0, "typing waits for a pause");
    page.wait();
    let first = HistoryPageRequest {
        limit: HistoryPageRequest::DEFAULT_PAGE_SIZE,
        filter: HistoryFilter {
            query: "invoice".to_owned(),
            failed_only: false,
        },
    };
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history").as_slice(),
        std::slice::from_ref(&first)
    );

    page.click("history-load-more");
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history"),
        [
            first.clone(),
            HistoryPageRequest {
                limit: 2 * HistoryPageRequest::DEFAULT_PAGE_SIZE,
                ..first
            },
        ]
    );
}

#[test]
fn history_rows_render_again_only_on_the_minute_tick() {
    let entries = vec![
        entry(7, UtcMillis(NOW.0 - HOUR)),
        entry(8, UtcMillis(NOW.0 - 2 * HOUR)),
    ];
    let mut page = super::super::tests::shell(snapshot(Phase::Idle), management(entries));
    page.click("nav-history");
    let rows = |rendered: &[&str]| {
        rendered
            .iter()
            .filter(|name| **name == "HistoryRow")
            .count()
    };

    let before = page.render_counts();
    page.publish(crate::test_support::state_event(
        crate::test_support::RECORDING,
        |_| {},
    ));
    assert_eq!(rows(&page.rendered_since(&before)), 0, "a phase change");

    let before = page.render_counts();
    page.act(|mut context| {
        context
            .management
            .write()
            .begin_busy(UiOperation::CheckForUpdates);
    });
    let rendered = page.rendered_since(&before);
    assert!(rendered.contains(&"HistoryScreen"), "{rendered:?}");
    assert_eq!(rows(&rendered), 0, "a screen render with the same entries");

    // The clock reads whole milliseconds; the tick must read a later one.
    std::thread::sleep(std::time::Duration::from_millis(2));
    let before = page.render_counts();
    page.wait();
    assert_eq!(rows(&page.rendered_since(&before)), 2, "the minute tick");
}

#[test]
fn typing_in_the_search_renders_the_search_not_the_rest_of_the_window() {
    let mut page = super::super::tests::shell(snapshot(Phase::Idle), one_entry());
    page.click("nav-history");
    let before = page.render_counts();

    page.input("history-search", "inv")
        .input("history-search", "invoice");

    assert_eq!(page.rendered_since(&before), ["SearchBox", "TextInput"]);
    assert_eq!(
        page.context().management.read().history_filter.query,
        "",
        "the filter changes once typing pauses"
    );
    page.wait();
    assert_eq!(
        page.context().management.read().history_filter.query,
        "invoice"
    );
}

#[test]
fn a_delayed_refresh_cannot_shrink_the_window_requested_by_load_more() {
    let mut management = one_entry();
    more_to_load(&mut management);
    let mut page = history_page(management);
    page.publish(EventPayload::HistoryChanged)
        .click("history-load-more");
    let requests = fake::requests::<HistoryPageRequest>("get_history");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].limit, HistoryPageRequest::DEFAULT_PAGE_SIZE);
    assert_eq!(requests[1].limit, 2 * HistoryPageRequest::DEFAULT_PAGE_SIZE);
    let expanded = HistoryPage {
        entries: vec![entry(7, NOW), entry(6, UtcMillis(NOW.0 - HOUR))],
        has_more: true,
    };
    fake::answer_nth("get_history", 1, Ok(expanded.clone()));
    page.settle();
    fake::answer_nth(
        "get_history",
        0,
        Ok(HistoryPage {
            entries: vec![entry(7, NOW)],
            has_more: false,
        }),
    );
    page.settle();
    let context = page.context();
    let management = context.management.read();
    assert_eq!(management.history.entries(), expanded.entries);
    assert_eq!(management.history.has_more(), expanded.has_more);
    assert_eq!(management.history_window_pages(), 2);
    drop(management);
    assert!(tag(&page.html(), "history-entry-6").is_some());
    page.publish(EventPayload::HistoryChanged);
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history")[2].limit,
        requests[1].limit
    );
}

#[test]
fn a_read_overlapping_deletion_cannot_restore_the_row_or_report_an_obsolete_failure() {
    for obsolete in [
        Ok(HistoryPage {
            entries: one_entry().history.entries().to_vec(),
            has_more: true,
        }),
        Err(AppError::new(
            ErrorCategory::HistoryUnavailable,
            "history_unavailable",
            "History storage is unavailable",
        )),
    ] {
        let mut page = history_page(one_entry());
        page.publish(EventPayload::HistoryChanged)
            .click("history-delete-7")
            .click("history-delete-7-confirm")
            .answer("delete_history_entry", Ok(()));
        assert_eq!(fake::count("get_history"), 2);
        fake::answer_nth(
            "get_history",
            1,
            Ok(HistoryPage {
                entries: Vec::new(),
                has_more: false,
            }),
        );
        page.settle();
        assert!(tag(&page.html(), "history-entry-7").is_none());
        fake::answer_nth("get_history", 0, obsolete);
        page.settle();
        let context = page.context();
        let management = context.management.read();
        assert_eq!(
            management.history,
            HistoryRows::Loaded {
                filter: HistoryFilter::default(),
                entries: Vec::new(),
                has_more: false,
            }
        );
        assert_eq!(management.history_window_pages(), 1);
        assert!(management.notice.is_none());
        assert!(tag(&page.html(), "history-entry-7").is_none());
    }
}

#[test]
fn the_failed_filter_and_an_empty_search_offer_a_way_back() {
    let mut page = history_page(one_entry());
    assert!(
        tag(&page.html(), "history-failed-filter")
            .is_some_and(|tag| tag.contains("aria-pressed=\"false\""))
    );
    page.click("history-failed-filter");
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history"),
        [HistoryPageRequest {
            limit: HistoryPageRequest::DEFAULT_PAGE_SIZE,
            filter: HistoryFilter {
                query: String::new(),
                failed_only: true
            },
        },]
    );

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
        fake::requests::<HistoryPageRequest>("get_history"),
        [HistoryPageRequest {
            limit: HistoryPageRequest::DEFAULT_PAGE_SIZE,
            filter: HistoryFilter::default(),
        }]
    );
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
    assert_eq!(
        keys,
        crate::components::ui::shortcut_parts(&Shortcut::default())
    );
    assert!(tag(&html, "history-clear").is_none(), "nothing to clear");
}

#[test]
fn the_header_summarizes_retention_and_links_to_privacy() {
    let mut page = history_page(one_entry());
    let settings = crate::test_support::editable_settings();
    assert!(page.html().contains(&retention_summary(
        settings.history_limit,
        settings.history_retention,
        settings.keep_audio
    )));
    page.click("history-retention-settings")
        .click("history-open-folder");
    assert_eq!(
        page.context().navigator.route(),
        Route::Settings(SettingsRoute::PrivacyData)
    );
    assert_eq!(fake::count("open_recordings_folder"), 1);
}

#[test]
fn deleting_an_entry_or_the_history_requires_confirmation() {
    let mut page = history_page(one_entry());
    page.click("history-delete-7");
    assert_eq!(fake::count("delete_history_entry"), 0);
    page.click("history-delete-7-confirm");
    assert_eq!(fake::count("delete_history_entry"), 1);
    assert_eq!(
        fake::requests::<EntryId>("delete_history_entry"),
        [EntryId(7)]
    );
    let html = page.html();
    assert!(tag(&html, "history-delete-7").is_some_and(|tag| tag.contains("disabled")));
    assert!(
        tag(&html, "history-clear").is_some_and(|tag| tag.contains("disabled")
            && tag.contains(r#"aria-describedby="history-change-reason""#))
    );
    assert!(text(&html, "history-change-reason").contains("Wait for the current change"));
    fake::answer("delete_history_entry", Ok(()));
    page.settle();

    page.click("history-clear");
    assert_eq!(fake::count("clear_history"), 0);
    page.click("history-clear-confirm");
    assert_eq!(fake::count("clear_history"), 1);
}

#[test]
fn copy_confirms_on_the_button_that_was_pressed() {
    let state = management(vec![
        entry(7, UtcMillis(NOW.0 - HOUR)),
        entry(8, UtcMillis(NOW.0 - 2 * HOUR)),
    ]);
    let mut page = history_page(state);
    page.click("history-copy-7");
    assert_eq!(fake::requests::<String>("copy_text"), ["cleaned 7"]);
    fake::answer("copy_text", Ok(()));
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

    page.click("history-copy-8");
    assert_eq!(
        fake::requests::<String>("copy_text"),
        ["cleaned 7", "cleaned 8"]
    );
    page.answer(
        "copy_text",
        Err::<(), _>(AppError::new(
            ErrorCategory::General,
            "clipboard_unavailable",
            "The clipboard is unavailable",
        )),
    );
    let html = page.html();
    assert_eq!(text(&html, "history-copy-8").trim(), "Copy failed");
    assert_eq!(text(&html, "history-copy-7").trim(), "Copy");
    assert!(
        page.context().management.read().notice.is_none(),
        "copy failures stay on the pressed row"
    );
    page.wait();
    assert_eq!(text(&page.html(), "history-copy-8").trim(), "Copy");
    assert_eq!(text(&page.html(), "history-copy-7").trim(), "Copy");
}

#[test]
fn storage_unavailability_replaces_the_list_with_a_reload() {
    let mut management = management(vec![
        entry(7, UtcMillis(NOW.0 - HOUR)),
        entry(8, UtcMillis(NOW.0 - 2 * HOUR)),
    ]);
    more_to_load(&mut management);
    let mut page = history_page(management);
    let loaded = page.html();
    for present in [
        "history-entry-7",
        "history-entry-8",
        "history-search",
        "history-clear",
        "history-load-more",
    ] {
        assert!(tag(&loaded, present).is_some(), "{present} was loaded");
    }
    page.publish(EventPayload::HistoryChanged);
    let request = HistoryPageRequest::first_page(HistoryFilter::default());
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history"),
        std::slice::from_ref(&request)
    );
    page.answer(
        "get_history",
        Err::<HistoryPage, _>(AppError::new(
            ErrorCategory::HistoryUnavailable,
            "history_unavailable",
            "History storage is unavailable",
        )),
    );
    let html = page.html();
    assert!(html.contains("History storage is unavailable") && html.contains("role=\"alert\""));
    assert_eq!(text(&html, "history-reload").trim(), "Try again");
    for absent in [
        "history-entry-7",
        "history-entry-8",
        "history-search",
        "history-clear",
        "history-load-more",
    ] {
        assert!(
            tag(&html, absent).is_none(),
            "{absent} shown while unavailable"
        );
    }
    let context = page.context();
    let management = context.management.read();
    assert_eq!(
        management.history,
        HistoryRows::Unavailable("History storage is unavailable".to_owned())
    );
    drop(management);
    page.click("history-reload");
    assert_eq!(
        fake::requests::<HistoryPageRequest>("get_history"),
        [request.clone(), request]
    );
}

#[test]
fn a_search_that_cannot_run_explains_itself_in_place_of_stale_rows() {
    let mut management = one_entry();
    assert!(management.apply_history_search(&"x".repeat(300)));
    management.fail_history_refresh(&whispernow_domain::AppError {
        category: whispernow_domain::ErrorCategory::General,
        code: "text_too_large".to_owned(),
        summary: "This search is too long. Shorten it and try again.".to_owned(),
    });
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

    management.finish_history_refresh(HistoryPage {
        entries: one_entry().history.entries().to_vec(),
        has_more: false,
    });
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
