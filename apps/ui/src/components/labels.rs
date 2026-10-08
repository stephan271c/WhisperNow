use whispernow_api::{ClearDataCategory, DownloadPhase, ModelStateDto, RecordingMode, SetupStep};

pub(crate) const fn recording_mode(mode: RecordingMode) -> &'static str {
    match mode {
        RecordingMode::Hold => "Hold to record",
        RecordingMode::Toggle => "Press to start and stop",
    }
}

pub(crate) const fn model_state(state: ModelStateDto) -> &'static str {
    match state {
        ModelStateDto::Unloaded => "Not loaded",
        ModelStateDto::Downloading => "Downloading",
        ModelStateDto::Loading => "Loading",
        ModelStateDto::Ready => "Ready",
        ModelStateDto::Failed => "Could not load",
    }
}

pub(crate) const fn download_phase(phase: DownloadPhase) -> &'static str {
    match phase {
        DownloadPhase::Downloading => "Downloading",
        DownloadPhase::ValidatingArchive => "Checking the download",
        DownloadPhase::Extracting => "Unpacking",
        DownloadPhase::Installing => "Installing",
        DownloadPhase::Complete => "Finished",
    }
}

pub(crate) const fn clear_category(category: ClearDataCategory) -> &'static str {
    match category {
        ClearDataCategory::Settings => "Settings",
        ClearDataCategory::History => "History",
        ClearDataCategory::Models => "Speech models",
        ClearDataCategory::Logs => "Logs",
    }
}

/// A byte count in KiB or MiB, to one decimal place.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    let (unit, suffix) = if bytes >= MIB {
        (MIB, "MiB")
    } else {
        (1024, "KiB")
    };
    let whole = bytes / unit;
    let tenth = (bytes % unit) * 10 / unit;
    format!("{whole}.{tenth} {suffix}")
}

pub(crate) const fn setup_step(step: SetupStep) -> &'static str {
    match step {
        SetupStep::Welcome => "Welcome",
        SetupStep::Permissions => "Permissions",
        SetupStep::Model => "Speech model",
        SetupStep::Shortcuts => "Shortcuts",
        SetupStep::TryIt => "Try it",
    }
}
