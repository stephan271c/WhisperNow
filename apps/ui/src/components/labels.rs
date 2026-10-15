use whispernow_domain::ipc::ModelState;
use whispernow_domain::{ClearDataCategory, DownloadPhase, RecordingMode, SetupStep};

pub(crate) const fn recording_mode(mode: RecordingMode) -> &'static str {
    match mode {
        RecordingMode::Hold => "Hold to record",
        RecordingMode::Toggle => "Press to start and stop",
    }
}

pub(crate) const fn model_state(state: ModelState) -> &'static str {
    match state {
        ModelState::Unloaded => "Not loaded",
        ModelState::Downloading => "Downloading",
        ModelState::Loading => "Loading",
        ModelState::Ready => "Ready",
        ModelState::Failed => "Could not load",
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
        ClearDataCategory::Models => "Models",
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

/// English names of the BCP-47 tags speech models list.
const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fil", "Filipino"),
    ("fo", "Faroese"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("ht", "Haitian Creole"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("jv", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mg", "Malagasy"),
    ("mi", "Māori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Burmese"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("ps", "Pashto"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("su", "Sundanese"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("th", "Thai"),
    ("tk", "Turkmen"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("tt", "Tatar"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
];

/// The English name of a language tag; the tag itself when it is not listed.
pub(crate) fn language_name(tag: &str) -> &str {
    LANGUAGE_NAMES
        .iter()
        .find_map(|&(listed, name)| (listed == tag).then_some(name))
        .unwrap_or(tag)
}
