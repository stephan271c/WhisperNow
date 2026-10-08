//! [`ReleaseWatcher`] learns when a held recording shortcut is released, because GNOME's
//! `GlobalShortcuts` portal sometimes drops the release. The portal still decides when a
//! shortcut is pressed; without the helper, releases fall back to the portal's key repeats.
//!
//! [`HelperPastePlatform`] pastes on Wayland without a remote desktop session. The clipboard
//! goes through `XWayland`, which GNOME shares with Wayland applications, and the helper's
//! virtual keyboard (`--paste`) presses the paste shortcut. Nothing asks for consent and GNOME
//! shows no screen-sharing indicator, which it keeps for at least five seconds after every
//! portal paste.

use std::{
    collections::HashMap,
    ffi::OsStr,
    fs, io,
    io::{BufRead as _, BufReader, Read, Write as _},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use whispernow_api::Shortcut;
use whispernow_core::text_output::{KeystrokeOutcome, PastePlatform};
use whispernow_platform::{
    ClipboardContents, NativeClipboard, NativePasteError, PasteKeys, wayland::active_paste_keys,
};

use crate::shortcuts::{ShortcutBinding, ShortcutId};

const HELPER: &str = "whispernow-input-helper";
const INSTALLED_DIRS: [&str; 2] = ["/usr/local/libexec/whispernow", "/usr/libexec/whispernow"];
const SETUP_SCRIPT: &str = "install_input_helper.sh";
/// The helper creates its keyboard and lets the compositor add it first.
const READY_TIMEOUT: Duration = Duration::from_secs(2);
/// The helper waits up to a second for held modifiers before it answers.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(3);

/// Starts the helper with `args`. Each line it writes goes to `on_line` until that returns
/// false, and each line of its errors to the log.
fn spawn(
    path: &Path,
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    mut on_line: impl FnMut(String) -> bool + Send + 'static,
) -> io::Result<Child> {
    let mut child = Command::new(path)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(stdout) = child.stdout.take() {
        thread::spawn(move || {
            for line in lines(stdout) {
                if !on_line(line) {
                    return;
                }
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        thread::spawn(move || {
            for line in lines(stderr) {
                tracing::warn!(message = %line, "input helper");
            }
        });
    }
    Ok(child)
}

fn lines(reader: impl Read) -> impl Iterator<Item = String> {
    BufReader::new(reader).lines().map_while(Result::ok)
}

fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Receives each reported release with its `CLOCK_MONOTONIC` time, the clock
/// of portal shortcut timestamps.
pub(crate) type OnRelease = Arc<dyn Fn(ShortcutId, Duration) + Send + Sync>;

pub(crate) struct ReleaseWatcher {
    chords: HashMap<ShortcutId, String>,
    child: Option<Child>,
    on_release: OnRelease,
    reported_missing: bool,
}

impl ReleaseWatcher {
    pub(crate) fn new(on_release: OnRelease) -> Self {
        Self {
            chords: HashMap::new(),
            child: None,
            on_release,
            reported_missing: false,
        }
    }

    pub(crate) fn watch(&mut self, binding: &ShortcutBinding) {
        self.chords.insert(
            binding.id,
            format!("{:x}={}", binding.id.0, chord(&binding.shortcut)),
        );
        self.restart();
    }

    pub(crate) fn unwatch(&mut self, id: ShortcutId) {
        if self.chords.remove(&id).is_some() {
            self.restart();
        }
    }

    pub(crate) fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            stop(&mut child);
        }
    }

    /// The helper takes its chords as arguments, so a change starts a new one.
    fn restart(&mut self) {
        self.stop();
        if self.chords.is_empty() {
            return;
        }
        let Some(helper) = helper_path() else {
            if !std::mem::replace(&mut self.reported_missing, true) {
                tracing::info!("{HELPER} is not installed; shortcut releases rely on key repeats");
            }
            return;
        };
        let on_release = Arc::clone(&self.on_release);
        let spawned = spawn(&helper, self.chords.values(), move |line| {
            if let Some((id, at)) = parse_release(&line) {
                on_release(id, at);
            }
            true
        });
        match spawned {
            Ok(child) => self.child = Some(child),
            Err(error) => {
                tracing::warn!(%error, helper = %helper.display(), "the key release watcher did not start");
            }
        }
    }
}

impl Drop for ReleaseWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) struct HelperPastePlatform {
    clipboard: NativeClipboard,
    helper: Child,
    commands: ChildStdin,
    answers: mpsc::Receiver<String>,
    /// Runs the accessibility lookup of the active window.
    runtime: tokio::runtime::Runtime,
}

impl HelperPastePlatform {
    /// Starts the helper's keyboard, or `None` when the helper is missing, its
    /// keyboard cannot be created, or `XWayland` has no clipboard.
    pub(crate) fn connect() -> Option<Self> {
        let path = helper_path()?;
        let clipboard = NativeClipboard::connect()
            .inspect_err(|error| tracing::warn!(?error, "the XWayland clipboard is unavailable"))
            .ok()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        let (sender, answers) = mpsc::channel();
        let mut helper = spawn(&path, ["--paste"], move |line| sender.send(line).is_ok())
            .inspect_err(|error| tracing::warn!(%error, "the paste helper did not start"))
            .ok()?;
        let Some(commands) = helper.stdin.take() else {
            stop(&mut helper);
            return None;
        };
        let platform = Self {
            clipboard,
            helper,
            commands,
            answers,
            runtime,
        };
        match platform.answers.recv_timeout(READY_TIMEOUT).as_deref() {
            Ok("ready") => Some(platform),
            _ => None,
        }
    }
}

impl Drop for HelperPastePlatform {
    fn drop(&mut self) {
        stop(&mut self.helper);
    }
}

impl PastePlatform for HelperPastePlatform {
    type Snapshot = ClipboardContents;
    type Error = NativePasteError;

    fn snapshot_clipboard(&mut self) -> Result<Option<ClipboardContents>, NativePasteError> {
        self.clipboard.snapshot()
    }

    fn write_transcript(&mut self, text: &str) -> Result<(), NativePasteError> {
        self.clipboard.write_text(text)
    }

    fn clipboard_holds(&mut self, text: &str) -> Result<bool, NativePasteError> {
        self.clipboard.holds_text(text)
    }

    fn restore_clipboard(&mut self, snapshot: ClipboardContents) -> Result<(), NativePasteError> {
        self.clipboard.restore(&snapshot)
    }

    fn send_paste_shortcut(&mut self) -> KeystrokeOutcome<NativePasteError> {
        let command = match self.runtime.block_on(active_paste_keys()) {
            PasteKeys::ControlV => "ctrl-v",
            PasteKeys::ControlShiftV => "ctrl-shift-v",
        };
        // An answer that arrived after an earlier timeout belongs to that paste.
        while self.answers.try_recv().is_ok() {}
        if writeln!(self.commands, "{command}")
            .and_then(|()| self.commands.flush())
            .is_err()
        {
            return KeystrokeOutcome::NotSent(NativePasteError::Unavailable);
        }
        match self.answers.recv_timeout(ANSWER_TIMEOUT).as_deref() {
            Ok("sent") => KeystrokeOutcome::Sent,
            Ok("held") => KeystrokeOutcome::NotSent(NativePasteError::ModifierStillHeld),
            Ok("refused") => KeystrokeOutcome::NotSent(NativePasteError::InjectionFailed),
            _ => KeystrokeOutcome::Uncertain(NativePasteError::InjectionFailed),
        }
    }
}

fn helper_path() -> Option<PathBuf> {
    let beside_app = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(HELPER)));
    INSTALLED_DIRS
        .iter()
        .map(|dir| PathBuf::from(dir).join(HELPER))
        .chain(beside_app)
        .find(|path| path.is_file())
}

/// The command that installs the helper setgid `input` with its uinput rule,
/// when this build carries the install script. Root cannot read a running
/// `AppImage`'s files, so its helper and script are first copied to `setup_dir`;
/// a source build uses the repository's script.
pub(crate) fn setup_command(setup_dir: &Path) -> Option<String> {
    let script = match std::env::var_os("APPDIR") {
        Some(app_dir) => copy_setup(Path::new(&app_dir), setup_dir)
            .inspect_err(|error| tracing::warn!(%error, "the helper setup could not be copied"))
            .ok()?,
        // `target/release/whispernow-desktop` beside `scripts`.
        None => std::env::current_exe()
            .ok()?
            .ancestors()
            .nth(3)?
            .join("scripts")
            .join(SETUP_SCRIPT),
    };
    script
        .is_file()
        .then(|| format!("sudo sh {}", shell_quoted(&script.to_string_lossy())))
}

fn copy_setup(app_dir: &Path, setup_dir: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(setup_dir)?;
    fs::copy(
        app_dir.join("usr/libexec/whispernow").join(HELPER),
        setup_dir.join(HELPER),
    )?;
    let script = setup_dir.join(SETUP_SCRIPT);
    fs::copy(
        app_dir.join("usr/share/whispernow").join(SETUP_SCRIPT),
        &script,
    )?;
    Ok(script)
}

/// `text` as one shell word.
fn shell_quoted(text: &str) -> String {
    if text
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "/._-+".contains(c))
    {
        text.to_owned()
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    }
}

fn parse_release(line: &str) -> Option<(ShortcutId, Duration)> {
    let (id, millis) = line.split_once(' ')?;
    Some((
        ShortcutId(u32::from_str_radix(id, 16).ok()?),
        Duration::from_millis(millis.parse().ok()?),
    ))
}

/// The helper's chord for `shortcut`: each key's Linux input codes, modifiers on either side.
/// Letters and digits are their QWERTY positions; on another layout a letter chord may not
/// match, and its release then falls back to key repeats.
fn chord(shortcut: &Shortcut) -> String {
    shortcut
        .modifiers
        .iter()
        .map(|modifier| modifier.evdev())
        .chain([shortcut.key.evdev()])
        .map(|codes| {
            codes
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("+")
}

#[cfg(test)]
mod tests {
    use whispernow_api::{ShortcutKey, ShortcutModifier};

    use super::*;

    #[test]
    fn a_setup_path_is_one_shell_word() {
        assert_eq!(
            shell_quoted("/home/a/.local/share/x/install_input_helper.sh"),
            "/home/a/.local/share/x/install_input_helper.sh"
        );
        assert_eq!(shell_quoted("/home/a b/it's.sh"), r"'/home/a b/it'\''s.sh'");
    }

    #[test]
    fn an_appimage_copies_its_helper_and_script_out() {
        let app_dir = tempfile::tempdir().expect("app dir");
        let setup_dir = tempfile::tempdir().expect("setup dir");
        for (dir, name) in [
            ("usr/libexec/whispernow", HELPER),
            ("usr/share/whispernow", SETUP_SCRIPT),
        ] {
            fs::create_dir_all(app_dir.path().join(dir)).expect("dir");
            fs::write(app_dir.path().join(dir).join(name), name).expect("file");
        }
        let target = setup_dir.path().join("helper-setup");
        let script = copy_setup(app_dir.path(), &target).expect("copied");
        assert_eq!(script, target.join(SETUP_SCRIPT));
        assert_eq!(
            fs::read_to_string(target.join(HELPER)).expect("helper"),
            HELPER
        );
        assert_eq!(fs::read_to_string(script).expect("script"), SETUP_SCRIPT);
    }

    #[test]
    fn shortcuts_become_input_code_chords() {
        let chord_of = |modifiers: &[ShortcutModifier], key| {
            chord(&Shortcut {
                modifiers: modifiers.iter().copied().collect(),
                key,
            })
        };
        assert_eq!(
            chord_of(&[ShortcutModifier::Control], ShortcutKey::Space),
            "29,97+57"
        );
        assert_eq!(
            chord_of(
                &[
                    ShortcutModifier::Meta,
                    ShortcutModifier::Shift,
                    ShortcutModifier::Alt,
                    ShortcutModifier::Control
                ],
                ShortcutKey::Enter
            ),
            "29,97+56,100+42,54+125,126+28,96"
        );
        assert_eq!(
            chord_of(&[ShortcutModifier::Alt], ShortcutKey::A),
            "56,100+30"
        );
        assert_eq!(
            chord_of(&[ShortcutModifier::Control], ShortcutKey::M),
            "29,97+50"
        );
        assert_eq!(
            chord_of(&[ShortcutModifier::Control], ShortcutKey::Digit0),
            "29,97+11"
        );
        assert_eq!(
            chord_of(&[ShortcutModifier::Control], ShortcutKey::F12),
            "29,97+88"
        );
        assert_eq!(
            chord_of(&[ShortcutModifier::Control], ShortcutKey::F24),
            "29,97+194"
        );
    }

    #[test]
    fn release_lines_carry_the_shortcut_and_its_time() {
        assert_eq!(
            parse_release("8003e 123456"),
            Some((ShortcutId(0x8003e), Duration::from_millis(123_456)))
        );
        assert_eq!(parse_release("8003e"), None);
        assert_eq!(parse_release("zz 1"), None);
    }
}
