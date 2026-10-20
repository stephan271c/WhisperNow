//! [`ReleaseWatcher`] learns when a held recording shortcut is released, because GNOME's
//! `GlobalShortcuts` portal sometimes drops the release. The portal still decides when a
//! shortcut is pressed; without a matching helper press, activation timeouts approximate releases.
//!
//! [`HelperPastePlatform`] pastes on Wayland without a remote desktop session. The clipboard
//! goes through `XWayland`, which GNOME shares with Wayland applications, and the helper's
//! virtual keyboard (`--paste`) presses the paste shortcut. Nothing asks for consent and GNOME
//! shows no screen-sharing indicator, which it keeps for at least five seconds after every
//! portal paste.

use std::{
    ffi::OsStr,
    fs, io,
    io::{BufRead as _, BufReader, Read, Write as _},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use whispernow_core::text_output::{KeystrokeOutcome, PastePlatform};
use whispernow_platform::{
    ClipboardContents, NativeClipboard, NativePasteError, PasteKeys, wayland::active_paste_keys,
};

const HELPER: &str = "whispernow-input-helper";
const INSTALLED_DIRS: [&str; 2] = ["/usr/local/libexec/whispernow", "/usr/libexec/whispernow"];
const SETUP_SCRIPT: &str = "install_input_helper.sh";
/// The helper creates its keyboard and lets the compositor add it first.
const READY_TIMEOUT: Duration = Duration::from_secs(2);
/// The helper waits up to a second for held modifiers before it answers.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(3);

/// `None` reports stdout closing; returning false stops reading it.
fn spawn(
    path: &Path,
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    mut on_line: impl FnMut(Option<String>) -> bool + Send + 'static,
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
                if !on_line(Some(line)) {
                    return;
                }
            }
            on_line(None);
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReleaseEvent {
    Ready,
    Held(Duration),
    Released(Duration),
    Unavailable,
}

impl ReleaseEvent {
    fn parse(line: &str) -> Option<Self> {
        match line {
            "ready" => Some(Self::Ready),
            "unavailable" => Some(Self::Unavailable),
            _ => {
                let (time, held) = line
                    .strip_prefix("held ")
                    .map_or((line, false), |time| (time, true));
                let at = Duration::from_millis(time.parse().ok()?);
                Some(if held {
                    Self::Held(at)
                } else {
                    Self::Released(at)
                })
            }
        }
    }
}

pub(crate) struct ReleaseWatcher {
    child: Option<Child>,
    reported_missing: bool,
}

impl ReleaseWatcher {
    pub(crate) fn new() -> Self {
        Self {
            child: None,
            reported_missing: false,
        }
    }

    pub(crate) fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            stop(&mut child);
        }
    }

    pub(crate) fn watch(
        &mut self,
        chord: &str,
        on_event: impl Fn(ReleaseEvent) + Send + Sync + 'static,
    ) {
        self.stop();
        let Some(helper) = helper_path() else {
            if !std::mem::replace(&mut self.reported_missing, true) {
                tracing::info!(
                    "{HELPER} is not installed; shortcut releases rely on activation timeouts"
                );
            }
            on_event(ReleaseEvent::Unavailable);
            return;
        };
        let on_event = Arc::new(on_event);
        let reader_event = Arc::clone(&on_event);
        let spawned = spawn(&helper, [chord], move |line| {
            let event = match line.as_deref() {
                None => ReleaseEvent::Unavailable,
                Some(line) => {
                    let Some(event) = ReleaseEvent::parse(line) else {
                        tracing::warn!("the release helper returned an invalid response");
                        reader_event(ReleaseEvent::Unavailable);
                        return false;
                    };
                    event
                }
            };
            reader_event(event);
            line.is_some()
        });
        match spawned {
            Ok(child) => self.child = Some(child),
            Err(error) => {
                tracing::warn!(%error, helper = %helper.display(), "the key release watcher did not start");
                on_event(ReleaseEvent::Unavailable);
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
        let mut helper = spawn(&path, ["--paste"], move |line| {
            line.is_some_and(|line| sender.send(line).is_ok())
        })
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
            // "failed", a timeout or a closed pipe: the helper can fail after emitting part of
            // the shortcut.
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

#[cfg(test)]
mod tests {
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
    fn a_helper_reports_its_output_and_stdout_closing() {
        let (sender, messages) = mpsc::channel();
        let mut helper = spawn(
            Path::new("/bin/sh"),
            ["-c", "printf 'ready\\n123456\\n'"],
            move |line| sender.send(line).is_ok(),
        )
        .expect("helper");
        for expected in [Some("ready"), Some("123456"), None] {
            assert_eq!(
                messages
                    .recv_timeout(Duration::from_secs(2))
                    .expect("helper message")
                    .as_deref(),
                expected,
            );
        }
        assert!(helper.wait().expect("helper exit").success());
    }

    #[test]
    fn release_lines_carry_the_monotonic_time() {
        assert_eq!(
            ReleaseEvent::parse("123456"),
            Some(ReleaseEvent::Released(Duration::from_millis(123_456)))
        );
        assert_eq!(
            ReleaseEvent::parse("held 123456"),
            Some(ReleaseEvent::Held(Duration::from_millis(123_456)))
        );
        assert_eq!(ReleaseEvent::parse("ready"), Some(ReleaseEvent::Ready));
        assert_eq!(
            ReleaseEvent::parse("unavailable"),
            Some(ReleaseEvent::Unavailable)
        );
        assert_eq!(ReleaseEvent::parse("not-a-time"), None);
        assert_eq!(ReleaseEvent::parse("123 456"), None);
        assert_eq!(ReleaseEvent::parse("held not-a-time"), None);
    }
}
