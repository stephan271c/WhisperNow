mod activation;
mod autostart;
mod gateway;
#[cfg(target_os = "linux")]
mod input_helper;
mod overlay;
#[cfg(target_os = "linux")]
mod portal_identity;
mod services;
mod shortcuts;
mod tray;
mod updater;
#[cfg(target_os = "linux")]
mod wayland_shortcuts;

use std::{io, sync::Arc, time::Duration};

use gateway::{
    Gateway,
    commands::{
        cancel_download, cancel_recording, clear_api_key, clear_data, clear_history,
        complete_setup, copy_last_transcript, data_locations, delete_history_entry, delete_model,
        dismiss_recording_overlay, download_model, get_history, get_history_audio, get_permissions,
        get_settings, get_setup_state, get_state, list_audio_devices, list_models,
        list_provider_models, open_main_window, open_permission_settings, open_recordings_folder,
        request_permission, restart_setup, resume_shortcuts, save_setup_checkpoint, select_model,
        set_api_key, set_start_on_login, start_test_dictation, stop_recording, suspend_shortcuts,
        test_microphone, update_settings,
    },
    forward_events,
};
use services::NativeServices;
use tauri::Manager as _;
use tauri_plugin_global_shortcut::ShortcutState;
use whispernow_domain::WindowGeometry;

use crate::shortcuts::{
    KeyTransition, ShortcutCoordinator, ShortcutEvent, ShortcutId, ShortcutRegistrar,
    TauriShortcutRegistrar, cancel_binding, recording_binding,
};
use crate::{
    activation::ActivationGate,
    autostart::{AutostartRegistration, TauriAutostart},
    overlay::{RecordingOverlay, TauriOverlayWindow},
    tray::{TrayAction, TrayController},
};

pub(crate) const SETTINGS_WINDOW: &str = "settings";
pub(crate) const RECORDING_BAR_WINDOW: &str = "recording-bar";

const SHUTDOWN_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(20);

/// Declares the Tauri invoke handler and, from the same tokens, the command
/// paths it registers, so the command catalog test checks the real handler.
macro_rules! registered_commands {
    ($($($segment:ident)::+),+ $(,)?) => {
        #[cfg(test)]
        const REGISTERED_COMMANDS: &[&str] = &[$(stringify!($($segment)::+)),+];

        fn invoke_handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
            tauri::generate_handler![$($($segment)::+),+]
        }
    };
}

registered_commands![
    get_settings,
    update_settings,
    set_start_on_login,
    set_api_key,
    clear_api_key,
    list_provider_models,
    list_audio_devices,
    test_microphone,
    list_models,
    download_model,
    cancel_download,
    select_model,
    delete_model,
    get_setup_state,
    save_setup_checkpoint,
    complete_setup,
    restart_setup,
    get_permissions,
    request_permission,
    open_permission_settings,
    start_test_dictation,
    stop_recording,
    cancel_recording,
    dismiss_recording_overlay,
    copy_last_transcript,
    open_main_window,
    get_state,
    get_history,
    get_history_audio,
    open_recordings_folder,
    delete_history_entry,
    clear_history,
    data_locations,
    clear_data,
    suspend_shortcuts,
    resume_shortcuts,
    updater::get_update_state,
    updater::check_for_updates,
    updater::dismiss_update,
    updater::restart_to_update,
    updater::open_update_release_page,
];

#[cfg_attr(mobile, tauri::mobile_entry_point)]
#[allow(
    clippy::too_many_lines,
    reason = "desktop plugin, state, window, command, and shutdown wiring stays auditable in one root"
)]
pub fn run() {
    whispernow_platform::init_logging();
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "starting WhisperNow");
    let (shortcut_events, mut shortcut_receiver) = tokio::sync::mpsc::unbounded_channel();
    let plugin_shortcut_events = shortcut_events.clone();
    let setup_shortcut_events = shortcut_events;
    let native_wayland = native_wayland_session();
    let floating_windows = !native_wayland || use_xwayland_windows();
    let activation = Arc::new(ActivationGate::default());
    let callback_activation = Arc::clone(&activation);
    let public_key = option_env!("WHISPERNOW_UPDATER_PUBLIC_KEY")
        .unwrap_or_default()
        .trim()
        .to_owned();
    let mut updater_plugin =
        tauri_plugin_updater::Builder::new().default_version_comparator(|current, release| {
            updater::is_newer_stable(&current, &release.version)
        });
    if !public_key.is_empty() {
        updater_plugin = updater_plugin.pubkey(public_key.clone());
    }
    let builder = tauri::Builder::default()
        // The single-instance plugin must be registered first so no other plugin performs
        // duplicate side effects in a secondary process.
        .plugin(tauri_plugin_single_instance::init(
            move |app, _args, _cwd| {
                let activate = callback_activation.request_activation(|| {
                    app.try_state::<Gateway>()
                        .map(|state| state.services.state.snapshot().phase)
                });
                if activate {
                    show_settings_window(app);
                }
            },
        ))
        .plugin(tauri_plugin_autostart::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(updater_plugin.build());
    let builder = if native_wayland {
        builder
    } else {
        builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |_app, shortcut, event| {
                    let id = ShortcutId(shortcut.id());
                    let transition = match event.state() {
                        ShortcutState::Pressed => KeyTransition::Pressed,
                        ShortcutState::Released => KeyTransition::Released,
                    };
                    let _queued =
                        plugin_shortcut_events.send(ShortcutEvent::Transition { id, transition });
                })
                .build(),
        )
    };
    let application = builder
        .setup(move |app| {
            let tray = Arc::new(TrayController::default());
            tray.build(app, handle_tray_action)?;
            let overlay = RecordingOverlay::new(
                Arc::new(TauriOverlayWindow::new(app.handle().clone())),
                floating_windows,
            );
            let (events, event_receiver) = services::Events::channel();
            let resource_dir = app.path().resource_dir()?;
            let (services, enable_autostart) =
                NativeServices::compose(events, native_wayland, resource_dir).map_err(|error| {
                    io::Error::other(format!("desktop setup failed ({})", error.code))
                })?;
            if !services.output.can_copy() {
                tray.mark_copy_unavailable();
            }
            let settings = services.settings.load();
            let start_minimized = settings.start_minimized;
            let autostart: Arc<dyn AutostartRegistration> =
                Arc::new(TauriAutostart::new(app.handle().clone()));
            if enable_autostart && let Err(error) = autostart.set_enabled(true) {
                services
                    .state
                    .publish_error(gateway::autostart_error(error));
            }
            if let Some(geometry) = settings.window_geometry {
                restore_settings_geometry(app, geometry);
            }
            // Linux draws the main window's titlebar in the page, which asks the window
            // whether it is decorated. Setup finishes before the event loop answers any
            // page request, so the page only ever sees this frame.
            #[cfg(target_os = "linux")]
            if let Some(window) = app.get_webview_window(SETTINGS_WINDOW)
                && window.set_decorations(false).is_err()
            {
                tracing::warn!("settings window decorations could not be removed");
            }
            let persisted_binding = recording_binding(&settings);
            let (registrar, startup_failure) = shortcut_registrar(
                app.handle().clone(),
                native_wayland,
                setup_shortcut_events.clone(),
            )?;
            apply_initial_window_policy(app, start_minimized);
            let services = Arc::new(services);
            let probe_app = app.handle().clone();
            let shortcuts = Arc::new(ShortcutCoordinator::new(
                registrar,
                persisted_binding,
                Arc::new(move || {
                    probe_app
                        .try_state::<Gateway>()
                        .is_some_and(|state| state.shortcuts_enabled())
                }),
                startup_failure,
            ));
            if let Some(error) = shortcuts.failure() {
                tracing::warn!("configured push-to-talk shortcut is unavailable");
                services.state.publish_error(gateway::shortcut_error(error));
            }
            // Registered only while a dictation runs, so nothing is registered here.
            shortcuts.set_cancel_binding(cancel_binding(&settings));
            app.manage(Gateway::new(
                services,
                shortcuts,
                autostart,
                Arc::clone(&activation),
                Arc::clone(&tray),
            ));
            NativeServices::spawn_retention_checks(Arc::clone(&app.state::<Gateway>().services));
            let updates_enabled = !cfg!(debug_assertions) && !public_key.is_empty();
            let update_controller =
                updater::UpdateController::new(app.handle().clone(), updates_enabled);
            app.manage(Arc::clone(&update_controller));
            if updates_enabled {
                updater::spawn_periodic_checks(update_controller);
            }
            // A second launch during setup asked for the window before there was a state.
            if activation.release_deferred_activation(
                &app.state::<Gateway>().services.state.snapshot().phase,
            ) {
                show_settings_window(app.handle());
            }
            forward_events(app.handle().clone(), overlay, event_receiver)?;
            app.state::<Gateway>().services.state.publish();
            let shortcut_app = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = shortcut_receiver.recv().await {
                    match event {
                        ShortcutEvent::Transition { id, transition } => {
                            let state = shortcut_app.state::<Gateway>();
                            if let Some(action) = state.shortcuts.input(id, transition) {
                                state.shortcut_action(action);
                            }
                        }
                        ShortcutEvent::CapabilityLost(error) => {
                            tracing::warn!(?error, "Wayland shortcut portal capability was lost");
                            // Invalidation waits for any in-flight native registration.
                            let lost_app = shortcut_app.clone();
                            let _ = tauri::async_runtime::spawn_blocking(move || {
                                lost_app.state::<Gateway>().shortcut_capability_lost(error);
                            })
                            .await;
                        }
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            let label = window.label();
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && [SETTINGS_WINDOW, RECORDING_BAR_WINDOW].contains(&label)
            {
                api.prevent_close();
                if label == SETTINGS_WINDOW
                    && let (Ok(position), Ok(size)) = (window.outer_position(), window.inner_size())
                {
                    let geometry = WindowGeometry {
                        x: position.x,
                        y: position.y,
                        width: size.width,
                        height: size.height,
                    };
                    // The save takes the settings lock and writes a file, so it
                    // runs off the event loop; the window hides immediately.
                    let app = window.app_handle().clone();
                    tauri::async_runtime::spawn_blocking(move || {
                        if let Some(state) = app.try_state::<Gateway>()
                            && let Err(error) =
                                state.services.settings.save_window_geometry(geometry)
                        {
                            tracing::warn!(code = error.code, "window geometry could not be saved");
                        }
                    });
                }
                if window.hide().is_err() {
                    tracing::warn!(label, "window could not be hidden");
                }
                // A hidden window cannot finish a shortcut capture, so global
                // shortcuts must not stay suspended on its behalf.
                let app = window.app_handle().clone();
                tauri::async_runtime::spawn(async move {
                    let Some(state) = app.try_state::<Gateway>() else {
                        return;
                    };
                    if let Err(error) = state.set_shortcut_capture(false).await {
                        tracing::warn!(code = error.code, "shortcuts could not be resumed");
                    }
                });
            }
        })
        .invoke_handler(invoke_handler())
        .build(tauri::generate_context!());
    let application = application.unwrap_or_else(|error| panic!("Tauri setup failed: {error}"));
    application.run(|app, event| {
        if let tauri::RunEvent::ExitRequested { api, code, .. } = event {
            let state = app.state::<Gateway>();
            if state.shutdown_complete() {
                return;
            }
            api.prevent_exit();
            if !state.begin_shutdown() {
                return;
            }
            hide_recording_bar(app);
            hide_settings_window(app);
            let cleanup_app = app.clone();
            let watchdog_app = app.clone();
            let exit_code = code.unwrap_or(0);
            let _watchdog = std::thread::Builder::new()
                .name("whispernow-shutdown-watchdog".to_owned())
                .spawn(move || {
                    std::thread::sleep(SHUTDOWN_WATCHDOG_TIMEOUT);
                    if !watchdog_app.state::<Gateway>().shutdown_complete() {
                        tracing::error!("shutdown deadline exceeded; forcing process exit");
                        std::process::exit(1);
                    }
                });
            tauri::async_runtime::spawn_blocking(move || {
                cleanup_app.state::<Gateway>().finish_shutdown();
                cleanup_app.exit(exit_code);
            });
        }
    });
}

fn shortcut_registrar(
    app: tauri::AppHandle,
    native_wayland: bool,
    events: tokio::sync::mpsc::UnboundedSender<ShortcutEvent>,
) -> tauri::Result<(
    Arc<dyn ShortcutRegistrar>,
    Option<shortcuts::ShortcutRegistrationError>,
)> {
    #[cfg(target_os = "linux")]
    if native_wayland {
        let identity = portal_identity::PortalIdentity::new(
            app.config().identifier.clone(),
            app.package_info().name.clone(),
            &app.path().data_dir()?,
        );
        let (registrar, failure) =
            wayland_shortcuts::WaylandShortcutRegistrar::spawn(identity, events);
        return Ok((registrar, failure));
    }
    let _ = (native_wayland, events);
    Ok((Arc::new(TauriShortcutRegistrar::new(app)), None))
}

#[cfg(target_os = "linux")]
fn native_wayland_session() -> bool {
    wayland_shortcuts::native_wayland_session()
}

#[cfg(not(target_os = "linux"))]
const fn native_wayland_session() -> bool {
    false
}

/// GNOME gives a native Wayland window no way to stay above other apps or choose
/// its position, so the recording bar could not float. In a Wayland session the
/// windows use `XWayland` instead, before GTK starts; shortcuts and paste
/// still go through the portals. An explicit `GDK_BACKEND` wins.
#[cfg(target_os = "linux")]
fn use_xwayland_windows() -> bool {
    if let Some(backend) = std::env::var_os("GDK_BACKEND") {
        return backend
            .to_str()
            .is_some_and(|backend| backend.starts_with("x11"));
    }
    if std::env::var_os("DISPLAY").is_none() {
        return false;
    }
    gdk::set_allowed_backends("x11");
    true
}

#[cfg(not(target_os = "linux"))]
const fn use_xwayland_windows() -> bool {
    false
}

/// Runs on the event-loop thread. Lifecycle actions can wait for processing,
/// native shortcut round trips, and state publication, so they run on a
/// blocking worker and report their outcome through state and error events.
fn handle_tray_action(app: &tauri::AppHandle, action: TrayAction) {
    match action {
        TrayAction::Settings => {
            show_settings_window(app);
        }
        TrayAction::CheckForUpdates => {
            if let Err(error) = app
                .state::<Gateway>()
                .services
                .state
                .refuse_during_maintenance()
            {
                tracing::debug!(code = error.code, "update check was not admitted");
                return;
            }
            let controller = Arc::clone(app.state::<Arc<updater::UpdateController>>().inner());
            tauri::async_runtime::spawn(async move {
                let _ = controller.check(true).await;
            });
        }
        TrayAction::CopyLastTranscript => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                app.state::<Gateway>()
                    .copy_last_transcript_from_tray()
                    .await;
            });
        }
        TrayAction::CancelRecording => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                app.state::<Gateway>().cancel_dictation().await;
            });
        }
        TrayAction::TogglePause => {
            let app = app.clone();
            tauri::async_runtime::spawn_blocking(move || toggle_pause(&app.state::<Gateway>()));
        }
        // The exit handler owns shutdown admission; a quit during shutdown is ignored there.
        TrayAction::Quit => app.exit(0),
    }
}

fn toggle_pause(state: &Gateway) {
    let paused = matches!(state.services.state.lock().phase, services::Phase::Paused);
    let dictation = &state.services.dictation;
    let changed = if paused {
        dictation.resume()
    } else {
        dictation.pause()
    };
    if let Err(error) = changed {
        tracing::warn!(
            code = error.code,
            paused,
            "the tray could not change dictation"
        );
        state.publish_error(&error);
    }
}

fn apply_initial_window_policy(app: &tauri::App, start_minimized: bool) {
    if start_minimized {
        hide_settings_window(app.handle());
    } else {
        show_settings_window(app.handle());
    }
    hide_recording_bar(app.handle());
}

/// Window changes are queued to the event-loop thread and never awaited. State
/// publication runs on worker threads, and blocking native window getters there
/// would wait on the event loop while it may itself wait on application state.
pub(crate) fn show_settings_window(app: &tauri::AppHandle) {
    let main_thread_app = app.clone();
    let scheduled = app.run_on_main_thread(move || {
        let Some(window) = main_thread_app.get_webview_window(SETTINGS_WINDOW) else {
            return;
        };
        if window.show().is_err() {
            tracing::warn!("the settings window could not be shown");
        } else if window.set_focus().is_err() {
            tracing::warn!("the settings window could not be focused");
        }
    });
    if scheduled.is_err() {
        tracing::warn!("the settings window show could not be scheduled");
    }
}

pub(crate) fn hide_settings_window(app: &tauri::AppHandle) {
    hide_window(app, SETTINGS_WINDOW);
}

pub(crate) fn hide_recording_bar(app: &tauri::AppHandle) {
    hide_window(app, RECORDING_BAR_WINDOW);
}

fn hide_window(app: &tauri::AppHandle, label: &'static str) {
    let main_thread_app = app.clone();
    let scheduled = app.run_on_main_thread(move || {
        let Some(window) = main_thread_app.get_webview_window(label) else {
            return;
        };
        if window.hide().is_err() {
            tracing::warn!(label, "window could not be hidden");
        }
    });
    if scheduled.is_err() {
        tracing::warn!(label, "window hide could not be scheduled");
    }
}

fn restore_settings_geometry(app: &tauri::App, geometry: WindowGeometry) {
    let Some(window) = app.get_webview_window(SETTINGS_WINDOW) else {
        return;
    };
    let monitors: Vec<_> = window
        .available_monitors()
        .unwrap_or_default()
        .iter()
        .map(overlay::monitor_area)
        .collect();
    if monitors.is_empty() {
        return;
    }
    let geometry = clamp_geometry_to_monitors(geometry, &monitors);
    let _ = window.set_size(tauri::PhysicalSize::new(geometry.width, geometry.height));
    let _ = window.set_position(tauri::PhysicalPosition::new(geometry.x, geometry.y));
}

fn clamp_geometry_to_monitors(
    mut geometry: WindowGeometry,
    monitors: &[overlay::MonitorArea],
) -> WindowGeometry {
    geometry.width = geometry.width.clamp(720, 3_840);
    geometry.height = geometry.height.clamp(520, 2_160);
    let bounds = overlay::PhysicalRect {
        x: geometry.x,
        y: geometry.y,
        width: geometry.width,
        height: geometry.height,
    };
    let Some(monitor) = monitors
        .iter()
        .find(|monitor| monitor.work_area.overlaps(&bounds))
        .or_else(|| monitors.first())
    else {
        return geometry;
    };
    let work = monitor.work_area;
    geometry.width = geometry.width.min(work.width);
    geometry.height = geometry.height.min(work.height);
    let max_x = i64::from(work.x) + i64::from(work.width - geometry.width);
    let max_y = i64::from(work.y) + i64::from(work.height - geometry.height);
    geometry.x = i64::from(geometry.x)
        .clamp(i64::from(work.x), max_x)
        .try_into()
        .unwrap_or(work.x);
    geometry.y = i64::from(geometry.y)
        .clamp(i64::from(work.y), max_y)
        .try_into()
        .unwrap_or(work.y);
    geometry
}

#[cfg(test)]
mod tests;
