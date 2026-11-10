#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // WebKitGTK's DMA-BUF renderer draws a blank window on some drivers, such as NVIDIA on
    // Wayland (issue #4). Set before any thread starts, and only when the user hasn't set it.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        #[allow(unsafe_code, reason = "set_var runs before any other thread exists")]
        unsafe {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }
    whispernow_desktop::run();
}
