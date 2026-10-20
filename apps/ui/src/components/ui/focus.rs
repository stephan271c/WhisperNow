use dioxus::prelude::*;

pub(crate) fn take_focus(element: Signal<Option<std::rc::Rc<MountedData>>>) {
    if let Some(element) = element.peek().clone() {
        focus(element);
    }
}

pub(crate) fn focus(element: std::rc::Rc<MountedData>) {
    spawn(async move {
        let _ = element.set_focus(true).await;
    });
}
