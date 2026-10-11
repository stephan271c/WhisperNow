use dioxus::prelude::*;

/// The item next to position `index` once it is gone: the one after it, else the one before.
pub(crate) fn neighbour<T: Clone>(items: &[T], index: usize) -> Option<T> {
    items
        .get(index + 1)
        .or_else(|| index.checked_sub(1).and_then(|before| items.get(before)))
        .cloned()
}

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
