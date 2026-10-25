//! The desktop as native tests see it: every call is recorded and waits until the test
//! answers it, and every timer waits until the test runs the timers.

use std::{
    cell::{Cell, RefCell},
    future::poll_fn,
    rc::Rc,
    task::{Poll, Waker},
};
use whispernow_domain::{AppError, ipc::COMMAND_NAMES};

use super::{Platform, WindowFrame};

use serde::Serialize;
use serde_json::Value;

type Reply = Result<Value, AppError>;

#[derive(Default)]
struct Slot {
    reply: Option<Reply>,
    answered: bool,
    waker: Option<Waker>,
}

struct Call {
    command: &'static str,
    request: Value,
    slot: Rc<RefCell<Slot>>,
}

#[derive(Default)]
struct Timers {
    runs: u64,
    waiting: Vec<Waker>,
    live: usize,
}

/// Counts a delay as live until it finishes or its task is cancelled.
struct LiveDelay;

impl Drop for LiveDelay {
    fn drop(&mut self) {
        TIMERS.with_borrow_mut(|timers| timers.live = timers.live.saturating_sub(1));
    }
}

thread_local! {
    static FRAME: Cell<(WindowFrame, Platform)> = const { Cell::new((WindowFrame::Native, Platform::Other)) };
    static CALLS: RefCell<Vec<Call>> = const { RefCell::new(Vec::new()) };
    static TIMERS: RefCell<Timers> = RefCell::new(Timers::default());
    static THEMES: RefCell<Vec<Option<String>>> = const { RefCell::new(Vec::new()) };
}

pub(super) async fn delay() {
    let started = TIMERS.with_borrow_mut(|timers| {
        timers.live += 1;
        timers.runs
    });
    let _live = LiveDelay;
    poll_fn(|context| {
        TIMERS.with_borrow_mut(|timers| {
            if timers.runs > started {
                Poll::Ready(())
            } else {
                timers.waiting.push(context.waker().clone());
                Poll::Pending
            }
        })
    })
    .await;
}

pub(crate) fn run_timers() {
    let waiting = TIMERS.with_borrow_mut(|timers| {
        timers.runs += 1;
        std::mem::take(&mut timers.waiting)
    });
    for waker in waiting {
        waker.wake();
    }
}

/// How many delays have started and neither finished nor been cancelled.
pub(crate) fn live_delays() -> usize {
    TIMERS.with_borrow(|timers| timers.live)
}

pub(super) async fn invoke(command: &'static str, request: Value) -> Reply {
    assert!(
        COMMAND_NAMES.contains(&command),
        "the desktop has no {command} command"
    );
    let slot = Rc::new(RefCell::new(Slot::default()));
    CALLS.with_borrow_mut(|calls| {
        calls.push(Call {
            command,
            request,
            slot: Rc::clone(&slot),
        });
    });
    poll_fn(|context| {
        let mut slot = slot.borrow_mut();
        if let Some(reply) = slot.reply.take() {
            return Poll::Ready(reply);
        }
        slot.waker = Some(context.waker().clone());
        Poll::Pending
    })
    .await
}

fn reply(answer: Result<impl Serialize, AppError>) -> Reply {
    answer.map(|value| serde_json::to_value(value).expect("serializable answer"))
}

pub(crate) fn set_frame(frame: WindowFrame, platform: Platform) {
    FRAME.set((frame, platform));
}

pub(super) fn frame() -> (WindowFrame, Platform) {
    FRAME.get()
}

pub(super) fn apply_theme(theme: Option<&str>) {
    THEMES.with_borrow_mut(|themes| themes.push(theme.map(str::to_owned)));
}

/// Every theme the document was given, in order.
pub(crate) fn themes() -> Vec<Option<String>> {
    THEMES.with_borrow(Clone::clone)
}

pub(crate) fn reset() {
    FRAME.set((WindowFrame::Native, Platform::Other));
    THEMES.with_borrow_mut(Vec::clear);
    CALLS.with_borrow_mut(Vec::clear);
    TIMERS.with_borrow_mut(|timers| *timers = Timers::default());
}

pub(crate) fn answer(command: &str, answer: Result<impl Serialize, AppError>) {
    let index = CALLS.with_borrow(|calls| {
        calls
            .iter()
            .filter(|call| call.command == command)
            .position(|call| !call.slot.borrow().answered)
            .unwrap_or_else(|| panic!("no {command} call is waiting"))
    });
    answer_nth(command, index, answer);
}

pub(crate) fn answer_nth(command: &str, index: usize, answer: Result<impl Serialize, AppError>) {
    let answer = reply(answer);
    let waker = CALLS.with_borrow(|calls| {
        let call = calls
            .iter()
            .filter(|call| call.command == command)
            .nth(index)
            .unwrap_or_else(|| panic!("no {command} call at {index}"));
        let mut slot = call.slot.borrow_mut();
        assert!(!slot.answered, "{command} call at {index} was answered");
        slot.reply = Some(answer);
        slot.answered = true;
        slot.waker.take()
    });
    if let Some(waker) = waker {
        waker.wake();
    }
}

pub(crate) fn commands() -> Vec<&'static str> {
    CALLS.with_borrow(|calls| calls.iter().map(|call| call.command).collect())
}

pub(crate) fn requests<T: serde::de::DeserializeOwned>(command: &str) -> Vec<T> {
    CALLS.with_borrow(|calls| {
        calls
            .iter()
            .filter(|call| call.command == command)
            .map(|call| {
                serde_json::from_value(call.request.clone()).expect("request of the asked type")
            })
            .collect()
    })
}

pub(crate) fn count(command: &str) -> usize {
    CALLS.with_borrow(|calls| calls.iter().filter(|call| call.command == command).count())
}

/// Fails the oldest unanswered call of `command` as if the bridge had dropped it.
pub(crate) fn fail(command: &str) {
    answer(command, Err::<(), _>(crate::client::connection_error()));
}
