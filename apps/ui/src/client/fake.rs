//! The desktop as native tests see it: every call is recorded and waits until the test
//! answers it, and every timer waits until the test runs the timers.

use std::{
    cell::RefCell,
    future::poll_fn,
    rc::Rc,
    task::{Poll, Waker},
};
use whispernow_domain::AppError;

use serde::Serialize;
use serde_json::Value;

type Reply = Result<Value, AppError>;

#[derive(Default)]
struct Slot {
    reply: Option<Reply>,
    /// Whether the call has an answer, taken or not.
    answered: bool,
    waker: Option<Waker>,
}

struct Call {
    command: &'static str,
    request: Option<Value>,
    slot: Rc<RefCell<Slot>>,
}

#[derive(Default)]
struct Desktop {
    calls: Vec<Call>,
}

#[derive(Default)]
struct Timers {
    runs: u64,
    waiting: Vec<Waker>,
}

thread_local! {
    static DESKTOP: RefCell<Desktop> = RefCell::new(Desktop::default());
    static TIMERS: RefCell<Timers> = RefCell::new(Timers::default());
}

pub(super) async fn delay() {
    let started = TIMERS.with_borrow(|timers| timers.runs);
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

pub(super) async fn invoke(command: &'static str, request: Option<Value>) -> Reply {
    let slot = Rc::new(RefCell::new(Slot::default()));
    DESKTOP.with_borrow_mut(|desktop| {
        desktop.calls.push(Call {
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

pub(crate) fn reset() {
    DESKTOP.with_borrow_mut(|desktop| *desktop = Desktop::default());
    TIMERS.with_borrow_mut(|timers| *timers = Timers::default());
}

/// Answers the oldest unanswered call of `command`.
pub(crate) fn answer(command: &str, answer: Result<impl Serialize, AppError>) {
    let answer = reply(answer);
    let waker = DESKTOP.with_borrow(|desktop| {
        let call = desktop
            .calls
            .iter()
            .find(|call| call.command == command && !call.slot.borrow().answered)
            .unwrap_or_else(|| panic!("no {command} call is waiting"));
        let mut slot = call.slot.borrow_mut();
        slot.reply = Some(answer);
        slot.answered = true;
        slot.waker.take()
    });
    if let Some(waker) = waker {
        waker.wake();
    }
}

/// Every command called so far, oldest first.
pub(crate) fn commands() -> Vec<&'static str> {
    DESKTOP.with_borrow(|desktop| desktop.calls.iter().map(|call| call.command).collect())
}

/// The requests of every call of `command` so far, oldest first.
pub(crate) fn requests<T: serde::de::DeserializeOwned>(command: &str) -> Vec<T> {
    DESKTOP.with_borrow(|desktop| {
        desktop
            .calls
            .iter()
            .filter(|call| call.command == command)
            .map(|call| {
                serde_json::from_value(call.request.clone().unwrap_or_default())
                    .expect("request of the asked type")
            })
            .collect()
    })
}

pub(crate) fn count(command: &str) -> usize {
    DESKTOP.with_borrow(|desktop| {
        desktop
            .calls
            .iter()
            .filter(|call| call.command == command)
            .count()
    })
}

/// Fails the oldest unanswered call of `command` as if the bridge had dropped it.
pub(crate) fn fail(command: &str) {
    answer(command, Err::<(), _>(crate::client::connection_error()));
}
