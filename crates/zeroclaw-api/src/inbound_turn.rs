//! The inbound message a channel turn answers, and the turn's id, for hooks that run inside the turn.
//!
//! The channel runner's inbound-message hook runs before the turn exists, and a tool-call hook only learns the turn id
//! from its call id. This task-local, scoped around the turn by the channel runner, gives any hook that runs inside
//! the turn both, including a hook that runs before the turn's first model call.

use std::future::Future;

/// The turn a channel message started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InboundTurn {
    /// The turn's id: the first part of the id of each of its tool calls, `{turn_id}:{iteration}:{call_index}`.
    pub turn_id: String,
    /// The channel's id for the message that started the turn (`ChannelMessage::id`); empty when the channel gave none.
    pub message_id: String,
}

tokio::task_local! {
    static INBOUND_TURN: InboundTurn;
}

/// Runs `future` with the turn `turn_id`, started by the message `message_id`, readable through [`current`].
pub async fn scope<F: Future>(turn_id: &str, message_id: &str, future: F) -> F::Output {
    let turn = InboundTurn {
        turn_id: turn_id.to_owned(),
        message_id: message_id.to_owned(),
    };
    INBOUND_TURN.scope(turn, future).await
}

/// The channel turn the calling task runs in, or `None` outside one (scheduled jobs, the gateway, the CLI).
pub fn current() -> Option<InboundTurn> {
    INBOUND_TURN.try_with(Clone::clone).ok()
}
