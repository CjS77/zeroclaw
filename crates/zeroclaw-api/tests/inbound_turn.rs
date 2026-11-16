use zeroclaw_api::inbound_turn::{InboundTurn, current, scope};

#[tokio::test]
async fn a_scoped_future_reads_its_turn_and_the_next_reads_its_own() {
    let first = scope("turn-a", "$a", async { current() }).await;
    let second = scope("turn-b", "$b", async { current() }).await;
    let turn = |turn_id: &str, message_id: &str| InboundTurn {
        turn_id: turn_id.into(),
        message_id: message_id.into(),
    };
    assert_eq!(first, Some(turn("turn-a", "$a")));
    assert_eq!(second, Some(turn("turn-b", "$b")));
}

#[tokio::test]
async fn outside_a_scope_there_is_no_turn() {
    assert_eq!(current(), None);
}
