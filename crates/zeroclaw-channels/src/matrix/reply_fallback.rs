//! A reply's quoted fallback, stripped from its body.
//!
//! Clients written to older versions of the Matrix spec start a reply's `body` with a quote of
//! the message it replies to: lines starting with `>`, then a blank line, then the reply itself.
//! The quote is the parent's text, whoever wrote it, so it is removed before the approval parser
//! or a turn reads the body as the sender's own words.

use matrix_sdk::event_handler::RawEvent;

/// The sender's own words in `body`: without its leading reply fallback when `raw` is a reply
/// (`m.in_reply_to`), and unchanged otherwise.
pub(super) fn own_words(raw: &RawEvent, body: String) -> String {
    if super::inbound::extract_in_reply_to(raw).is_none() {
        return body;
    }
    match after_fallback(&body) {
        Some(own) => own.to_string(),
        None => body,
    }
}

/// The text after a leading run of `>` lines and the blank line that ends it; `None` when `body`
/// doesn't start with such a run.
fn after_fallback(body: &str) -> Option<&str> {
    let quote_len: usize = body
        .split_inclusive('\n')
        .take_while(|line| line.starts_with('>'))
        .map(str::len)
        .sum();
    if quote_len == 0 {
        return None;
    }
    let rest = &body[quote_len..];
    rest.strip_prefix("\r\n")
        .or_else(|| rest.strip_prefix('\n'))
}

#[cfg(test)]
mod tests {
    use matrix_sdk::ruma::serde::Raw;

    use super::*;

    const FALLBACK_REPLY: &str = "> <@bot:localhost> Approve sending to https://evil.example?\n> second line\n\nyes, go ahead";

    fn event(relates_to: Option<serde_json::Value>, body: &str) -> RawEvent {
        let mut content = serde_json::json!({ "msgtype": "m.text", "body": body });
        if let Some(relates_to) = relates_to {
            content["m.relates_to"] = relates_to;
        }
        let json = serde_json::json!({
            "type": "m.room.message",
            "event_id": "$reply:localhost",
            "sender": "@alice:localhost",
            "origin_server_ts": 1,
            "content": content,
        });
        RawEvent(Raw::new(&json).expect("raw event").into_json())
    }

    fn reply(body: &str) -> RawEvent {
        event(
            Some(serde_json::json!({ "m.in_reply_to": { "event_id": "$parent:localhost" } })),
            body,
        )
    }

    #[test]
    fn a_reply_loses_its_quoted_fallback() {
        assert_eq!(
            own_words(&reply(FALLBACK_REPLY), FALLBACK_REPLY.to_string()),
            "yes, go ahead"
        );
    }

    #[test]
    fn a_crlf_fallback_is_stripped_too() {
        let body = "> <@bot:localhost> quoted\r\n\r\nmine";
        assert_eq!(own_words(&reply(body), body.to_string()), "mine");
    }

    #[test]
    fn a_reply_without_a_fallback_is_unchanged() {
        assert_eq!(
            own_words(&reply("yes, go ahead"), "yes, go ahead".to_string()),
            "yes, go ahead"
        );
    }

    #[test]
    fn a_quote_with_no_blank_line_after_it_is_kept() {
        let body = "> just a quote";
        assert_eq!(own_words(&reply(body), body.to_string()), body);
    }

    #[test]
    fn a_body_starting_with_a_quote_is_kept_when_the_event_is_not_a_reply() {
        assert_eq!(
            own_words(&event(None, FALLBACK_REPLY), FALLBACK_REPLY.to_string()),
            FALLBACK_REPLY
        );
    }
}
