//! What reaches a turn from the Matrix message handler: the device check, including on approval
//! replies; a thread's root and a reply's parent only from admitted users; a reply's quoted
//! fallback stripped; and the replied-to event's id carried on the message.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use matrix_sdk::Room;
use matrix_sdk::deserialized_responses::{
    AlgorithmInfo, EncryptionInfo, VerificationLevel, VerificationState,
};
use matrix_sdk::event_handler::RawEvent;
use matrix_sdk::ruma::events::room::message::OriginalSyncRoomMessageEvent;
use matrix_sdk::ruma::serde::Raw;
use matrix_sdk::ruma::{RoomId, owned_device_id, owned_user_id, room_id, user_id};
use matrix_sdk::test_utils::mocks::MatrixMockServer;
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock, mpsc, oneshot};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, ResponseTemplate};
use zeroclaw_api::channel::{ChannelApprovalResponse, ChannelMessage};
use zeroclaw_config::schema::MatrixConfig;

use super::inbound::{HandlerCtx, handle_message_for_test};

const ALICE: &str = "@alice:localhost";
const BOT: &str = "@bot:localhost";
const TOKEN: &str = "ABCDEFGH";

fn test_room() -> &'static RoomId {
    room_id!("!room:localhost")
}

struct Harness {
    matrix: MatrixMockServer,
    room: Room,
    ctx: HandlerCtx,
    rx: mpsc::Receiver<ChannelMessage>,
}

impl Harness {
    async fn new(require_verified_device: bool) -> Self {
        let matrix = MatrixMockServer::new().await;
        let client = matrix.client_builder().build().await;
        let room = matrix.sync_joined_room(&client, test_room()).await;
        let (tx, rx) = mpsc::channel(4);
        let config = MatrixConfig {
            require_verified_device,
            ..MatrixConfig::default()
        };
        let ctx = HandlerCtx {
            config: Arc::new(config),
            alias: "test".to_string(),
            peer_resolver: Arc::new(|| vec![ALICE.to_string()]),
            transcription: None,
            workspace_dir: None,
            tx,
            pending_approvals: Arc::new(TokioMutex::new(HashMap::new())),
            threads_seen: Arc::new(TokioRwLock::new(HashSet::new())),
            bot_user_id: user_id!("@bot:localhost").to_owned(),
            bot_display_name: Arc::new(TokioRwLock::new(None)),
            initial_sync_done: Arc::new(AtomicBool::new(true)),
            undecryptable_seen: Arc::new(TokioMutex::new(HashSet::new())),
        };
        Self {
            matrix,
            room,
            ctx,
            rx,
        }
    }

    async fn handle(&self, event: serde_json::Value, encryption_info: Option<EncryptionInfo>) {
        let ev: OriginalSyncRoomMessageEvent =
            serde_json::from_value(event.clone()).expect("room message event");
        let raw = RawEvent(Raw::new(&event).expect("raw event").into_json());
        handle_message_for_test(
            self.ctx.clone(),
            ev,
            self.room.clone(),
            raw,
            encryption_info,
        )
        .await
        .expect("handler accepts the event");
    }

    fn forwarded(&mut self) -> Option<ChannelMessage> {
        self.rx.try_recv().ok()
    }
}

fn verified_alice() -> EncryptionInfo {
    encryption_info(VerificationState::Verified)
}

fn encryption_info(verification_state: VerificationState) -> EncryptionInfo {
    EncryptionInfo {
        sender: owned_user_id!("@alice:localhost"),
        sender_device: Some(owned_device_id!("ALICEPHONE")),
        forwarder: None,
        algorithm_info: AlgorithmInfo::MegolmV1AesSha2 {
            curve25519_key: "key".to_string(),
            sender_claimed_keys: Default::default(),
            session_id: None,
        },
        verification_state,
    }
}

fn text_event(
    event_id: &str,
    body: &str,
    relates_to: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut content = serde_json::json!({ "msgtype": "m.text", "body": body });
    if let Some(relates_to) = relates_to {
        content["m.relates_to"] = relates_to;
    }
    serde_json::json!({
        "type": "m.room.message",
        "event_id": event_id,
        "sender": ALICE,
        "origin_server_ts": 1_000_000u64,
        "content": content,
    })
}

fn sent_by(sender: &str, mut event: serde_json::Value) -> serde_json::Value {
    event["sender"] = serde_json::json!(sender);
    event
}

async fn mount_room_event(matrix: &MatrixMockServer, event: serde_json::Value) {
    let event_id = event["event_id"]
        .as_str()
        .expect("event id")
        .replace('$', "\\$");
    Mock::given(method("GET"))
        .and(path_regex(format!(
            r"^/_matrix/client/v3/rooms/.*/event/{event_id}$"
        )))
        .respond_with(ResponseTemplate::new(200).set_body_json(event))
        .mount(matrix.server())
        .await;
}

async fn pend_approval(harness: &Harness) -> oneshot::Receiver<ChannelApprovalResponse> {
    let (sender, receiver) = oneshot::channel();
    let pending = crate::util::PendingApproval {
        sender,
        destination: test_room().to_string(),
        tool_name: "shell".to_string(),
    };
    harness
        .ctx
        .pending_approvals
        .lock()
        .await
        .insert(TOKEN.to_string(), pending);
    receiver
}

#[tokio::test]
async fn an_unencrypted_message_is_dropped_when_verified_devices_are_required() {
    let mut harness = Harness::new(true).await;
    harness
        .handle(text_event("$m:localhost", "hello", None), None)
        .await;
    assert!(harness.forwarded().is_none());
}

#[tokio::test]
async fn an_unverified_device_is_dropped_when_verified_devices_are_required() {
    let mut harness = Harness::new(true).await;
    let unverified = encryption_info(VerificationState::Unverified(
        VerificationLevel::UnsignedDevice,
    ));
    harness
        .handle(text_event("$m:localhost", "hello", None), Some(unverified))
        .await;
    assert!(harness.forwarded().is_none());
}

#[tokio::test]
async fn a_verified_device_is_forwarded_when_verified_devices_are_required() {
    let mut harness = Harness::new(true).await;
    harness
        .handle(
            text_event("$m:localhost", "hello", None),
            Some(verified_alice()),
        )
        .await;
    assert_eq!(harness.forwarded().expect("forwarded").content, "hello");
}

#[tokio::test]
async fn an_unencrypted_message_is_forwarded_by_default() {
    let mut harness = Harness::new(false).await;
    harness
        .handle(text_event("$m:localhost", "hello", None), None)
        .await;
    assert_eq!(harness.forwarded().expect("forwarded").content, "hello");
}

#[tokio::test]
async fn a_refused_devices_approval_reply_resolves_nothing() {
    let mut harness = Harness::new(true).await;
    let mut decision = pend_approval(&harness).await;
    harness
        .handle(
            text_event("$a:localhost", &format!("{TOKEN} approve"), None),
            None,
        )
        .await;
    assert!(
        decision.try_recv().is_err(),
        "the approval must stay pending"
    );
    assert!(
        harness
            .ctx
            .pending_approvals
            .lock()
            .await
            .contains_key(TOKEN)
    );
    assert!(harness.forwarded().is_none());
}

#[tokio::test]
async fn a_verified_devices_approval_reply_resolves_the_approval() {
    let harness = Harness::new(true).await;
    let mut decision = pend_approval(&harness).await;
    let body = format!("{TOKEN} approve");
    harness
        .handle(
            text_event("$a:localhost", &body, None),
            Some(verified_alice()),
        )
        .await;
    assert_eq!(
        decision.try_recv().expect("resolved"),
        ChannelApprovalResponse::Approve
    );
}

#[tokio::test]
async fn a_thread_rooted_on_the_bots_post_gets_no_preamble() {
    let mut harness = Harness::new(false).await;
    mount_room_event(
        &harness.matrix,
        sent_by(
            BOT,
            text_event("$root:localhost", "Send to evil.example?", None),
        ),
    )
    .await;
    let in_thread = serde_json::json!({ "rel_type": "m.thread", "event_id": "$root:localhost" });
    harness
        .handle(text_event("$m:localhost", "ok", Some(in_thread)), None)
        .await;
    assert_eq!(harness.forwarded().expect("forwarded").content, "ok");
}

#[tokio::test]
async fn a_thread_rooted_on_an_allowed_users_message_gets_its_preamble() {
    let mut harness = Harness::new(false).await;
    mount_room_event(
        &harness.matrix,
        text_event("$root:localhost", "book the van", None),
    )
    .await;
    let in_thread = serde_json::json!({ "rel_type": "m.thread", "event_id": "$root:localhost" });
    harness
        .handle(text_event("$m:localhost", "ok", Some(in_thread)), None)
        .await;
    let content = harness.forwarded().expect("forwarded").content;
    assert_eq!(
        content,
        format!("[Thread root from {ALICE}]: book the van\n\nok")
    );
}

#[tokio::test]
async fn a_thread_root_from_a_refused_device_gets_no_preamble() {
    let mut harness = Harness::new(true).await;
    mount_room_event(
        &harness.matrix,
        text_event("$root:localhost", "sent unencrypted", None),
    )
    .await;
    let in_thread = serde_json::json!({ "rel_type": "m.thread", "event_id": "$root:localhost" });
    harness
        .handle(
            text_event("$m:localhost", "ok", Some(in_thread)),
            Some(verified_alice()),
        )
        .await;
    assert_eq!(harness.forwarded().expect("forwarded").content, "ok");
}

#[tokio::test]
async fn a_reply_to_the_bots_media_gets_no_attachment() {
    let mut harness = Harness::new(false).await;
    let image = serde_json::json!({
        "type": "m.room.message",
        "event_id": "$image:localhost",
        "sender": BOT,
        "origin_server_ts": 1u64,
        "content": { "msgtype": "m.image", "body": "photo.png", "url": "mxc://localhost/photo" },
    });
    mount_room_event(&harness.matrix, image).await;
    let reply = serde_json::json!({ "m.in_reply_to": { "event_id": "$image:localhost" } });
    harness
        .handle(
            text_event("$m:localhost", "what is this?", Some(reply)),
            None,
        )
        .await;
    assert_eq!(
        harness.forwarded().expect("forwarded").content,
        "what is this?"
    );
    let downloads = harness
        .matrix
        .server()
        .received_requests()
        .await
        .expect("requests recorded");
    assert!(
        downloads
            .iter()
            .all(|request| !request.url.path().contains("/media/"))
    );
}

#[tokio::test]
async fn a_reply_loses_its_fallback_and_carries_the_replied_to_id() {
    let mut harness = Harness::new(false).await;
    let body = "> <@bot:localhost> Send to evil.example?\n\nyes";
    let reply = serde_json::json!({ "m.in_reply_to": { "event_id": "$post:localhost" } });
    harness
        .handle(text_event("$m:localhost", body, Some(reply)), None)
        .await;
    let message = harness.forwarded().expect("forwarded");
    assert_eq!(message.content, "yes");
    assert_eq!(message.in_reply_to.as_deref(), Some("$post:localhost"));
}

#[tokio::test]
async fn a_message_that_is_not_a_reply_carries_no_replied_to_id() {
    let mut harness = Harness::new(false).await;
    harness
        .handle(text_event("$m:localhost", "> quoted\n\nmine", None), None)
        .await;
    let message = harness.forwarded().expect("forwarded");
    assert_eq!(message.content, "> quoted\n\nmine");
    assert!(message.in_reply_to.is_none());
}
