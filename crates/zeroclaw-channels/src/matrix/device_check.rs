//! The device check: which devices may reach the agent through this channel.
//!
//! The sender allowlist admits a Matrix user ID, but anyone signed in to that account on a new
//! device sends as the same user. With `require_verified_device` on, an event must also have been
//! encrypted by a device of its sender that this client sees as `Verified`: cross-signed by the
//! sender's identity, which this client has verified. `allowed_devices` can narrow that to named
//! devices. The check runs on an inbound event before anything reads it, approval replies
//! included, and on the events the handler fetches into a turn: a thread's root and a reply's
//! parent.

use matrix_sdk::{
    Room,
    deserialized_responses::{EncryptionInfo, TimelineEvent, VerificationState},
    ruma::{RoomId, UserId},
};
use zeroclaw_config::schema::MatrixConfig;

/// Whether `config` admits an inbound event from `sender`, decrypted with `encryption_info`
/// (`None` when the event was not encrypted). A refusal is logged with the sender, device and
/// room, never the event's content.
pub(super) fn admit(
    config: &MatrixConfig,
    sender: &UserId,
    encryption_info: Option<&EncryptionInfo>,
    room: &Room,
) -> bool {
    let admitted = admits(config, sender, encryption_info);
    if !admitted {
        log_refusal(sender, encryption_info, room.room_id());
    }
    admitted
}

/// Whether an event the handler fetched from the room, a thread's root or a reply's parent, may
/// enter a turn: its sender is on the allowlist, and `config` admits its device.
pub(super) fn admits_fetched(
    config: &MatrixConfig,
    allowed_users: &[String],
    event: &TimelineEvent,
) -> bool {
    event.sender().is_some_and(|sender| {
        super::allowlist::user_allowed(allowed_users, sender.as_str())
            && admits(config, &sender, event.encryption_info().map(|info| &**info))
    })
}

fn admits(
    config: &MatrixConfig,
    sender: &UserId,
    encryption_info: Option<&EncryptionInfo>,
) -> bool {
    !config.require_verified_device
        || encryption_info
            .is_some_and(|info| is_verified(sender, info) && is_allowed(config, sender, info))
}

/// matrix-sdk vouches for `info.sender` and `info.sender_device` only when the state is
/// `Verified`, and the event's own `sender` field must name the same user.
fn is_verified(sender: &UserId, info: &EncryptionInfo) -> bool {
    info.verification_state == VerificationState::Verified && info.sender == sender
}

fn is_allowed(config: &MatrixConfig, sender: &UserId, info: &EncryptionInfo) -> bool {
    if config.allowed_devices.is_empty() {
        return true;
    }
    let listed = config.allowed_devices.get(sender.as_str());
    info.sender_device.as_ref().is_some_and(|device| {
        listed.is_some_and(|devices| devices.iter().any(|d| d == device.as_str()))
    })
}

fn log_refusal(sender: &UserId, encryption_info: Option<&EncryptionInfo>, room_id: &RoomId) {
    ::zeroclaw_log::record!(
        INFO,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_attrs(
            ::serde_json::json!({
                "sender": sender.as_str(),
                "device": encryption_info.and_then(|info| info.sender_device.as_ref()).map(|d| d.as_str()),
                "encrypted": encryption_info.is_some(),
                "room_id": room_id.as_str(),
            })
        ),
        "matrix: drop event from a device the device check refuses"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use matrix_sdk::{
        deserialized_responses::{AlgorithmInfo, VerificationLevel},
        ruma::{owned_device_id, user_id},
    };

    use super::*;

    const ALICE: &str = "@alice:localhost";

    fn info(sender: &str, state: VerificationState) -> EncryptionInfo {
        EncryptionInfo {
            sender: sender.try_into().expect("user id"),
            sender_device: Some(owned_device_id!("ALICEPHONE")),
            forwarder: None,
            algorithm_info: AlgorithmInfo::MegolmV1AesSha2 {
                curve25519_key: "key".to_string(),
                sender_claimed_keys: Default::default(),
                session_id: None,
            },
            verification_state: state,
        }
    }

    fn verified() -> EncryptionInfo {
        info(ALICE, VerificationState::Verified)
    }

    fn unverified() -> EncryptionInfo {
        info(
            ALICE,
            VerificationState::Unverified(VerificationLevel::UnsignedDevice),
        )
    }

    fn config(require_verified_device: bool) -> MatrixConfig {
        MatrixConfig {
            require_verified_device,
            ..MatrixConfig::default()
        }
    }

    fn with_devices(devices: &[(&str, &[&str])]) -> MatrixConfig {
        let allowed_devices: HashMap<String, Vec<String>> = devices
            .iter()
            .map(|(user, ids)| {
                (
                    user.to_string(),
                    ids.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        MatrixConfig {
            allowed_devices,
            ..config(true)
        }
    }

    fn alice() -> &'static UserId {
        user_id!("@alice:localhost")
    }

    #[test]
    fn off_by_default_and_then_admits_every_event() {
        assert!(!MatrixConfig::default().require_verified_device);
        assert!(MatrixConfig::default().allowed_devices.is_empty());
        let off = config(false);
        let mismatched = info("@mallory:localhost", VerificationState::Verified);
        [
            None,
            Some(&unverified()),
            Some(&verified()),
            Some(&mismatched),
        ]
        .into_iter()
        .for_each(|encryption| assert!(admits(&off, alice(), encryption)));
    }

    #[test]
    fn when_required_an_unencrypted_event_is_refused() {
        assert!(!admits(&config(true), alice(), None));
    }

    #[test]
    fn when_required_an_unverified_device_is_refused() {
        assert!(!admits(&config(true), alice(), Some(&unverified())));
    }

    #[test]
    fn when_required_a_verified_device_is_admitted() {
        assert!(admits(&config(true), alice(), Some(&verified())));
    }

    #[test]
    fn when_required_a_sender_mismatch_is_refused() {
        let mismatched = info("@mallory:localhost", VerificationState::Verified);
        assert!(!admits(&config(true), alice(), Some(&mismatched)));
    }

    #[test]
    fn allowed_devices_admit_only_the_listed_device_of_its_user() {
        assert!(admits(
            &with_devices(&[(ALICE, &["ALICEPHONE"])]),
            alice(),
            Some(&verified())
        ));
        assert!(!admits(
            &with_devices(&[(ALICE, &["ALICETABLET"])]),
            alice(),
            Some(&verified())
        ));
        assert!(!admits(
            &with_devices(&[("@bob:localhost", &["ALICEPHONE"])]),
            alice(),
            Some(&verified())
        ));
    }

    #[test]
    fn allowed_devices_never_admit_an_unverified_device() {
        assert!(!admits(
            &with_devices(&[(ALICE, &["ALICEPHONE"])]),
            alice(),
            Some(&unverified())
        ));
    }

    #[test]
    fn allowed_devices_are_ignored_when_verification_is_not_required() {
        let config = MatrixConfig {
            require_verified_device: false,
            ..with_devices(&[(ALICE, &["ALICETABLET"])])
        };
        assert!(admits(&config, alice(), None));
    }

    #[test]
    fn allowed_devices_parse_from_toml() {
        let config: MatrixConfig = toml::from_str(
            r#"
            homeserver = "https://matrix.example.org"
            require_verified_device = true
            [allowed_devices]
            "@alice:localhost" = ["ALICEPHONE"]
            "#,
        )
        .expect("config parses");
        assert!(config.require_verified_device);
        assert_eq!(
            config.allowed_devices[ALICE],
            vec!["ALICEPHONE".to_string()]
        );
    }
}
