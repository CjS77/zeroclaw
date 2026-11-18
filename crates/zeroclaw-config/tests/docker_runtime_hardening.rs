//! The Docker runtime's always-on container hardening, and its whole-workspace
//! mount when no embedding program has set a workspace subfolder. The
//! subfolder slot is process-wide, so this binary never sets it.
#![cfg(unix)]

#[path = "support/docker_command.rs"]
mod docker_command;

use docker_command::{docker_args, value_after, workspace};
use zeroclaw_config::platform::docker_subfolder::workspace_subfolder;

#[test]
fn every_container_drops_capabilities_and_privileges_and_limits_processes() {
    let workspace = workspace();
    let args = docker_args(workspace.path()).unwrap();

    assert_eq!(value_after(&args, "--cap-drop"), Some("ALL"));
    assert_eq!(
        value_after(&args, "--security-opt"),
        Some("no-new-privileges")
    );
    assert_eq!(value_after(&args, "--pids-limit"), Some("256"));
    assert_eq!(
        value_after(&args, "--tmpfs"),
        Some("/tmp:rw,nosuid,nodev,size=256m")
    );
}

#[test]
fn without_a_subfolder_the_whole_workspace_is_mounted() {
    let workspace = workspace();
    std::fs::create_dir(workspace.path().join("work")).unwrap();
    let args = docker_args(workspace.path()).unwrap();
    let canonical = workspace.path().canonicalize().unwrap();

    assert_eq!(workspace_subfolder(), None);
    assert_eq!(
        value_after(&args, "--volume"),
        Some(format!("{}:/workspace:rw", canonical.display()).as_str())
    );
    assert_eq!(value_after(&args, "--workdir"), Some("/workspace"));
}
