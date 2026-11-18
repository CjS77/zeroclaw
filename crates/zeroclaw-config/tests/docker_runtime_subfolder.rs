//! The Docker runtime with an embedding program's workspace subfolder set:
//! only `<workspace>/work` is mounted, and anything that could widen the
//! mount is refused. The slot is process-wide, so every test here sets it to
//! the same name.
#![cfg(unix)]

#[path = "support/docker_command.rs"]
mod docker_command;

use docker_command::{docker_args, value_after, workspace};
use zeroclaw_config::platform::docker_subfolder::{
    mount_only_workspace_subfolder, workspace_subfolder,
};

fn mount_only_work() {
    mount_only_workspace_subfolder("work").unwrap();
}

fn refusal(workspace: &std::path::Path) -> String {
    let error = docker_args(workspace).unwrap_err();
    format!("{error:#}")
}

#[test]
fn only_the_work_folder_is_mounted_and_the_workdir_stays_the_workspace() {
    mount_only_work();
    let workspace = workspace();
    std::fs::create_dir(workspace.path().join("work")).unwrap();
    let args = docker_args(workspace.path()).unwrap();
    let work = workspace.path().canonicalize().unwrap().join("work");

    assert_eq!(workspace_subfolder(), Some("work"));
    assert_eq!(
        value_after(&args, "--volume"),
        Some(format!("{}:/workspace/work:rw", work.display()).as_str())
    );
    assert_eq!(args.iter().filter(|arg| *arg == "--volume").count(), 1);
    assert_eq!(value_after(&args, "--workdir"), Some("/workspace"));
}

#[test]
fn a_work_symlink_is_refused_wherever_it_points() {
    mount_only_work();
    let outside = tempfile::tempdir().unwrap();
    [
        outside.path(),
        std::path::Path::new(".."),
        std::path::Path::new("."),
        std::path::Path::new("skills"),
    ]
    .into_iter()
    .for_each(|target| {
        let workspace = workspace();
        std::fs::create_dir(workspace.path().join("skills")).unwrap();
        std::os::unix::fs::symlink(target, workspace.path().join("work")).unwrap();
        let message = refusal(workspace.path());
        assert!(
            message.contains("must be a real folder directly inside the workspace"),
            "symlink to {}: {message}",
            target.display()
        );
    });
}

#[test]
fn a_missing_work_folder_is_refused() {
    mount_only_work();
    let workspace = workspace();
    let work = workspace.path().canonicalize().unwrap().join("work");

    let message = refusal(workspace.path());
    assert!(
        message.contains(&format!(
            "{} cannot be resolved: No such file or directory",
            work.display()
        )),
        "{message}"
    );
}

#[test]
fn a_work_file_is_refused() {
    mount_only_work();
    let workspace = workspace();
    std::fs::write(workspace.path().join("work"), "not a folder").unwrap();

    let message = refusal(workspace.path());
    assert!(message.contains("is not a directory"), "{message}");
}

#[test]
fn a_name_that_is_not_one_folder_is_refused() {
    mount_only_work();
    ["", ".", "..", "a/b", "/work", "work/", "../work"]
        .into_iter()
        .for_each(|name| {
            let error = mount_only_workspace_subfolder(name).unwrap_err();
            assert!(
                error.to_string().contains("must be one folder name"),
                "{name:?}: {error}"
            );
        });
}

#[test]
fn a_second_different_name_is_refused_and_the_same_name_is_accepted() {
    mount_only_work();
    let error = mount_only_workspace_subfolder("other").unwrap_err();

    assert!(error.to_string().contains("already set"), "{error}");
    assert!(mount_only_workspace_subfolder("work").is_ok());
    assert_eq!(workspace_subfolder(), Some("work"));
}
