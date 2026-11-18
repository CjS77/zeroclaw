//! Lets a program that embeds ZeroClaw mount one subfolder of the agent
//! workspace into Docker runtime containers, instead of the whole workspace.
//!
//! The workspace also holds the files every turn reads as instructions
//! (`AGENTS.md`, `MEMORY.md`). With a subfolder set, those files are not in
//! the container, and the rest of `/workspace` belongs to the container's own
//! root filesystem: read-only under `read_only_rootfs`, and gone when the
//! container exits.

use anyhow::{Result, bail};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// Where the Docker runtime mounts the workspace inside the container.
const CONTAINER_WORKSPACE: &str = "/workspace";

static WORKSPACE_SUBFOLDER: OnceLock<String> = OnceLock::new();

/// Mount only `<workspace>/<name>` into every Docker runtime container, at
/// `/workspace/<name>`, for the rest of this process.
///
/// A program that embeds ZeroClaw calls this once, at start-up. The workdir
/// stays `/workspace`, so a workspace-relative path such as `<name>/report.pdf`
/// names the same file for the file tools and the shell. Each command then
/// refuses to start unless `<workspace>/<name>` is a real directory directly
/// inside the workspace, not a symlink.
///
/// # Errors
///
/// Returns an error if `name` is not one normal path component (empty, `.`,
/// `..`, or containing a separator), or if a different name is already set.
pub fn mount_only_workspace_subfolder(name: &str) -> Result<()> {
    if !is_one_normal_component(name) {
        bail!("Docker workspace subfolder must be one folder name, got {name:?}");
    }
    let set = WORKSPACE_SUBFOLDER.get_or_init(|| name.to_owned());
    if set != name {
        bail!("Docker workspace subfolder is already set to {set:?}, refusing {name:?}");
    }
    Ok(())
}

/// The workspace subfolder that Docker runtime containers mount, if an
/// embedding program set one with [`mount_only_workspace_subfolder`].
pub fn workspace_subfolder() -> Option<&'static str> {
    WORKSPACE_SUBFOLDER.get().map(String::as_str)
}

fn is_one_normal_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    let first_is_normal = matches!(components.next(), Some(Component::Normal(_)));
    first_is_normal && components.next().is_none() && !name.contains(['/', '\\'])
}

/// The host folder to mount, given the canonical workspace: the workspace
/// itself, or its subfolder when one is set.
pub(crate) fn mount_source(workspace: PathBuf) -> Result<PathBuf> {
    match workspace_subfolder() {
        Some(name) => subfolder_mount_source(&workspace, name),
        None => Ok(workspace),
    }
}

/// The container path the mount source is mounted at.
pub(crate) fn mount_target() -> String {
    workspace_subfolder().map_or_else(
        || CONTAINER_WORKSPACE.to_owned(),
        |name| format!("{CONTAINER_WORKSPACE}/{name}"),
    )
}

fn subfolder_mount_source(workspace: &Path, name: &str) -> Result<PathBuf> {
    let path = workspace.join(name);
    let resolved = match path.canonicalize() {
        Ok(resolved) => resolved,
        Err(error) => bail!(
            "Docker workspace subfolder {} cannot be resolved: {error}",
            path.display()
        ),
    };
    if !resolved.is_dir() {
        bail!(
            "Docker workspace subfolder {} is not a directory",
            path.display()
        );
    }
    if resolved != path {
        bail!(
            "Docker workspace subfolder {} resolves to {}; it must be a real folder directly inside the workspace {}",
            path.display(),
            resolved.display(),
            workspace.display()
        );
    }
    Ok(resolved)
}
