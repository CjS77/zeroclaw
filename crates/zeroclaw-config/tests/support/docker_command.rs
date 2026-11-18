//! Builds Docker runtime shell commands against a fake `docker` launcher on a
//! temporary PATH, and reads back the arguments they would pass.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use zeroclaw_config::platform::{DockerRuntime, RuntimeAdapter};
use zeroclaw_config::schema::DockerRuntimeConfig;

/// The arguments the default Docker runtime passes to `docker` for a shell
/// command run in `workspace`.
pub fn docker_args(workspace: &Path) -> anyhow::Result<Vec<String>> {
    let bin = tempfile::tempdir()?;
    let launcher = bin.path().join("docker");
    std::fs::write(&launcher, "#!/bin/sh\nexit 1\n")?;
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755))?;

    let runtime = DockerRuntime::new(DockerRuntimeConfig::default());
    let command = runtime.build_shell_command_with_effective_path(
        "true",
        workspace,
        Some(bin.path().as_os_str()),
    )?;
    Ok(command
        .as_std()
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect())
}

/// The value that follows `flag` in `args`, if the flag is there.
pub fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
}

/// A temporary workspace holding an instruction file, as a real one does.
pub fn workspace() -> tempfile::TempDir {
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("AGENTS.md"), "instructions").unwrap();
    workspace
}
