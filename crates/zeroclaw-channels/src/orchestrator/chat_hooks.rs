//! The hook runner a channel agent's chat turns use.

use std::sync::Arc;

use zeroclaw_config::schema::Config;
use zeroclaw_runtime::hooks::{HookRunner, with_embedder_hooks};

/// The hook runner for a channel agent's chat turns: the `[hooks]` handlers when `[hooks]` is
/// enabled, plus the embedder hooks installed with
/// [`install_embedder_hooks`](zeroclaw_runtime::hooks::install_embedder_hooks), whether or not
/// it is. `None` when there are neither.
pub fn chat_hook_runner(config: &Config) -> Option<Arc<HookRunner>> {
    let configured = config
        .hooks
        .enabled
        .then(|| HookRunner::from_root_config(config));
    with_embedder_hooks(configured).map(Arc::new)
}
