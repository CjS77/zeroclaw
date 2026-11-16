//! Hooks that a binary embedding ZeroClaw installs once, at start-up, for every agent turn.
//!
//! A binary that embeds the runtime, instead of running `zeroclaw daemon`, calls
//! [`install_embedder_hooks`] before it starts the channel runner or the scheduler. The
//! installed hooks join the hook runner of every chat turn, next to the configured hooks and
//! whether or not `[hooks]` is enabled, and of every turn `agent::run` runs: scheduled jobs,
//! sub-agents and manual cron runs.

use std::sync::{Arc, PoisonError, RwLock};

use super::{HookHandler, HookRunner};

/// The hooks an embedding binary adds to every agent turn. The default adds none.
#[derive(Clone, Default)]
pub struct EmbedderHooks {
    hooks: Vec<Arc<dyn HookHandler>>,
}

impl EmbedderHooks {
    /// An empty set of hooks.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `hook` to every turn's hook runner.
    #[must_use]
    pub fn with(mut self, hook: Arc<dyn HookHandler>) -> Self {
        self.hooks.push(hook);
        self
    }
}

/// The process-wide slot. A lock rather than a once-only cell, so tests can reinstall.
static INSTALLED: RwLock<EmbedderHooks> = RwLock::new(EmbedderHooks { hooks: Vec::new() });

/// Install `hooks` for every agent turn this process starts from now on, replacing any
/// installed before. Turns already running keep the hooks they started with.
pub fn install_embedder_hooks(hooks: EmbedderHooks) {
    *INSTALLED.write().unwrap_or_else(PoisonError::into_inner) = hooks;
}

/// `configured` with the installed embedder hooks added. A fresh runner when nothing is
/// configured but hooks are installed; `None` when neither has any.
pub fn with_embedder_hooks(configured: Option<HookRunner>) -> Option<HookRunner> {
    let installed = INSTALLED
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if installed.hooks.is_empty() {
        return configured;
    }
    let mut runner = configured.unwrap_or_default();
    installed
        .hooks
        .into_iter()
        .for_each(|hook| runner.register_shared(hook));
    Some(runner)
}
