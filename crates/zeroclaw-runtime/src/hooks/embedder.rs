//! Hooks that a binary embedding ZeroClaw adds to the agent turns it starts.
//!
//! A binary that embeds the runtime, instead of running `zeroclaw daemon`, hands
//! [`EmbedderHooks`] to the channel runner (chat turns) and to the cron scheduler
//! (cron turns, through [`crate::agent::loop_::AgentRunOverrides`]). They join each
//! turn's [`HookRunner`] next to the configured hooks, and apply whether or not
//! `[hooks]` is enabled.
//!
//! They do not reach a sub-agent (`delegate`, `spawn_subagent`) or a manual cron run
//! (the `cron_run` tool, the gateway, RPC): those paths are not handed them.

use std::sync::Arc;

use super::{HookHandler, HookRunner};

/// The hooks an embedding binary adds to every agent turn it starts. The default adds none.
#[derive(Clone, Default)]
pub struct EmbedderHooks {
    hooks: Vec<Arc<dyn HookHandler>>,
}

impl EmbedderHooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `hook` to every turn's hook runner.
    #[must_use]
    pub fn with(mut self, hook: Arc<dyn HookHandler>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// `configured` with the embedder's hooks added. A fresh runner when nothing is
    /// configured but the embedder has hooks; `None` when neither has any.
    pub fn hook_runner(&self, configured: Option<HookRunner>) -> Option<HookRunner> {
        if self.hooks.is_empty() {
            return configured;
        }
        let mut runner = configured.unwrap_or_default();
        self.hooks
            .iter()
            .for_each(|hook| runner.register_shared(Arc::clone(hook)));
        Some(runner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NamedHook(&'static str);

    #[async_trait::async_trait]
    impl HookHandler for NamedHook {
        fn name(&self) -> &str {
            self.0
        }
    }

    #[test]
    fn hook_runner_is_none_when_nothing_has_hooks() {
        assert!(EmbedderHooks::new().hook_runner(None).is_none());
    }

    #[test]
    fn hook_runner_keeps_configured_hooks_without_embedder_hooks() {
        let mut configured = HookRunner::new();
        configured.register(Box::new(NamedHook("configured")));
        let runner = EmbedderHooks::new().hook_runner(Some(configured)).unwrap();
        assert_eq!(runner.handler_names(), ["configured"]);
    }

    #[test]
    fn hook_runner_adds_embedder_hooks_next_to_configured_ones() {
        let mut configured = HookRunner::new();
        configured.register(Box::new(NamedHook("configured")));
        let embedder = EmbedderHooks::new().with(Arc::new(NamedHook("embedder")));
        let runner = embedder.hook_runner(Some(configured)).unwrap();
        assert_eq!(runner.handler_names(), ["configured", "embedder"]);
        let alone = embedder.hook_runner(None).unwrap();
        assert_eq!(alone.handler_names(), ["embedder"]);
    }

    #[test]
    fn one_hook_is_shared_by_every_runner() {
        let hook: Arc<dyn HookHandler> = Arc::new(NamedHook("embedder"));
        let embedder = EmbedderHooks::new().with(Arc::clone(&hook));
        let _first = embedder.hook_runner(None).unwrap();
        let _second = embedder.hook_runner(None).unwrap();
        assert_eq!(
            Arc::strong_count(&hook),
            4,
            "the caller, the bundle and both runners"
        );
    }
}
