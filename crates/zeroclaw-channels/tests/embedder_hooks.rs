//! A channel agent's chat turns use the embedder hooks installed with
//! `install_embedder_hooks`, whether or not `[hooks]` is enabled.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use zeroclaw_channels::orchestrator::chat_hooks::chat_hook_runner;
use zeroclaw_config::schema::Config;
use zeroclaw_runtime::hooks::{EmbedderHooks, HookHandler, HookResult, install_embedder_hooks};

/// Counts the tool calls it sees.
struct CountingHook(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl HookHandler for CountingHook {
    fn name(&self) -> &str {
        "counting"
    }

    async fn before_tool_call(&self, name: String, args: Value) -> HookResult<(String, Value)> {
        self.0.fetch_add(1, Ordering::SeqCst);
        HookResult::Continue((name, args))
    }
}

#[tokio::test]
async fn chat_hook_runner_holds_installed_hooks_with_hooks_off() {
    let mut config = Config::default();
    config.hooks.enabled = false;
    assert!(
        chat_hook_runner(&config).is_none(),
        "no runner without `[hooks]` or embedder hooks"
    );

    let calls = Arc::new(AtomicUsize::new(0));
    install_embedder_hooks(EmbedderHooks::new().with(Arc::new(CountingHook(Arc::clone(&calls)))));
    let runner = chat_hook_runner(&config).expect("the embedder hooks make a runner");
    let result = runner.run_before_tool_call("shell".into(), json!({})).await;

    assert!(matches!(result, HookResult::Continue(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
