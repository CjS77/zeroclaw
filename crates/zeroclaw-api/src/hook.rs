use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

use crate::channel::ChannelMessage;
use crate::ingress::IngressContext;
use crate::model_provider::{ChatMessage, ChatResponse};
use crate::tool::ToolResult;

/// The turn a tool call belongs to, as the runtime stamped it when the turn
/// started.
///
/// Every value is runtime-owned: none is taken from the model's output or the
/// tool call's arguments, so a hook can base policy on it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HookTurnContext {
    /// The turn's id, the `trace_id` on the turn's log records.
    pub turn_id: String,
    /// The alias of the agent whose policy and tools the turn runs with.
    pub agent_alias: Option<String>,
    /// The channel the turn runs on, as the runtime names it in its logs, such
    /// as `"cli"` for a CLI or scheduled turn.
    pub channel: String,
    /// Where the channel delivers the turn's replies, such as a chat room.
    pub reply_target: Option<String>,
    /// The turn's ingress envelope: its sender, origin and, for an internally
    /// driven turn, the principal that started it.
    pub ingress: IngressContext,
}

impl HookTurnContext {
    /// Build the context of one turn. `agent_alias` names the agent the turn
    /// runs as; `reply_target` is where the channel delivers its replies.
    pub fn new(
        turn_id: impl Into<String>,
        agent_alias: Option<String>,
        channel: impl Into<String>,
        reply_target: Option<String>,
        ingress: IngressContext,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            agent_alias,
            channel: channel.into(),
            reply_target,
            ingress,
        }
    }
}

/// Opaque runtime context for one tool-call hook phase.
///
/// Correlated contexts carry one stable identity across the before and after
/// phases. Uncorrelated contexts are phase-local and must not be used to pair
/// callbacks or retain attribution-sensitive state.
///
/// Equality and hashing follow the identity alone, so a context can key a
/// map whether or not it carries its turn.
#[derive(Debug, Clone)]
pub struct ToolCallHookContext {
    invocation_id: String,
    correlated: bool,
    turn: Option<HookTurnContext>,
}

impl PartialEq for ToolCallHookContext {
    fn eq(&self, other: &Self) -> bool {
        (&self.invocation_id, self.correlated) == (&other.invocation_id, other.correlated)
    }
}

impl Eq for ToolCallHookContext {}

impl std::hash::Hash for ToolCallHookContext {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.invocation_id.hash(state);
        self.correlated.hash(state);
    }
}

impl ToolCallHookContext {
    /// Create a correlated context.
    ///
    /// The caller must provide a process-unique identity and reuse it for both
    /// phases of the same tool invocation.
    pub fn new(invocation_id: impl Into<String>) -> Self {
        Self {
            invocation_id: invocation_id.into(),
            correlated: true,
            turn: None,
        }
    }

    /// Create a phase-local context when cross-phase identity is unavailable.
    pub fn uncorrelated(invocation_id: impl Into<String>) -> Self {
        Self {
            invocation_id: invocation_id.into(),
            correlated: false,
            turn: None,
        }
    }

    /// Attach the turn the call belongs to.
    #[must_use]
    pub fn with_turn(mut self, turn: HookTurnContext) -> Self {
        self.turn = Some(turn);
        self
    }

    /// The turn the call belongs to. `None` when the caller ran the hook
    /// outside a turn, such as through the legacy context-free entry points.
    pub fn turn(&self) -> Option<&HookTurnContext> {
        self.turn.as_ref()
    }

    /// Return the opaque phase or invocation identity.
    pub fn invocation_id(&self) -> &str {
        &self.invocation_id
    }

    /// Whether this identity is stable across the before and after phases.
    pub fn is_correlated(&self) -> bool {
        self.correlated
    }
}

/// Result of a modifying hook — continue with (possibly modified) data, or cancel.
#[derive(Debug, Clone)]
pub enum HookResult<T> {
    Continue(T),
    Cancel(String),
}

impl<T> HookResult<T> {
    pub fn is_cancel(&self) -> bool {
        matches!(self, HookResult::Cancel(_))
    }
}

/// What a before-tool-call hook decides about one call.
///
/// `Continue` and `Cancel` mean what they mean for [`HookResult`]. `Ask` and
/// `Approve` also let the call run, possibly with a changed name or
/// arguments, but take over the risk profile's decision on whether an
/// operator must approve it first. A runtime that meets a variant it does not
/// know cancels the call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallDecision {
    /// Run the call. The risk profile decides whether to prompt.
    Continue(String, Value),
    /// Don't run the call. The reason becomes the call's result.
    Cancel(String),
    /// Run the call only if an operator approves it now: prompt whatever the
    /// risk profile says, and ignore the session's "Always" answers for this
    /// call, as for a tool the profile lists in `always_ask`; read-only mode
    /// still blocks the call if the operator approves it. When no operator can
    /// be asked, as in a scheduled turn, the call is not run, and the reason is
    /// in its result. An operator's "Always" answer still adds the tool to the
    /// session's list, so later calls the hook does not ask about skip the
    /// prompt.
    Ask {
        name: String,
        args: Value,
        reason: String,
    },
    /// Run the call without the prompt the risk profile would show, as if the
    /// operator had approved it. For a command tool such as `shell` that
    /// includes the command-level approval that lets it run higher-risk
    /// commands. Grants nothing where the profile would not prompt, and a tool
    /// the profile lists in `always_ask` is still prompted for.
    Approve {
        name: String,
        args: Value,
        reason: String,
    },
}

impl From<HookResult<(String, Value)>> for ToolCallDecision {
    fn from(result: HookResult<(String, Value)>) -> Self {
        match result {
            HookResult::Continue((name, args)) => Self::Continue(name, args),
            HookResult::Cancel(reason) => Self::Cancel(reason),
        }
    }
}

/// Trait for hook handlers. All methods have default no-op implementations.
/// Implement only the events you care about.
#[async_trait]
pub trait HookHandler: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32 {
        0
    }

    // --- Void hooks (parallel, fire-and-forget) ---
    async fn on_gateway_start(&self, _host: &str, _port: u16) {}
    async fn on_gateway_stop(&self) {}
    async fn on_session_start(&self, _session_id: &str, _channel: &str) {}
    async fn on_session_end(&self, _session_id: &str, _channel: &str) {}
    async fn on_llm_input(&self, _messages: &[ChatMessage], _model: &str) {}
    async fn on_llm_output(&self, _response: &ChatResponse) {}
    async fn on_after_tool_call(&self, _tool: &str, _result: &ToolResult, _duration: Duration) {}
    /// Observe tool completion with explicit correlation context.
    ///
    /// When `context.is_correlated()` is false, handlers must not use its
    /// phase-local identity to pair callbacks or retain attributed state.
    async fn on_after_tool_call_with_context(
        &self,
        _context: &ToolCallHookContext,
        tool: &str,
        result: &ToolResult,
        duration: Duration,
    ) {
        self.on_after_tool_call(tool, result, duration).await;
    }
    /// Observe tool completion with explicit correlation context and the
    /// arguments that were actually dispatched.
    ///
    /// Runners that know the final prepared arguments dispatch this variant
    /// instead of [`HookHandler::on_after_tool_call_with_context`]; the
    /// default delegates there, dropping the arguments. Handlers that audit
    /// arguments should override this variant so their export reflects what
    /// was dispatched — and so they never need to retain arguments between
    /// the before and after phases.
    async fn on_after_tool_call_with_context_and_args(
        &self,
        context: &ToolCallHookContext,
        tool: &str,
        _args: &Value,
        result: &ToolResult,
        duration: Duration,
    ) {
        self.on_after_tool_call_with_context(context, tool, result, duration)
            .await;
    }
    /// Observe that a tool-call context will never reach its after phase.
    ///
    /// Fired exactly once per correlated context whose before phase ran, when
    /// the call cannot complete: the before hook chain cancelled it, approval
    /// denied or replaced it, duplicate suppression skipped it, or execution
    /// failed or was interrupted before post-execution handling. Abandonment
    /// is NOT execution completion — handlers must never synthesize a tool
    /// result from it.
    ///
    /// Best-effort by nature: the runner dispatches abandonment on every
    /// awaited non-completion path, but a turn future dropped outright (an
    /// outer timeout, outer cancellation, or hard task abort) cannot run any
    /// async callback, abandonment included. Handlers must therefore avoid
    /// retaining safety-critical state between the before and after phases in
    /// the first place, and treat abandonment as a best-effort release signal
    /// rather than a guarantee.
    async fn on_tool_call_abandoned(&self, _context: &ToolCallHookContext, _tool: &str) {}
    async fn on_message_sent(&self, _channel: &str, _recipient: &str, _content: &str) {}
    async fn on_heartbeat_tick(&self) {}

    // --- Modifying hooks (sequential by priority, can cancel) ---
    async fn before_model_resolve(
        &self,
        model_provider: String,
        model: String,
    ) -> HookResult<(String, String)> {
        HookResult::Continue((model_provider, model))
    }

    async fn before_prompt_build(&self, prompt: String) -> HookResult<String> {
        HookResult::Continue(prompt)
    }

    async fn before_llm_call(
        &self,
        _messages: &mut Vec<ChatMessage>,
        _model: &mut String,
    ) -> HookResult<()> {
        HookResult::Continue(())
    }

    async fn before_tool_call(&self, name: String, args: Value) -> HookResult<(String, Value)> {
        HookResult::Continue((name, args))
    }

    /// Inspect or modify a tool call with explicit correlation context.
    ///
    /// When `context.is_correlated()` is false, handlers must not retain state
    /// that assumes the after phase will receive the same identity.
    async fn before_tool_call_with_context(
        &self,
        _context: &ToolCallHookContext,
        name: String,
        args: Value,
    ) -> HookResult<(String, Value)> {
        self.before_tool_call(name, args).await
    }

    /// Decide on a tool call before it runs: continue, cancel, or take over
    /// whether an operator must approve it (see [`ToolCallDecision`]).
    ///
    /// The default delegates to
    /// [`HookHandler::before_tool_call_with_context`], so a hook that only
    /// continues or cancels needs nothing new. When several hooks decide on
    /// the same call, an `Ask` from any of them wins over an `Approve`, and an
    /// `Approve` lapses if a later hook changes the call's name or arguments.
    ///
    /// Only turns that carry hooks run this: a delegated sub-agent's turn runs
    /// without them. A hook that panics is skipped, as for the other before
    /// hooks, and the call goes ahead on the other hooks' decisions.
    async fn before_tool_call_decision(
        &self,
        context: &ToolCallHookContext,
        name: String,
        args: Value,
    ) -> ToolCallDecision {
        self.before_tool_call_with_context(context, name, args)
            .await
            .into()
    }

    async fn on_message_received(&self, message: ChannelMessage) -> HookResult<ChannelMessage> {
        HookResult::Continue(message)
    }

    async fn on_message_sending(
        &self,
        channel: String,
        recipient: String,
        content: String,
    ) -> HookResult<(String, String, String)> {
        HookResult::Continue((channel, recipient, content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHook {
        name: String,
        priority: i32,
    }

    impl TestHook {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                name: name.to_string(),
                priority,
            }
        }
    }

    #[async_trait]
    impl HookHandler for TestHook {
        fn name(&self) -> &str {
            &self.name
        }
        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[test]
    fn hook_result_is_cancel() {
        let ok: HookResult<String> = HookResult::Continue("hi".into());
        assert!(!ok.is_cancel());
        let cancel: HookResult<String> = HookResult::Cancel("blocked".into());
        assert!(cancel.is_cancel());
    }

    #[test]
    fn tool_call_context_carries_its_turn_only_when_attached() {
        let bare = ToolCallHookContext::new("turn-1:0:0");
        assert_eq!(bare.turn(), None);

        let turn = HookTurnContext::new(
            "turn-1",
            Some("assistant".into()),
            "matrix",
            Some("!room:example.org".into()),
            IngressContext::channel(),
        );
        let stamped = bare.clone().with_turn(turn.clone());
        assert_eq!(stamped.turn(), Some(&turn));
        assert_eq!(stamped, bare, "the turn is not part of the identity");
        assert_eq!(stamped.invocation_id(), "turn-1:0:0");
        assert!(stamped.is_correlated());
    }

    #[tokio::test]
    async fn default_decision_follows_the_context_hook() {
        struct Cancelling;
        #[async_trait]
        impl HookHandler for Cancelling {
            fn name(&self) -> &str {
                "cancelling"
            }
            async fn before_tool_call_with_context(
                &self,
                _context: &ToolCallHookContext,
                _name: String,
                _args: Value,
            ) -> HookResult<(String, Value)> {
                HookResult::Cancel("no".into())
            }
        }
        let context = ToolCallHookContext::new("turn-1:0:0");
        let args = serde_json::json!({"cmd": "ls"});
        assert_eq!(
            TestHook::new("pass", 0)
                .before_tool_call_decision(&context, "shell".into(), args.clone())
                .await,
            ToolCallDecision::Continue("shell".into(), args.clone())
        );
        assert_eq!(
            Cancelling
                .before_tool_call_decision(&context, "shell".into(), args)
                .await,
            ToolCallDecision::Cancel("no".into())
        );
    }

    #[test]
    fn default_priority_is_zero() {
        struct MinimalHook;
        #[async_trait]
        impl HookHandler for MinimalHook {
            fn name(&self) -> &str {
                "minimal"
            }
        }
        assert_eq!(MinimalHook.priority(), 0);
    }

    #[tokio::test]
    async fn default_modifying_hooks_pass_through() {
        let hook = TestHook::new("test", 0);
        match hook
            .before_tool_call("shell".into(), serde_json::json!({"cmd": "ls"}))
            .await
        {
            HookResult::Continue((name, _args)) => assert_eq!(name, "shell"),
            HookResult::Cancel(_) => panic!("should not cancel"),
        }
    }

    #[tokio::test]
    async fn context_aware_defaults_delegate_to_legacy_tool_hooks() {
        use std::sync::{Arc, Mutex};

        struct LegacyHook {
            after: Arc<Mutex<Option<(String, bool, Duration)>>>,
        }

        #[async_trait]
        impl HookHandler for LegacyHook {
            fn name(&self) -> &str {
                "legacy"
            }

            async fn before_tool_call(
                &self,
                name: String,
                args: Value,
            ) -> HookResult<(String, Value)> {
                HookResult::Continue((format!("{name}_legacy"), args))
            }

            async fn on_after_tool_call(
                &self,
                tool: &str,
                result: &ToolResult,
                duration: Duration,
            ) {
                *self.after.lock().unwrap() = Some((tool.to_string(), result.success, duration));
            }
        }

        let after = Arc::new(Mutex::new(None));
        let hook = LegacyHook {
            after: Arc::clone(&after),
        };
        let context = ToolCallHookContext::new("opaque-id");
        let args = serde_json::json!({"cmd": "ls"});

        let before = hook
            .before_tool_call_with_context(&context, "shell".into(), args.clone())
            .await;
        match before {
            HookResult::Continue((name, actual_args)) => {
                assert_eq!(name, "shell_legacy");
                assert_eq!(actual_args, args);
            }
            HookResult::Cancel(_) => panic!("legacy hook should continue"),
        }

        let result = ToolResult {
            success: true,
            output: "ok".into(),
            error: None,
        };
        let duration = Duration::from_millis(12);
        hook.on_after_tool_call_with_context(&context, "shell", &result, duration)
            .await;

        assert_eq!(
            *after.lock().unwrap(),
            Some(("shell".to_string(), true, duration))
        );
    }

    #[test]
    fn tool_call_context_marks_unavailable_correlation() {
        let exact = ToolCallHookContext::new("exact");
        let unavailable = ToolCallHookContext::uncorrelated("legacy");

        assert!(exact.is_correlated());
        assert!(!unavailable.is_correlated());
        assert_eq!(unavailable.invocation_id(), "legacy");
    }

    #[tokio::test]
    async fn args_carrying_after_variant_defaults_to_context_variant() {
        use std::sync::{Arc, Mutex};

        struct ContextOnlyHook {
            seen: Arc<Mutex<Option<(String, String, bool)>>>,
        }

        #[async_trait]
        impl HookHandler for ContextOnlyHook {
            fn name(&self) -> &str {
                "context-only"
            }

            async fn on_after_tool_call_with_context(
                &self,
                context: &ToolCallHookContext,
                tool: &str,
                result: &ToolResult,
                _duration: Duration,
            ) {
                *self.seen.lock().unwrap() = Some((
                    context.invocation_id().to_string(),
                    tool.to_string(),
                    result.success,
                ));
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let hook = ContextOnlyHook {
            seen: Arc::clone(&seen),
        };
        let result = ToolResult {
            success: true,
            output: "ok".into(),
            error: None,
        };
        hook.on_after_tool_call_with_context_and_args(
            &ToolCallHookContext::new("invocation"),
            "shell",
            &serde_json::json!({"command": "ls"}),
            &result,
            Duration::ZERO,
        )
        .await;
        assert_eq!(
            *seen.lock().unwrap(),
            Some(("invocation".to_string(), "shell".to_string(), true)),
            "the args-carrying default must land in the context-aware variant"
        );
    }

    #[tokio::test]
    async fn default_abandonment_hook_compiles_and_is_a_noop() {
        // A pre-existing handler shape: implements only `name`. It must keep
        // compiling unchanged and the default abandonment must be a no-op.
        struct MinimalHook;
        #[async_trait]
        impl HookHandler for MinimalHook {
            fn name(&self) -> &str {
                "minimal"
            }
        }

        let hook = MinimalHook;
        hook.on_tool_call_abandoned(&ToolCallHookContext::new("invocation"), "shell")
            .await;
        hook.on_tool_call_abandoned(&ToolCallHookContext::uncorrelated("legacy"), "shell")
            .await;
    }
}
