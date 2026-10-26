//! What a tool-call hook sees and decides at the turn-loop boundary: real
//! turns through `run_tool_call_loop`, with a scripted model.

use super::*;
use crate::observability::NoopObserver;
use std::sync::Mutex;
use zeroclaw_api::channel::{
    Channel, ChannelApprovalRequest, ChannelApprovalResponse, ChannelMessage, SendMessage,
};
use zeroclaw_api::hook::{HookTurnContext, ToolCallDecision, ToolCallHookContext};
use zeroclaw_api::ingress::IngressContext;
use zeroclaw_providers::{ChatResponse, ToolCall};

/// Each hook phase that ran, with the turn context it received.
type SeenTurns = Arc<Mutex<Vec<(&'static str, Option<HookTurnContext>)>>>;

/// Records the turn context each tool-call hook phase receives.
#[derive(Default)]
struct TurnRecorder {
    seen: SeenTurns,
}

#[async_trait::async_trait]
impl crate::hooks::HookHandler for TurnRecorder {
    fn name(&self) -> &str {
        "turn-recorder"
    }

    async fn before_tool_call_with_context(
        &self,
        context: &ToolCallHookContext,
        name: String,
        args: serde_json::Value,
    ) -> crate::hooks::HookResult<(String, serde_json::Value)> {
        self.seen
            .lock()
            .expect("recorder lock")
            .push(("before", context.turn().cloned()));
        crate::hooks::HookResult::Continue((name, args))
    }

    async fn on_after_tool_call_with_context(
        &self,
        context: &ToolCallHookContext,
        _tool: &str,
        _result: &zeroclaw_api::tool::ToolResult,
        _duration: std::time::Duration,
    ) {
        self.seen
            .lock()
            .expect("recorder lock")
            .push(("after", context.turn().cloned()));
    }
}

/// Hands out its scripted responses in order, then a plain text answer.
struct ScriptedModel {
    responses: Mutex<Vec<ChatResponse>>,
}

impl ScriptedModel {
    fn calling(tools: &[&str]) -> Self {
        let tool_calls = tools
            .iter()
            .enumerate()
            .map(|(i, name)| ToolCall {
                id: format!("call-{i}"),
                name: (*name).to_string(),
                arguments: serde_json::json!({"n": i}).to_string(),
                extra_content: None,
            })
            .collect();
        Self {
            responses: Mutex::new(vec![ChatResponse {
                text: None,
                tool_calls,
                usage: None,
                reasoning_content: None,
            }]),
        }
    }
}

impl zeroclaw_api::attribution::Attributable for ScriptedModel {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Provider(zeroclaw_api::attribution::ProviderKind::Model(
            zeroclaw_api::attribution::ModelProviderKind::Custom,
        ))
    }
    fn alias(&self) -> &str {
        "scripted"
    }
}

#[async_trait::async_trait]
impl ModelProvider for ScriptedModel {
    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_system(
        &self,
        _system_prompt: Option<&str>,
        _message: &str,
        _model: &str,
        _temperature: Option<f64>,
    ) -> anyhow::Result<String> {
        Ok("done".into())
    }

    async fn chat(
        &self,
        _request: zeroclaw_api::model_provider::ChatRequest<'_>,
        _model: &str,
        _temperature: Option<f64>,
    ) -> anyhow::Result<ChatResponse> {
        let mut queue = self.responses.lock().expect("script lock");
        if queue.is_empty() {
            return Ok(ChatResponse {
                text: Some("done".to_string()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            });
        }
        Ok(queue.remove(0))
    }
}

/// A tool that succeeds and counts its runs.
struct CountingTool {
    name: &'static str,
    runs: Arc<std::sync::atomic::AtomicUsize>,
}

impl zeroclaw_api::attribution::Attributable for CountingTool {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Tool(zeroclaw_api::attribution::ToolKind::Plugin)
    }
    fn alias(&self) -> &str {
        self.name
    }
}

#[async_trait::async_trait]
impl crate::tools::Tool for CountingTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "counts its runs"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<crate::tools::ToolResult> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(crate::tools::ToolResult::ok("ran"))
    }
}

/// The turn under test: who it is for and what may approve its calls.
struct Turn<'a> {
    channel_name: &'a str,
    reply_target: Option<&'a str>,
    agent_alias: Option<&'a str>,
    ingress: IngressContext,
    approval: Option<&'a crate::approval::ApprovalManager>,
    channel: Option<&'a dyn zeroclaw_api::channel::Channel>,
}

impl Default for Turn<'_> {
    fn default() -> Self {
        Self {
            channel_name: "cli",
            reply_target: None,
            agent_alias: None,
            ingress: IngressContext::interactive(),
            approval: None,
            channel: None,
        }
    }
}

const TURN_ID: &str = "turn-hook-tests";

/// Run one turn; returns its result and the history it left.
async fn run_turn(
    turn: Turn<'_>,
    model: &ScriptedModel,
    tools: Vec<Box<dyn crate::tools::Tool>>,
    hooks: &crate::hooks::HookRunner,
) -> (anyhow::Result<String>, Vec<ChatMessage>) {
    let registry = crate::tools::scoped::ScopedToolRegistry::from_raw_for_test(tools);
    let observer = NoopObserver;
    let pacing = zeroclaw_config::schema::PacingConfig::default();
    let mut history = vec![ChatMessage::user("go")];
    let mut crumb_present = false;
    let mut injected_preamble = None;
    let result = run_tool_call_loop(ToolLoop {
        exec: ResolvedAgentExecution {
            model_access: ResolvedModelAccess {
                model_provider: model,
                provider_name: "scripted",
                model: "scripted-model",
                dispatch_model: "scripted-model",
                temperature: Some(0.0),
            },
            tools_registry: &registry,
            observer: &observer,
            silent: true,
            approval: turn.approval,
            multimodal_config: &zeroclaw_config::schema::MultimodalConfig::default(),
            config: None,
            max_tool_iterations: 3,
            hooks: Some(hooks),
            excluded_tools: &[],
            dedup_exempt_tools: &[],
            activated_tools: None,
            model_switch_callback: None,
            pacing: &pacing,
            strict_tool_parsing: false,
            parallel_tools: false,
            max_tool_result_chars: 0,
            context_limits: zeroclaw_config::schema::ResolvedContextLimits::legacy_fallback(0),
            context_limits_resolver: None,
            receipt_generator: None,
            knobs: &LoopKnobs::default(),
            security: None,
        },
        history: &mut history,
        history_has_trim_breadcrumb: &mut crumb_present,
        injected_memory_preamble: &mut injected_preamble,
        channel_name: turn.channel_name,
        channel_reply_target: turn.reply_target,
        cancellation_token: None,
        on_delta: None,
        shared_budget: None,
        channel: turn.channel,
        collected_receipts: None,
        event_tx: None,
        steering: None,
        new_messages_out: None,
        image_cache: None,
        memory: None,
        ingress: turn.ingress,
        agent_alias: turn.agent_alias,
        parent_agent_alias: None,
        turn_id: TURN_ID,
        served_route_sink: None,
        sop_reassembly: None,
    })
    .await;
    (result, history)
}

fn counting_tool(
    name: &'static str,
) -> (
    Box<dyn crate::tools::Tool>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    (
        Box::new(CountingTool {
            name,
            runs: Arc::clone(&runs),
        }),
        runs,
    )
}

fn owner_ingress() -> IngressContext {
    let mut ingress = IngressContext::channel();
    ingress.sender = Some("@owner:example.org".to_string());
    ingress
}

#[tokio::test]
async fn tool_call_hooks_receive_the_turn_they_belong_to() {
    let recorder = TurnRecorder::default();
    let seen = Arc::clone(&recorder.seen);
    let mut hooks = crate::hooks::HookRunner::new();
    hooks.register(Box::new(recorder));
    let (tool, runs) = counting_tool("lookup");

    run_turn(
        Turn {
            channel_name: "matrix",
            reply_target: Some("!room:example.org"),
            agent_alias: Some("assistant"),
            ingress: owner_ingress(),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &hooks,
    )
    .await
    .0
    .expect("turn completes");

    let expected = HookTurnContext::new(
        TURN_ID,
        Some("assistant".to_string()),
        "matrix",
        Some("!room:example.org".to_string()),
        owner_ingress(),
    );
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        *seen.lock().expect("recorder lock"),
        vec![
            ("before", Some(expected.clone())),
            ("after", Some(expected))
        ]
    );
}

#[tokio::test]
async fn a_scheduled_turn_names_its_job_to_the_hook() {
    let recorder = TurnRecorder::default();
    let seen = Arc::clone(&recorder.seen);
    let mut hooks = crate::hooks::HookRunner::new();
    hooks.register(Box::new(recorder));
    let (tool, _runs) = counting_tool("lookup");

    run_turn(
        Turn {
            ingress: IngressContext::cron_job("job-7", Some("nightly".to_string())),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &hooks,
    )
    .await
    .0
    .expect("turn completes");

    let seen = seen.lock().expect("recorder lock");
    let turn = seen[0]
        .1
        .as_ref()
        .expect("the before phase carries the turn");
    assert_eq!(
        turn.ingress.internal_principal,
        Some(zeroclaw_api::ingress::InternalPrincipal::Cron {
            job_id: "job-7".to_string(),
            job_name: Some("nightly".to_string()),
        })
    );
}

const REASON: &str = "the recipient is not in the owner's contacts";

/// Asks an operator about, or approves, every call whose arguments are
/// `matching` (every call when `None`).
struct DecidingHook {
    matching: Option<serde_json::Value>,
    approve: bool,
}

#[async_trait::async_trait]
impl crate::hooks::HookHandler for DecidingHook {
    fn name(&self) -> &str {
        "deciding"
    }

    async fn before_tool_call_decision(
        &self,
        _context: &ToolCallHookContext,
        name: String,
        args: serde_json::Value,
    ) -> ToolCallDecision {
        let reason = REASON.to_string();
        match &self.matching {
            Some(matching) if *matching != args => ToolCallDecision::Continue(name, args),
            _ if self.approve => ToolCallDecision::Approve { name, args, reason },
            _ => ToolCallDecision::Ask { name, args, reason },
        }
    }
}

fn deciding_hooks(matching: Option<serde_json::Value>, approve: bool) -> crate::hooks::HookRunner {
    let mut hooks = crate::hooks::HookRunner::new();
    hooks.register(Box::new(DecidingHook { matching, approve }));
    hooks
}

/// Answers approval prompts from a script, and records every prompt.
struct ScriptedApprover {
    answers: Mutex<Vec<ChannelApprovalResponse>>,
    prompts: Arc<Mutex<Vec<ChannelApprovalRequest>>>,
    unreachable: bool,
}

impl ScriptedApprover {
    fn answering(answers: Vec<ChannelApprovalResponse>) -> Self {
        Self {
            answers: Mutex::new(answers),
            prompts: Arc::default(),
            unreachable: false,
        }
    }
}

impl zeroclaw_api::attribution::Attributable for ScriptedApprover {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::AcpChannel)
    }
    fn alias(&self) -> &str {
        "scripted-approver"
    }
}

#[async_trait::async_trait]
impl Channel for ScriptedApprover {
    fn name(&self) -> &str {
        "scripted-approver"
    }

    async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn request_approval(
        &self,
        _recipient: &str,
        request: &ChannelApprovalRequest,
    ) -> anyhow::Result<Option<ChannelApprovalResponse>> {
        self.prompts
            .lock()
            .expect("prompt lock")
            .push(request.clone());
        anyhow::ensure!(!self.unreachable, "approver unreachable");
        let mut answers = self.answers.lock().expect("answer lock");
        Ok((!answers.is_empty()).then(|| answers.remove(0)))
    }
}

fn supervised_profile() -> zeroclaw_config::schema::RiskProfileConfig {
    zeroclaw_config::schema::RiskProfileConfig {
        level: crate::security::AutonomyLevel::Supervised,
        auto_approve: Vec::new(),
        always_ask: Vec::new(),
        ..zeroclaw_config::schema::RiskProfileConfig::default()
    }
}

fn mentions_reason(history: &[ChatMessage]) -> bool {
    history
        .iter()
        .any(|message| message.content.contains(REASON))
}

/// The model calls `lookup` twice in one batch; the owner answers "Always"
/// to the first prompt. Returns the prompts the owner saw and how often the
/// tool ran.
async fn always_then_second_lookup(hooks: &crate::hooks::HookRunner) -> (usize, usize) {
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover::answering(vec![
        ChannelApprovalResponse::AlwaysApprove,
        ChannelApprovalResponse::Approve,
    ]);
    let prompts = Arc::clone(&approver.prompts);
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            channel_name: "matrix",
            reply_target: Some("!room:example.org"),
            ingress: owner_ingress(),
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup", "lookup"]),
        vec![tool],
        hooks,
    )
    .await;
    result.expect("turn completes");
    let prompted = prompts.lock().expect("prompt lock").len();
    (prompted, runs.load(std::sync::atomic::Ordering::SeqCst))
}

#[tokio::test]
async fn an_always_answer_skips_later_prompts_for_the_tool() {
    let (prompted, ran) = always_then_second_lookup(&crate::hooks::HookRunner::new()).await;
    assert_eq!((prompted, ran), (1, 2));
}

#[tokio::test]
async fn a_hook_ask_prompts_despite_an_earlier_always_answer() {
    let hooks = deciding_hooks(Some(serde_json::json!({"n": 1})), false);
    let (prompted, ran) = always_then_second_lookup(&hooks).await;
    assert_eq!((prompted, ran), (2, 2));
}

#[tokio::test]
async fn a_hook_ask_in_a_scheduled_turn_does_not_run_the_call() {
    let (tool, runs) = counting_tool("lookup");

    let (result, history) = run_turn(
        Turn {
            ingress: IngressContext::cron_job("job-7", None),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &deciding_hooks(None, false),
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(
        mentions_reason(&history),
        "the call's result gives the hook's reason"
    );
}

#[tokio::test]
async fn a_hook_ask_with_no_channel_to_prompt_does_not_run_the_call() {
    let profile = zeroclaw_config::schema::RiskProfileConfig {
        auto_approve: vec!["lookup".to_string()],
        ..supervised_profile()
    };
    let approval = crate::approval::ApprovalManager::for_non_interactive(&profile);
    let (tool, runs) = counting_tool("lookup");

    let (result, history) = run_turn(
        Turn {
            approval: Some(&approval),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &deciding_hooks(None, false),
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(
        mentions_reason(&history),
        "the call's result gives the hook's reason"
    );
}

#[tokio::test]
async fn a_hook_approval_skips_the_profile_prompt() {
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover::answering(Vec::new());
    let prompts = Arc::clone(&approver.prompts);
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &deciding_hooks(None, true),
    )
    .await;

    result.expect("turn completes");
    assert!(prompts.lock().expect("prompt lock").is_empty());
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// A shell call in a turn whose approval channel reports no approval
/// support, which hands shell back to its own policy checks. Returns how often
/// the call ran and whether its result gives the hook's reason.
async fn shell_without_channel_prompts(hooks: &crate::hooks::HookRunner) -> (usize, bool) {
    let approval =
        crate::approval::ApprovalManager::for_non_interactive_backchannel(&supervised_profile());
    let approver = ScriptedApprover::answering(Vec::new());
    let (tool, runs) = counting_tool("shell");

    let (result, history) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["shell"]),
        vec![tool],
        hooks,
    )
    .await;

    result.expect("turn completes");
    (
        runs.load(std::sync::atomic::Ordering::SeqCst),
        mentions_reason(&history),
    )
}

#[tokio::test]
async fn shell_falls_back_to_its_own_checks_when_the_channel_cannot_prompt() {
    let ran = shell_without_channel_prompts(&crate::hooks::HookRunner::new()).await;
    assert_eq!(ran, (1, false));
}

#[tokio::test]
async fn a_hook_ask_on_shell_does_not_fall_back_when_the_channel_cannot_prompt() {
    let ran = shell_without_channel_prompts(&deciding_hooks(None, false)).await;
    assert_eq!(ran, (0, true));
}

#[tokio::test]
async fn a_hook_ask_prompts_in_read_only_mode() {
    let profile = zeroclaw_config::schema::RiskProfileConfig {
        level: crate::security::AutonomyLevel::ReadOnly,
        ..supervised_profile()
    };
    let approval = crate::approval::ApprovalManager::for_non_interactive(&profile);
    let (tool, runs) = counting_tool("lookup");

    let (result, history) = run_turn(
        Turn {
            approval: Some(&approval),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &deciding_hooks(None, false),
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(
        mentions_reason(&history),
        "the call's result gives the hook's reason"
    );
}

#[tokio::test]
async fn the_prompt_for_a_hook_ask_carries_the_reason() {
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover::answering(vec![
        ChannelApprovalResponse::Approve,
        ChannelApprovalResponse::Approve,
    ]);
    let prompts = Arc::clone(&approver.prompts);
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup", "lookup"]),
        vec![tool],
        &deciding_hooks(Some(serde_json::json!({"n": 1})), false),
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    let reasons: Vec<_> = prompts
        .lock()
        .expect("prompt lock")
        .iter()
        .map(|prompt| prompt.reason.clone())
        .collect();
    assert_eq!(reasons, vec![None, Some(REASON.to_string())]);
}

/// One approval answer a hook heard: the call's id, whether its turn came
/// with it, the answer and who gave it.
type HeardAnswer = (
    String,
    bool,
    ChannelApprovalResponse,
    zeroclaw_api::channel::ApprovalSource,
);

/// Records every approval answer it hears.
#[derive(Default)]
struct AnswerRecorder {
    heard: Arc<Mutex<Vec<HeardAnswer>>>,
}

#[async_trait::async_trait]
impl crate::hooks::HookHandler for AnswerRecorder {
    fn name(&self) -> &str {
        "answer-recorder"
    }

    async fn on_tool_call_approval(
        &self,
        context: &ToolCallHookContext,
        _tool: &str,
        response: &zeroclaw_api::channel::AttributedApprovalResponse,
    ) {
        self.heard.lock().expect("answer lock").push((
            context.invocation_id().to_string(),
            context.turn().is_some(),
            response.response.clone(),
            response.source,
        ));
    }
}

fn answer_recording_hooks(ask: bool) -> (crate::hooks::HookRunner, Arc<Mutex<Vec<HeardAnswer>>>) {
    let recorder = AnswerRecorder::default();
    let heard = Arc::clone(&recorder.heard);
    let mut hooks = crate::hooks::HookRunner::new();
    hooks.register(Box::new(recorder));
    if ask {
        hooks.register(Box::new(DecidingHook {
            matching: None,
            approve: false,
        }));
    }
    (hooks, heard)
}

#[tokio::test]
async fn hooks_hear_each_owner_answer_with_the_calls_context() {
    use zeroclaw_api::channel::ApprovalSource::Operator;
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover::answering(vec![
        ChannelApprovalResponse::Approve,
        ChannelApprovalResponse::Deny,
        ChannelApprovalResponse::AlwaysApprove,
    ]);
    let (hooks, heard) = answer_recording_hooks(false);
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup", "lookup", "lookup"]),
        vec![tool],
        &hooks,
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(
        *heard.lock().expect("answer lock"),
        vec![
            (
                format!("{TURN_ID}:0:0"),
                true,
                ChannelApprovalResponse::Approve,
                Operator
            ),
            (
                format!("{TURN_ID}:0:1"),
                true,
                ChannelApprovalResponse::Deny,
                Operator
            ),
            (
                format!("{TURN_ID}:0:2"),
                true,
                ChannelApprovalResponse::AlwaysApprove,
                Operator
            ),
        ]
    );
}

#[tokio::test]
async fn hooks_hear_the_runtime_deny_a_call_no_operator_could_answer() {
    use zeroclaw_api::channel::ApprovalSource::Unavailable;
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let no_channel = Turn {
        approval: Some(&approval),
        ..Turn::default()
    };
    let scheduled = Turn {
        ingress: IngressContext::cron_job("job-7", None),
        ..Turn::default()
    };

    for (turn, ask) in [(no_channel, false), (scheduled, true)] {
        let (hooks, heard) = answer_recording_hooks(ask);
        let (tool, runs) = counting_tool("lookup");
        let (result, _) = run_turn(
            turn,
            &ScriptedModel::calling(&["lookup"]),
            vec![tool],
            &hooks,
        )
        .await;

        result.expect("turn completes");
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            *heard.lock().expect("answer lock"),
            vec![(
                format!("{TURN_ID}:0:0"),
                true,
                ChannelApprovalResponse::Deny,
                Unavailable
            )]
        );
    }
}

#[tokio::test]
async fn hooks_hear_an_unreachable_approver_as_the_runtime_denying() {
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover {
        unreachable: true,
        ..ScriptedApprover::answering(Vec::new())
    };
    let (hooks, heard) = answer_recording_hooks(false);
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &hooks,
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        *heard.lock().expect("answer lock"),
        vec![(
            format!("{TURN_ID}:0:0"),
            true,
            ChannelApprovalResponse::Deny,
            zeroclaw_api::channel::ApprovalSource::Unreachable
        )]
    );
}

#[tokio::test]
async fn hooks_hear_nothing_for_calls_that_needed_no_answer() {
    let approval = crate::approval::ApprovalManager::for_non_interactive(&supervised_profile());
    let approver = ScriptedApprover::answering(Vec::new());
    let (mut hooks, heard) = answer_recording_hooks(false);
    hooks.register(Box::new(DecidingHook {
        matching: None,
        approve: true,
    }));
    let (tool, runs) = counting_tool("lookup");

    let (result, _) = run_turn(
        Turn {
            approval: Some(&approval),
            channel: Some(&approver),
            ..Turn::default()
        },
        &ScriptedModel::calling(&["lookup"]),
        vec![tool],
        &hooks,
    )
    .await;

    result.expect("turn completes");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(heard.lock().expect("answer lock").is_empty());
}
