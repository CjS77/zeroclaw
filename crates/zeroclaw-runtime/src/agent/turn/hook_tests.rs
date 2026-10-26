//! What a tool-call hook sees and decides at the turn-loop boundary: real
//! turns through `run_tool_call_loop`, with a scripted model.

use super::*;
use crate::observability::NoopObserver;
use std::sync::Mutex;
use zeroclaw_api::hook::{HookTurnContext, ToolCallHookContext};
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

async fn run_turn(
    turn: Turn<'_>,
    model: &ScriptedModel,
    tools: Vec<Box<dyn crate::tools::Tool>>,
    hooks: &crate::hooks::HookRunner,
) -> anyhow::Result<String> {
    let registry = crate::tools::scoped::ScopedToolRegistry::from_raw_for_test(tools);
    let observer = NoopObserver;
    let pacing = zeroclaw_config::schema::PacingConfig::default();
    let mut history = vec![ChatMessage::user("go")];
    let mut crumb_present = false;
    let mut injected_preamble = None;
    run_tool_call_loop(ToolLoop {
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
    .await
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
