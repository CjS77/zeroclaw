//! A turn `agent::run` runs, as a scheduled job's turn does, calls the embedder hooks
//! installed with `install_embedder_hooks`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{Router, routing::post};
use tempfile::TempDir;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};
use zeroclaw_runtime::hooks::{EmbedderHooks, HookHandler, install_embedder_hooks};

const FAKE_OPENAI_RESPONSE: &str = r#"{"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

/// Counts the model calls it sees.
struct CountingHook(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl HookHandler for CountingHook {
    fn name(&self) -> &str {
        "counting"
    }

    async fn on_llm_input(&self, _messages: &[ChatMessage], _model: &str) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A provider that answers every chat completion with "ok"; returns its base URI.
async fn spawn_mock_provider() -> String {
    let app = Router::new().route("/chat/completions", post(|| async { FAKE_OPENAI_RESPONSE }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    zeroclaw_spawn::spawn!(async move {
        let _ = axum::serve(listener, app.into_make_service()).await;
    });
    format!("http://{addr}")
}

fn config_for(provider_uri: String, data_dir: &std::path::Path) -> Config {
    let mut providers = zeroclaw_config::providers::Providers::default();
    let base = providers
        .models
        .ensure("custom", "default")
        .expect("`custom` slot must exist on ModelProviders");
    base.api_key = Some("test-key".to_string());
    base.model = Some("test-model".to_string());
    base.uri = Some(provider_uri);
    let agent = AliasedAgentConfig {
        enabled: true,
        model_provider: "custom.default".into(),
        risk_profile: "default".into(),
        ..Default::default()
    };
    let mut config = Config {
        data_dir: data_dir.to_path_buf(),
        config_path: data_dir.join("config.toml"),
        providers,
        agents: HashMap::from([("default".to_string(), agent)]),
        risk_profiles: HashMap::from([("default".to_string(), RiskProfileConfig::default())]),
        ..Config::default()
    };
    config.reliability.scheduler_retries = 0;
    config.reliability.provider_retries = 0;
    config.hooks.enabled = false;
    config
}

#[tokio::test]
async fn agent_run_turn_calls_installed_hooks() {
    let tmp = TempDir::new().unwrap();
    let config = config_for(spawn_mock_provider().await, tmp.path());
    let calls = Arc::new(AtomicUsize::new(0));
    install_embedder_hooks(EmbedderHooks::new().with(Arc::new(CountingHook(Arc::clone(&calls)))));

    let result = zeroclaw_runtime::agent::run(
        config,
        "default",
        Some("Say ok.".to_string()),
        None,
        None,
        Some(0.7),
        vec![],
        false,
        None,
        None,
        zeroclaw_api::ingress::TurnOrigin::Cron,
        Default::default(),
    )
    .await;

    assert!(result.is_ok(), "the run failed: {result:?}");
    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "the turn never called the embedder hook"
    );
}
