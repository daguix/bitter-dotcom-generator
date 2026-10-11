//! Two-pass final selection by an LLM.
//!
//! A single pass with a powerful model over a thousand names would be expensive
//! for work that is 90% obvious rejection. The inexpensive model therefore
//! triages first (fit score only, in batches of 100), and only the survivors are
//! fully described.
//!
//! The LLM never sees the Markov score: this is an independent judgment that
//! should not be anchored to a purely phonetic metric.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Provider {
    Anthropic,
    Openai,
    Ollama,
}

impl Provider {
    pub fn default_base(self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com/v1",
            Provider::Openai => "https://api.openai.com/v1",
            Provider::Ollama => "http://localhost:11434/v1",
        }
    }

    pub fn default_triage_model(self) -> &'static str {
        match self {
            Provider::Anthropic => "claude-sonnet-5-5",
            Provider::Openai => "gpt-5.4-mini",
            Provider::Ollama => "gpt-oss:20b",
        }
    }

    pub fn default_description_model(self) -> &'static str {
        match self {
            Provider::Anthropic => "claude-opus-5-5",
            Provider::Openai => "gpt-5.5",
            Provider::Ollama => "gpt-oss:20b",
        }
    }

    pub fn cache_model(self, api_base: &str, model: &str) -> String {
        // Hosted model identifiers are unambiguous on their own; only local or
        // overridden endpoints need qualifying.
        if self != Provider::Ollama && api_base == self.default_base() {
            model.to_string()
        } else {
            format!("{self:?}@{api_base}:{model}")
        }
    }
}

pub struct Judge {
    http: reqwest::Client,
    provider: Provider,
    api_url: String,
    key: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct Triage {
    pub name: String,
    /// Fit from 0 to 10 relative to the brief.
    pub fit: f32,
}

#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Verdict {
    pub name: String,
    pub tagline: String,
    pub description: String,
    /// Why this name fits the stated positioning.
    pub rationale: String,
    /// Concerns: similarity to a brand, pronunciation ambiguity, and so on.
    #[serde(default)]
    pub concerns: String,
    /// Model and prompt fingerprint that produced this description. A severity
    /// is meaningful only with them: changing the model or prompt changes the
    /// judgment, and reusing the old one would make it appear current.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    /// Severity of concerns, from 1 to 5. Without it, `concerns` cannot rank
    /// anything: a conscientious model always finds something to say about every
    /// name, and everything ends up flagged at the same level.
    #[serde(default)]
    pub severity: u8,
}

#[derive(Deserialize)]
struct Wrapper<T> {
    results: Vec<T>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
}
#[derive(Deserialize)]
struct Message {
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
}
#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

/// Claude models that accept server-side refusal fallbacks in their `"default"`
/// form. Sending the parameter to any other model is rejected.
const FALLBACK_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
];

/// Fallback location deliberately kept outside the repository, so a key stored
/// there cannot be committed accidentally.
fn key_file(name: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        std::path::Path::new(&home)
            .join(".config/bitter-dotcom-generator")
            .join(name),
    )
}

/// Looks for an API key in the environment, then in the configuration file.
fn find_key(var: &str, file: &str) -> Result<String> {
    if let Ok(k) = std::env::var(var) {
        let k = k.trim().to_string();
        if !k.is_empty() {
            return Ok(k);
        }
    }
    let path = key_file(file).context("HOME is unavailable")?;
    let raw = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no key found: neither {var} in the environment nor {}",
            path.display()
        )
    })?;
    let k = raw.trim().to_string();
    anyhow::ensure!(!k.is_empty(), "{} is empty", path.display());
    Ok(k)
}

fn optional_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

impl Judge {
    pub fn new(provider: Provider, api_base: &str, temperature: Option<f32>) -> Result<Judge> {
        let api_base = api_base.trim_end_matches('/');
        anyhow::ensure!(!api_base.is_empty(), "API base URL cannot be empty");
        let api_url = match provider {
            Provider::Anthropic => format!("{api_base}/messages"),
            Provider::Openai | Provider::Ollama => format!("{api_base}/chat/completions"),
        };
        let parsed = reqwest::Url::parse(&api_url).context("invalid API base URL")?;
        anyhow::ensure!(
            matches!(parsed.scheme(), "http" | "https"),
            "API base URL must use HTTP or HTTPS"
        );
        anyhow::ensure!(
            parsed.username().is_empty() && parsed.password().is_none(),
            "API credentials must not be embedded in the URL"
        );
        let key = match provider {
            Provider::Anthropic => Some(find_key("ANTHROPIC_API_KEY", "anthropic-key")?),
            Provider::Openai => Some(find_key("OPENAI_API_KEY", "openai-key")?),
            Provider::Ollama => optional_env("OLLAMA_API_KEY"),
        };
        // 120 seconds was insufficient: a reasoning model comparing ten names
        // with existing brands can easily exceed that timeout, silently losing
        // the batch.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .context("building the HTTP client")?;
        Ok(Judge {
            http,
            provider,
            api_url,
            key,
            temperature,
        })
    }

    /// `effort` applies to Claude only.
    async fn ask<T: for<'de> Deserialize<'de>>(
        &self,
        model: &str,
        system: &str,
        user: &str,
        effort: &str,
    ) -> Result<Vec<T>> {
        let mut body = match self.provider {
            Provider::Anthropic => {
                let mut body = serde_json::json!({
                    "model": model,
                    "max_tokens": 16000,
                    "system": system,
                    "messages": [{ "role": "user", "content": user }],
                    // No `output_config.format`: constrained to a JSON schema,
                    // claude-opus-5-5 did its analysis while thinking and then
                    // returned a single empty placeholder entry, in 13 of 14
                    // trials on 2026-10-10. The prompt alone yields complete
                    // JSON.
                    "output_config": { "effort": effort },
                });
                // A safety classifier can decline a request outright. The
                // default fallback reruns it on another model inside the same
                // call instead of losing the batch.
                if FALLBACK_MODELS.contains(&model) {
                    body["fallbacks"] = serde_json::json!("default");
                }
                body
            }
            Provider::Openai | Provider::Ollama => serde_json::json!({
                "model": model,
                "messages": [
                    { "role": "system", "content": system },
                    { "role": "user", "content": user },
                ],
                "response_format": { "type": "json_object" },
            }),
        };
        // A low temperature produces reproducible judgments, but reasoning
        // models accept only their default value and reject the request. Send it
        // only when explicitly requested.
        if let Some(t) = self.temperature {
            body["temperature"] = serde_json::json!(t);
        }

        // A lost batch means ten names silently disappear from the result. Retry
        // all transient failures: network interruption, timeout, quota, or server
        // outage.
        let mut last: anyhow::Error = anyhow::anyhow!("no attempts made");
        for attempt in 0..3 {
            if attempt > 0 {
                let wait = Duration::from_secs(5 << attempt);
                eprintln!("retrying in {}s after: {last:#}", wait.as_secs());
                tokio::time::sleep(wait).await;
            }
            match self.attempt(&body).await {
                Ok(v) => return Ok(v),
                // Waiting will not fix an invalid key or exhausted quota. More
                // retries would only bury the useful error message.
                Err(e) if permanent(&e) => return Err(e),
                Err(e) => last = e,
            }
        }
        Err(last).context("three unsuccessful attempts")
    }

    async fn attempt<T: for<'de> Deserialize<'de>>(
        &self,
        body: &serde_json::Value,
    ) -> Result<Vec<T>> {
        let resp = self.http.post(&self.api_url);
        let request = match (self.provider, &self.key) {
            (Provider::Anthropic, Some(key)) => {
                let request = resp
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01");
                if body.get("fallbacks").is_some() {
                    request.header("anthropic-beta", "server-side-fallback-2026-07-01")
                } else {
                    request
                }
            }
            (_, Some(key)) => resp.bearer_auth(key),
            (_, None) => resp,
        };
        let resp = request
            .json(body)
            .send()
            .await
            .with_context(|| format!("calling the {:?} API", self.provider))?;

        let status = resp.status();
        let text = resp.text().await.context("reading the response")?;
        if !status.is_success() {
            let fatal = matches!(status.as_u16(), 400 | 401 | 403 | 404)
                || text.contains("insufficient_quota")
                || text.contains("billing")
                || text.contains("credit balance");
            bail!(
                "{}the API returned {status}: {text}",
                if fatal { "[permanent] " } else { "" }
            );
        }

        let content = match self.provider {
            Provider::Anthropic => {
                let msg: MessagesResponse =
                    serde_json::from_str(&text).context("unexpected API response format")?;
                match msg.stop_reason.as_deref() {
                    // Retrying the same request would only be declined again.
                    Some("refusal") => bail!("[permanent] the model declined the request"),
                    Some("max_tokens") => bail!(
                        "[permanent] the response hit max_tokens; lower --batch or --describe-batch"
                    ),
                    _ => {}
                }
                // Thinking and fallback blocks carry no answer: keep text only.
                msg.content
                    .into_iter()
                    .filter(|b| b.kind == "text")
                    .map(|b| b.text)
                    .collect::<String>()
            }
            Provider::Openai | Provider::Ollama => {
                let chat: ChatResponse =
                    serde_json::from_str(&text).context("unexpected API response format")?;
                chat.choices
                    .into_iter()
                    .next()
                    .map(|c| c.message.content)
                    .context("response has no content")?
            }
        };
        // Without a schema the JSON may come wrapped in a code fence.
        let json = match (content.find('{'), content.rfind('}')) {
            (Some(start), Some(end)) if start < end => &content[start..=end],
            _ => content.as_str(),
        };
        let parsed: Wrapper<T> = serde_json::from_str(json)
            .with_context(|| format!("the model did not return the expected JSON: {content}"))?;
        Ok(parsed.results)
    }

    /// Pass 1: fit score only, in batches.
    pub async fn triage(&self, model: &str, brief: &str, names: &[String]) -> Result<Vec<Triage>> {
        let system = format!(
            "You rate candidate brand names for a company. Here is the company brief:\n\n\
             {brief}\n\n\
             For each name, give a fit score from 0 to 10 judging how well it could serve as \
             the name described in the brief: sound, memorability, suitability for the \
             technical audience it describes, and absence of unfortunate connotations in any major \
             language. Be harsh — most names deserve below 5. Reply with JSON only, in the form \
             {{\"results\": [{{\"name\": \"...\", \"fit\": 0.0}}]}}, one entry per name given, \
             no commentary."
        );
        let out: Vec<Triage> = self
            .ask(model, &system, &names.join("\n"), "medium")
            .await?;
        validate_response_names(&out, names, |t| &t.name)?;
        for t in &out {
            anyhow::ensure!(
                t.fit.is_finite() && (0.0..=10.0).contains(&t.fit),
                "score out of range for {}: {}",
                t.name,
                t.fit
            );
        }
        Ok(out)
    }

    /// Pass 2: full description of shortlisted names only.
    pub async fn describe(
        &self,
        model: &str,
        cache_model: &str,
        brief: &str,
        names: &[String],
    ) -> Result<Vec<Verdict>> {
        let system = describe_system(brief);
        // Collision screening is the point of this pass and draws on what the
        // model knows of existing brands: give it room to think.
        let mut out: Vec<Verdict> = self.ask(model, &system, &names.join("\n"), "high").await?;
        validate_response_names(&out, names, |v| &v.name)?;
        for v in &out {
            anyhow::ensure!(
                (1..=5).contains(&v.severity),
                "severity out of range for {}: {}",
                v.name,
                v.severity
            );
        }
        let id = prompt_id(cache_model, brief);
        for v in &mut out {
            v.model = cache_model.to_string();
            v.prompt = id.clone();
        }
        Ok(out)
    }
}

fn validate_response_names<T>(
    results: &[T],
    requested: &[String],
    name: impl Fn(&T) -> &str,
) -> Result<()> {
    let expected: HashSet<&str> = requested.iter().map(String::as_str).collect();
    anyhow::ensure!(
        expected.len() == requested.len(),
        "the requested batch contains duplicate names"
    );
    let mut seen = HashSet::new();
    for item in results {
        let current = name(item);
        anyhow::ensure!(
            expected.contains(current),
            "the response contains an unrequested name: {current}"
        );
        anyhow::ensure!(
            seen.insert(current),
            "duplicate name in the response: {current}"
        );
    }
    anyhow::ensure!(
        seen.len() == expected.len(),
        "incomplete response: {} name(s) requested, {} received",
        expected.len(),
        seen.len()
    );
    Ok(())
}

/// Pass-2 prompt, isolated so it can be fingerprinted: it determines the
/// resulting judgment just as much as the model does.
/// Returns true for errors no retry can fix: rejected key, nonexistent model,
/// exhausted quota, or malformed request.
fn permanent(e: &anyhow::Error) -> bool {
    format!("{e:#}").contains("[permanent]")
}

pub fn describe_system(brief: &str) -> String {
    format!(
        "You are naming a company or product. Here is the brief:\n\n{brief}\n\n\
             For each candidate name, write: a tagline (under 8 words), a description of the \
             product or positioning the name would suit (2 sentences), and a rationale tying \
             the name's sound or associations to the brief.\n\n\
             Then fill the `concerns` field. Treat this as the most important field, and be \
             actively suspicious rather than charitable. These names were coined by a Markov \
             model trained on the technical vocabulary of this domain, so their typical flaw is \
             not imitating a company but sitting one letter away from an ordinary word.\n\n\
             Weigh every collision by market distance, because trademark protection is scoped \
             to a sector. A collision with a company selling to this audience — the markets \
             named in the brief, or else infrastructure, devtools and open-source projects — is severe: a \
             shared distinctive stem is a real problem there even when the endings differ, \
             because an engineer could take the name for a spin-off of an existing product. \
             But a stem that is an ordinary word or a common technical term of the domain \
             (syn-, tile, frac-, vect-, grid, mesh, clock…) is not distinctive: these names are \
             built from that vocabulary on purpose, and sharing such a stem with a brand is not \
             a collision. Judge the whole name, written and spoken in full, against the brand \
             in full; flag it only if the two could actually be mistaken for one another. A \
             resemblance to a well-known brand in an unrelated market, such as clothing, food, \
             banking or travel, deserves one sentence and no more; it does not block the name \
             and must not drive the rating. State which of the two cases you are in.\n\n\
             Then apply the spoken test, in both directions. Reading direction: can someone \
             seeing the name written guess how to say it? Hearing direction, which matters more \
             and is easier to overlook: imagine the name said aloud in a meeting or a podcast, \
             with no spelling given, and write down what a listener would afterwards type into \
             a search box. If that differs from the actual name — because an ordinary word or a \
             common brand sits closer to what was heard — the name loses people every time it \
             is spoken. Treat that as a serious defect and name the word they would type.\n\n\
             Also flag awkward or offensive meanings in any major language, French included. \
             Use an empty string only when you genuinely find nothing after having looked.\n\n\
             Also give `severity`, an integer from 1 to 5 rating how damaging the concerns \
             are: 1 means a remark worth noting that would not stop anyone adopting the name, \
             3 means a real obstacle that needs deliberate handling, 5 means the name is \
             unusable — a direct collision with a company selling to this audience, a name that \
             reliably mishears as a common word, or a meaning that embarrasses. A resemblance \
             to a famous brand in an unrelated industry is a 1 or a 2, never a 5. Spread your \
             ratings across the scale; if every name receives the same severity the field tells \
             the reader nothing.\n\n\
             Reply with JSON only, in the form \
             {{\"results\": [{{\"name\": \"...\", \"tagline\": \"...\", \"description\": \"...\", \
             \"rationale\": \"...\", \"concerns\": \"...\", \"severity\": 1}}]}}, \
             one entry per name given."
    )
}

/// Short fingerprint of the (model, prompt) pair. The brief is included in the
/// prompt, so changing it also invalidates descriptions, as intended.
pub fn prompt_id(model: &str, brief: &str) -> String {
    // FNV-1a is stable across Rust versions, unlike the standard library's
    // default hasher.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model
        .as_bytes()
        .iter()
        .chain(describe_system(brief).as_bytes())
    {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_defaults_are_isolated_in_caches() {
        assert_eq!(Provider::Openai.default_base(), "https://api.openai.com/v1");
        assert_eq!(
            Provider::Anthropic.cache_model("https://api.anthropic.com/v1", "example"),
            "example"
        );
        assert_eq!(Provider::Ollama.default_base(), "http://localhost:11434/v1");
        assert_eq!(
            Provider::Openai.cache_model("https://api.openai.com/v1", "example"),
            "example"
        );
        assert_eq!(
            Provider::Ollama.cache_model("http://localhost:11434/v1", "example"),
            "Ollama@http://localhost:11434/v1:example"
        );
    }

    #[test]
    fn ollama_client_does_not_require_an_openai_key() {
        let judge = Judge::new(Provider::Ollama, "http://localhost:11434/v1/", None).unwrap();
        assert_eq!(judge.api_url, "http://localhost:11434/v1/chat/completions");
    }

    #[test]
    fn rejects_partial_or_foreign_responses() {
        let requested = vec!["alpha".to_string(), "beta".to_string()];
        let partial = vec![Triage {
            name: "alpha".to_string(),
            fit: 5.0,
        }];
        assert!(validate_response_names(&partial, &requested, |t| &t.name).is_err());

        let foreign = vec![
            Triage {
                name: "alpha".to_string(),
                fit: 5.0,
            },
            Triage {
                name: "gamma".to_string(),
                fit: 5.0,
            },
        ];
        assert!(validate_response_names(&foreign, &requested, |t| &t.name).is_err());
    }
}
