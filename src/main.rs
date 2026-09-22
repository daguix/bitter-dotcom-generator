mod check;
mod edit;
mod judge;
mod markov;
mod scan;
mod syllable;

use anyhow::{Context, Result};
use check::{Availability, Rdap, check_all};
use clap::{Parser, Subcommand};
use judge::{Judge, Provider, Triage, Verdict, prompt_id};
use markov::Markov;
use scan::{ScanOpts, scan};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "bitter", about = "Pronounceable .com domain name generator")]
struct Cli {
    /// Training corpus (one name per line). Defaults to the embedded corpus.
    #[arg(long, global = true)]
    corpus: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scans the entire [a-z]^L space and outputs the best pronounceable names.
    #[command(allow_negative_numbers = true)]
    Scan {
        /// Name length (6 or 7).
        #[arg(long, default_value_t = 7, value_parser = parse_scan_len)]
        len: usize,
        /// Average log-probability threshold per transition. Higher is stricter.
        #[arg(long, default_value_t = -2.0, value_parser = parse_finite_f32)]
        threshold: f32,
        /// Number of names retained after sorting.
        #[arg(long, default_value_t = 50_000)]
        top: usize,
        #[arg(long, default_value_t = 2, value_parser = parse_syllables)]
        min_syl: usize,
        #[arg(long, default_value_t = 3, value_parser = parse_syllables)]
        max_syl: usize,
        /// Reject names sharing an n-gram of this size with the corpus. 0
        /// disables the filter. Protects against both seed similarity and blandness.
        #[arg(long, default_value_t = 5)]
        novelty: usize,
        /// Reject names fewer than N edits away from a seed. The n-gram filter
        /// misses copies such as cenroid and centroid, which share no 5-gram.
        /// 0 disables the filter.
        #[arg(long, default_value_t = 3)]
        min_edit: usize,
        /// Output file (JSONL). Defaults to stdout.
        #[arg(long)]
        out: Option<String>,
    },
    /// Prints model scores for given names to calibrate the threshold.
    Score { words: Vec<String> },
    /// Checks names from a JSONL file and retains only available domains.
    Check {
        /// Output from `scan` (JSONL).
        input: String,
        /// Requests per second. 20 is a restrained rate; do not overdo it.
        #[arg(long, default_value_t = 20, value_parser = parse_positive_u32)]
        rate: u32,
        #[arg(long, default_value_t = 8, value_parser = parse_positive_usize)]
        concurrency: usize,
        /// Checks only the top N entries in the file.
        #[arg(long)]
        limit: Option<usize>,
        /// Journal of known verdicts, so a domain is never checked twice.
        #[arg(long, default_value = "cache/rdap.jsonl")]
        cache: String,
        #[arg(long)]
        out: Option<String>,
    },
    /// Sends available names to the LLM for brief-based triage and description.
    Judge {
        /// Output from `check` (JSONL of available names).
        input: String,
        /// LLM provider. Ollama uses its local OpenAI-compatible endpoint.
        #[arg(long, value_enum, default_value_t = Provider::Openai)]
        provider: Provider,
        /// API base URL. Defaults to the selected provider's standard endpoint.
        #[arg(long)]
        api_base: Option<String>,
        /// Positioning brief submitted to the model.
        #[arg(long, default_value = "data/brief.txt")]
        brief: String,
        /// Triage model for pass 1. Most of the volume goes through it.
        #[arg(long)]
        triage_model: Option<String>,
        /// Description model for pass 2, used only on finalists.
        #[arg(long)]
        model: Option<String>,
        /// Names per request in pass 1.
        #[arg(long, default_value_t = 100, value_parser = parse_positive_usize)]
        batch: usize,
        /// Names per request in pass 2. Smaller than pass 1 because every name
        /// needs a full description, and a short batch returns sooner.
        #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
        describe_batch: usize,
        /// Number of finalists sent for description.
        #[arg(long, default_value_t = 40)]
        keep: usize,
        /// Two finalists may not share an n-gram of this size. Without this
        /// constraint, fit-based sorting returns clusters of near-duplicates
        /// (sequing, sequary, sequigi…): many names, few ideas. Must be positive.
        #[arg(long, default_value_t = 4, value_parser = parse_positive_usize)]
        diversity: usize,
        /// Concurrent requests.
        #[arg(long, default_value_t = 4, value_parser = parse_positive_usize)]
        concurrency: usize,
        /// Temperature. Omitted by default because recent reasoning models reject
        /// any value other than their default.
        #[arg(long, value_parser = parse_finite_f32)]
        temperature: Option<f32>,
        /// Pass-1 score journal, loaded at startup. A scored name is never sent
        /// back to the model and therefore never paid for twice.
        #[arg(long, default_value = "cache/fit.jsonl")]
        fit_cache: String,
        /// Stops after pass 1 and writes the selection without descriptions.
        #[arg(long)]
        triage_only: bool,
        /// Journal of existing descriptions. A name already described by the same
        /// model and prompt is never sent back to the model.
        #[arg(long, default_value = "cache/judgments.jsonl")]
        judged_cache: String,
        #[arg(long)]
        out: Option<String>,
    },
}

/// Default corpus embedded so the binary remains usable on its own. The on-disk
/// file takes precedence when present; otherwise edits to data/seeds.txt would
/// have no effect until recompilation and scans could silently use stale data.
const EMBEDDED_SEEDS: &str = include_str!("../data/seeds.txt");
const DEFAULT_SEEDS: &str = "data/seeds.txt";

fn parse_positive_usize(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("{s:?} is not an integer"))?;
    (n > 0)
        .then_some(n)
        .ok_or_else(|| "the value must be greater than zero".to_string())
}

fn parse_positive_u32(s: &str) -> std::result::Result<u32, String> {
    let n: u32 = s.parse().map_err(|_| format!("{s:?} is not an integer"))?;
    (n > 0)
        .then_some(n)
        .ok_or_else(|| "the value must be greater than zero".to_string())
}

fn parse_scan_len(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("{s:?} is not an integer"))?;
    matches!(n, 6 | 7)
        .then_some(n)
        .ok_or_else(|| "length must be 6 or 7".to_string())
}

fn parse_syllables(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s.parse().map_err(|_| format!("{s:?} is not an integer"))?;
    (1..=4)
        .contains(&n)
        .then_some(n)
        .ok_or_else(|| "the syllable count must be between 1 and 4".to_string())
}

fn parse_finite_f32(s: &str) -> std::result::Result<f32, String> {
    let n: f32 = s.parse().map_err(|_| format!("{s:?} is not a number"))?;
    n.is_finite()
        .then_some(n)
        .ok_or_else(|| "the value must be finite".to_string())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 63 && name.bytes().all(|b| b.is_ascii_lowercase())
}

fn ensure_valid_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for name in names {
        anyhow::ensure!(
            valid_name(name),
            "invalid name: {name:?} (expected 1 to 63 lowercase a-z letters)"
        );
        anyhow::ensure!(seen.insert(name), "duplicate name in input: {name}");
    }
    Ok(())
}

fn corpus(path: Option<&str>) -> Result<Vec<String>> {
    let raw = match path {
        Some(p) => std::fs::read_to_string(p).with_context(|| format!("reading {p}"))?,
        None => {
            std::fs::read_to_string(DEFAULT_SEEDS).unwrap_or_else(|_| EMBEDDED_SEEDS.to_string())
        }
    };
    Ok(raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && l.bytes().all(|b| b.is_ascii_lowercase()))
        .map(String::from)
        .collect())
}

/// All n-grams of size `k` present in the corpus.
fn ngrams(words: &[String], k: usize) -> HashSet<&[u8]> {
    let mut set = HashSet::new();
    if k == 0 {
        return set;
    }
    for w in words {
        let b = w.as_bytes();
        for i in 0..b.len().saturating_sub(k - 1) {
            set.insert(&b[i..i + k]);
        }
    }
    set
}

fn load_model(path: Option<&str>) -> Result<(Markov, Vec<String>)> {
    let words = corpus(path)?;
    anyhow::ensure!(!words.is_empty(), "empty corpus");
    eprintln!("corpus: {} seeds", words.len());
    Ok((Markov::train(&words), words))
}

#[derive(serde::Deserialize)]
struct ScanRow {
    name: String,
    score: f32,
}

/// Known verdicts loaded from the journal.
fn load_cache(path: &str) -> Result<HashMap<String, Availability>> {
    let mut map = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(map); // first run
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("unreadable cache line: {line}"))?;
        let (Some(name), Some(verdict)) = (v["name"].as_str(), v["verdict"].as_str()) else {
            continue;
        };
        let a = match verdict {
            "free" => Availability::Free,
            "taken" => Availability::Taken,
            // Do not cache unknown results; they deserve another attempt.
            _ => continue,
        };
        map.insert(name.to_string(), a);
    }
    Ok(map)
}

/// Existing fit scores loaded from the journal.
///
/// Two models do not score identically, so a score is meaningful only with the
/// model that produced it. Without this key, switching models would silently
/// reuse old scores and make `--triage-model` ineffective.
fn load_fit_cache(path: &str, model: &str) -> Result<HashMap<String, f32>> {
    let mut map = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(map); // first run
    };
    let (mut other_models, mut legacy) = (0usize, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("unreadable cache line: {line}"))?;
        let (Some(name), Some(fit)) = (v["name"].as_str(), v["fit"].as_f64()) else {
            continue;
        };
        match v["model"].as_str() {
            Some(m) if m == model => {
                map.insert(name.to_string(), fit as f32);
            }
            Some(_) => other_models += 1,
            // Accept journal entries written before the model was recorded, but
            // never silently.
            None => {
                legacy += 1;
                map.insert(name.to_string(), fit as f32);
            }
        }
    }
    if other_models > 0 {
        eprintln!("cache: {other_models} scores ignored because they came from another model");
    }
    if legacy > 0 {
        eprintln!(
            "cache: {legacy} scores without a recorded model reused; \
             delete them if they did not come from {model}"
        );
    }
    Ok(map)
}

/// Existing descriptions restricted to the current (model, prompt) pair.
fn load_judged(path: &str, model: &str, prompt: &str) -> Result<HashMap<String, Verdict>> {
    let mut map = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(map);
    };
    let mut stale = 0usize;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Verdict = serde_json::from_str(line)
            .with_context(|| format!("unreadable judgment line: {line}"))?;
        if v.model == model && v.prompt == prompt {
            map.insert(v.name.clone(), v);
        } else {
            stale += 1;
        }
    }
    if stale > 0 {
        eprintln!("cache: {stale} descriptions ignored due to a different model or prompt");
    }
    Ok(map)
}

/// Rewrites the description journal by increasing severity, then by name.
fn sort_judgments(path: &str) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("rereading {path}"))?;
    let mut rows: Vec<Verdict> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .context("unreadable description journal")?;
    rows.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.name.cmp(&b.name))
    });
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("rewriting {path}"))?,
    );
    for r in &rows {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    f.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Score { words } => {
            ensure_valid_names(words.iter().map(String::as_str))?;
            let (m, _) = load_model(cli.corpus.as_deref())?;
            for w in words {
                println!("{:<10} {:.3}", w, m.score(&w));
            }
        }
        Cmd::Scan {
            len,
            threshold,
            top,
            min_syl,
            max_syl,
            novelty,
            min_edit,
            out,
        } => {
            anyhow::ensure!(
                min_syl <= max_syl,
                "min-syl must be less than or equal to max-syl"
            );
            let (m, words) = load_model(cli.corpus.as_deref())?;
            let t0 = std::time::Instant::now();
            let mut found = scan(
                &m,
                &ScanOpts {
                    len,
                    threshold,
                    min_syl,
                    max_syl,
                },
            );
            let elapsed = t0.elapsed();
            // The model overfits its corpus: at order 3 it stitches memorized
            // fragments into near-copies ("sendesk" beside Zendesk). Reject any
            // name sharing an n-gram with the corpus, both for novelty and to
            // avoid near-copies of seeds.
            let banned = ngrams(&words, novelty);
            let before = found.len();
            found.retain(|c| {
                let b = c.name.as_bytes();
                novelty == 0
                    || !(0..b.len().saturating_sub(novelty - 1))
                        .any(|i| banned.contains(&b[i..i + novelty]))
            });
            let dropped = before - found.len();

            // Apply after n-grams: the set is already much smaller and edit
            // distance is substantially more expensive.
            let before_edit = found.len();
            if min_edit > 0 {
                found.retain(|c| !edit::near_any(&c.name, &words, min_edit));
            }
            let close_copies = before_edit - found.len();

            found.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            let kept = found.len().min(top);
            eprintln!(
                "{} names above threshold in {:.1}s ({} shared n-grams, {} close copies) — {} retained",
                found.len(),
                elapsed.as_secs_f32(),
                dropped,
                close_copies,
                kept
            );
            if let Some(c) = found.first() {
                eprintln!("best: {} ({:.3})", c.name, c.score);
            }
            if let Some(c) = found.get(kept.saturating_sub(1)) {
                eprintln!("last retained: {} ({:.3})", c.name, c.score);
            }

            let mut sink: Box<dyn Write> = match &out {
                Some(p) => Box::new(std::io::BufWriter::new(
                    std::fs::File::create(p).with_context(|| format!("creating {p}"))?,
                )),
                None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
            };
            for c in &found[..kept] {
                writeln!(sink, r#"{{"name":"{}","score":{:.4}}}"#, c.name, c.score)?;
            }
            sink.flush()?;
        }
        Cmd::Check {
            input,
            rate,
            concurrency,
            limit,
            cache,
            out,
        } => {
            let text =
                std::fs::read_to_string(&input).with_context(|| format!("reading {input}"))?;
            let mut rows: Vec<ScanRow> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()
                .context("unexpected input JSONL format")?;
            if let Some(n) = limit {
                rows.truncate(n);
            }
            ensure_valid_names(rows.iter().map(|r| r.name.as_str()))?;

            let known = load_cache(&cache)?;
            let pending: Vec<String> = rows
                .iter()
                .map(|r| r.name.clone())
                .filter(|n| !known.contains_key(n))
                .collect();
            eprintln!(
                "{} names, {} already known, {} to check — about {:.0} min at {}/s",
                rows.len(),
                rows.len() - pending.len(),
                pending.len(),
                pending.len() as f64 / rate as f64 / 60.0,
                rate
            );

            let mut journal = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&cache)
                    .with_context(|| format!("opening journal {cache}"))?,
            );
            let mut fresh = HashMap::new();
            let mut done = 0usize;
            check_all(
                Arc::new(Rdap::new()?),
                pending,
                rate,
                concurrency,
                |name, verdict| {
                    let label = match verdict {
                        Availability::Free => "free",
                        Availability::Taken => "taken",
                        Availability::Unknown => "unknown",
                    };
                    writeln!(
                        journal,
                        "{}",
                        serde_json::json!({ "name": name, "verdict": label })
                    )?;
                    journal.flush()?;
                    fresh.insert(name.to_string(), verdict);
                    done += 1;
                    if done.is_multiple_of(200) {
                        eprint!("\r{done} checked…");
                    }
                    Ok(())
                },
            )
            .await?;
            journal.flush()?;
            eprintln!();

            let mut sink: Box<dyn Write> = match &out {
                Some(p) => Box::new(std::io::BufWriter::new(
                    std::fs::File::create(p).with_context(|| format!("creating {p}"))?,
                )),
                None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
            };
            let mut free = 0;
            for r in &rows {
                let verdict = fresh.get(&r.name).or_else(|| known.get(&r.name));
                if verdict == Some(&Availability::Free) {
                    writeln!(sink, r#"{{"name":"{}","score":{:.4}}}"#, r.name, r.score)?;
                    free += 1;
                }
            }
            sink.flush()?;
            eprintln!(
                "{free} available out of {} ({:.1}%)",
                rows.len(),
                100.0 * free as f64 / rows.len() as f64
            );
        }
        Cmd::Judge {
            input,
            provider,
            api_base,
            brief,
            triage_model,
            model,
            batch,
            describe_batch,
            keep,
            diversity,
            concurrency,
            temperature,
            fit_cache,
            triage_only,
            judged_cache,
            out,
        } => {
            let api_base = api_base
                .as_deref()
                .unwrap_or_else(|| provider.default_base())
                .trim_end_matches('/')
                .to_string();
            anyhow::ensure!(!api_base.is_empty(), "API base URL cannot be empty");
            let triage_model =
                triage_model.unwrap_or_else(|| provider.default_triage_model().to_string());
            let model = model.unwrap_or_else(|| provider.default_description_model().to_string());
            let triage_cache_model = provider.cache_model(&api_base, &triage_model);
            let description_cache_model = provider.cache_model(&api_base, &model);

            let text =
                std::fs::read_to_string(&input).with_context(|| format!("reading {input}"))?;
            let names: Vec<String> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str::<ScanRow>(l).map(|r| r.name))
                .collect::<Result<_, _>>()
                .context("unexpected input JSONL format")?;
            ensure_valid_names(names.iter().map(String::as_str))?;
            let brief = std::fs::read_to_string(&brief)
                .with_context(|| format!("reading brief {brief}"))?;
            anyhow::ensure!(!names.is_empty(), "no names to judge");
            // Always print this: it is the description-cache key, and without it
            // there is no way to know what the cache contains.
            let fingerprint = prompt_id(&description_cache_model, &brief);
            eprintln!(
                "provider: {provider:?}, API: {api_base}, description prompt: {description_cache_model} / {fingerprint}"
            );

            let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

            // Pass 1 costs money. Write each result immediately so an interruption
            // never causes it to be purchased twice.
            let cached = load_fit_cache(&fit_cache, &triage_cache_model)?;
            let mut rated: Vec<Triage> = names
                .iter()
                .filter_map(|n| {
                    cached.get(n).map(|&fit| Triage {
                        name: n.clone(),
                        fit,
                    })
                })
                .collect();
            let pending: Vec<String> = names
                .iter()
                .filter(|n| !cached.contains_key(*n))
                .cloned()
                .collect();
            eprintln!(
                "pass 1: {} names, {} already scored, {} remaining in {} batches, model {}",
                names.len(),
                rated.len(),
                pending.len(),
                pending.len().div_ceil(batch),
                triage_model
            );

            // Build the client only when a request is actually needed. Requiring
            // a key for work fully served by cache would be needless friction.
            // Pass 2 will construct it later if its description cache is incomplete.
            let judge = if pending.is_empty() {
                None
            } else {
                Some(Arc::new(Judge::new(provider, &api_base, temperature)?))
            };

            let mut journal = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&fit_cache)
                    .with_context(|| format!("opening journal {fit_cache}"))?,
            );
            let mut set = tokio::task::JoinSet::new();
            for chunk in pending.chunks(batch) {
                let judge = judge
                    .clone()
                    .expect("client required when a batch remains to be scored");
                let (sem, brief) = (sem.clone(), brief.clone());
                let (chunk, model) = (chunk.to_vec(), triage_model.clone());
                set.spawn(async move {
                    let _p = sem.acquire_owned().await.expect("open semaphore");
                    judge.triage(&model, &brief, &chunk).await
                });
            }
            while let Some(res) = set.join_next().await {
                // A failed batch must not take the others down. Report it and
                // continue with the available results.
                match res.context("triage task interrupted")? {
                    Ok(done) => {
                        for t in &done {
                            writeln!(
                                journal,
                                "{}",
                                serde_json::json!({
                                    "name": t.name,
                                    "fit": t.fit,
                                    "model": triage_cache_model,
                                })
                            )?;
                        }
                        // Flush after every batch: a journal that waits until
                        // program exit protects nothing.
                        journal.flush()?;
                        rated.extend(done);
                    }
                    Err(e) => eprintln!("batch skipped: {e:#}"),
                }
            }
            anyhow::ensure!(!rated.is_empty(), "pass 1 returned no usable results");

            rated.sort_unstable_by(|a, b| b.fit.total_cmp(&a.fit));

            // Greedy selection by fit: accept a name only when it shares no
            // n-gram with a finalist already retained.
            let crowded = rated.len();
            if diversity > 0 {
                let mut taken: HashSet<Vec<u8>> = HashSet::new();
                let mut kept = Vec::with_capacity(rated.len());
                for r in rated {
                    let b = r.name.as_bytes();
                    let grams: Vec<Vec<u8>> = if b.len() < diversity {
                        Vec::new()
                    } else {
                        (0..=b.len() - diversity)
                            .map(|i| b[i..i + diversity].to_vec())
                            .collect()
                    };
                    if grams.iter().any(|g| taken.contains(g)) {
                        continue;
                    }
                    taken.extend(grams);
                    kept.push(r);
                }
                rated = kept;
            }
            eprintln!(
                "diversity: {} retained out of {} ({} near-duplicates rejected)",
                rated.len(),
                crowded,
                crowded - rated.len()
            );
            if triage_only {
                let path = out.as_deref().unwrap_or("out/selected.jsonl");
                let mut f = std::io::BufWriter::new(
                    std::fs::File::create(path).with_context(|| format!("creating {path}"))?,
                );
                // Write the entire selection, not just the finalists. Rank and
                // score let the cutoff move later without querying the model again.
                for (i, r) in rated.iter().enumerate() {
                    writeln!(
                        f,
                        r#"{{"rank":{},"name":"{}","fit":{},"finalist":{}}}"#,
                        i + 1,
                        r.name,
                        r.fit,
                        i < keep
                    )?;
                }
                f.flush()?;
                eprintln!(
                    "{} selected names written to {path}, including {} finalists",
                    rated.len(),
                    keep.min(rated.len())
                );
                return Ok(());
            }

            rated.truncate(keep);
            eprintln!(
                "pass 2: {} finalists (fit {:.1} to {:.1}), model {}",
                rated.len(),
                rated.last().map(|r| r.fit).unwrap_or(0.0),
                rated.first().map(|r| r.fit).unwrap_or(0.0),
                model
            );
            let finalists: Vec<String> = rated.iter().map(|r| r.name.clone()).collect();
            let cached_descriptions =
                load_judged(&judged_cache, &description_cache_model, &fingerprint)?;
            let to_describe: Vec<String> = finalists
                .iter()
                .filter(|n| !cached_descriptions.contains_key(*n))
                .cloned()
                .collect();
            if !cached_descriptions.is_empty() {
                eprintln!(
                    "{} descriptions loaded from cache, {} to request",
                    finalists.len() - to_describe.len(),
                    to_describe.len()
                );
            }
            // No descriptions needed: no key is required because everything is cached.
            let judge = match judge {
                Some(j) => Some(j),
                None if !to_describe.is_empty() => {
                    Some(Arc::new(Judge::new(provider, &api_base, temperature)?))
                }
                None => None,
            };
            let mut journal = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&judged_cache)
                    .with_context(|| format!("opening journal {judged_cache}"))?,
            );
            let mut sink: Box<dyn Write> = match &out {
                Some(p) => Box::new(std::io::BufWriter::new(
                    std::fs::File::create(p).with_context(|| format!("creating {p}"))?,
                )),
                None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
            };
            // Send batches together as in pass 1. Running them sequentially only
            // multiplies latency by the number of batches.
            let mut set = tokio::task::JoinSet::new();
            for chunk in to_describe.chunks(describe_batch) {
                let judge = judge.clone().expect("client required for descriptions");
                let (sem, brief) = (sem.clone(), brief.clone());
                let (chunk, model, cache_model) = (
                    chunk.to_vec(),
                    model.clone(),
                    description_cache_model.clone(),
                );
                set.spawn(async move {
                    let _p = sem.acquire_owned().await.expect("open semaphore");
                    judge.describe(&model, &cache_model, &brief, &chunk).await
                });
            }
            let mut written = 0usize;
            let mut failures = 0usize;
            // Emit cached descriptions immediately. There is no reason to wait
            // for the network for work that has already been paid for.
            for n in &finalists {
                if let Some(v) = cached_descriptions.get(n) {
                    writeln!(sink, "{}", serde_json::to_string(v)?)?;
                    written += 1;
                }
            }
            sink.flush()?;
            while let Some(res) = set.join_next().await {
                match res.context("description task interrupted")? {
                    Ok(v) => {
                        for verdict in &v {
                            let line = serde_json::to_string(verdict)?;
                            writeln!(sink, "{line}")?;
                            writeln!(journal, "{line}")?;
                            written += 1;
                        }
                        // Write on arrival, so file order follows batch completion
                        // rather than ranking.
                        sink.flush()?;
                        journal.flush()?;
                        eprintln!("{written}/{} described", finalists.len());
                    }
                    Err(e) => {
                        failures += 1;
                        eprintln!("description batch skipped: {e:#}");
                    }
                }
            }
            // The total alone is misleading when some entries came from cache.
            // Report what is missing so complete API failure cannot look successful.
            let missing = finalists.len() - written;
            if missing > 0 {
                eprintln!(
                    "{written} names described out of {} — {missing} missing, {failures} failed batch(es)",
                    finalists.len()
                );
            } else {
                eprintln!("{written} names described");
            }
            // Exiting with status 0 after all descriptions fail would imply
            // success and make an empty file look like a valid result.
            if written == 0 && !finalists.is_empty() {
                anyhow::bail!(
                    "no descriptions obtained for {} finalists; see errors above",
                    finalists.len()
                );
            }

            // The journal is written in arrival order so interruptions lose
            // nothing. Sort it when the run finishes: the file is also read by
            // humans, and severity is the useful ordering for choosing a name.
            drop(journal);
            if !to_describe.is_empty()
                && let Err(e) = sort_judgments(&judged_cache)
            {
                eprintln!("cache left in arrival order: {e:#}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_defaults_to_seven_letters() {
        let cli = Cli::try_parse_from(["bitter", "scan"]).unwrap();
        let Cmd::Scan { len, .. } = cli.cmd else {
            unreachable!("scan command expected");
        };
        assert_eq!(len, 7);
    }

    #[test]
    fn cli_rejects_dangerous_values() {
        assert!(Cli::try_parse_from(["bitter", "scan", "--len", "2"]).is_err());
        assert!(
            Cli::try_parse_from(["bitter", "check", "input.jsonl", "--concurrency", "0"]).is_err()
        );
        assert!(Cli::try_parse_from(["bitter", "judge", "input.jsonl", "--batch", "0"]).is_err());
        assert!(
            Cli::try_parse_from(["bitter", "judge", "input.jsonl", "--diversity", "0"]).is_err()
        );
    }

    #[test]
    fn validates_names_strictly() {
        assert!(ensure_valid_names(["alpha", "beta"]).is_ok());
        assert!(ensure_valid_names(["Alpha"]).is_err());
        assert!(ensure_valid_names(["alpha", "alpha"]).is_err());
    }

    #[test]
    fn loads_the_default_corpus() {
        let (_, words) = load_model(None).unwrap();
        assert!(!words.is_empty());
    }
}
