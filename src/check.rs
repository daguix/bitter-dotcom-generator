//! Availability checking behind a trait, so the channel remains a configuration
//! choice rather than an architectural one.
//!
//! The default implementation queries Verisign RDAP: a documented, authoritative
//! public service that requires no registration. At a moderate rate over a
//! shortlist of a few tens of thousands of names, this is ordinary traffic,
//! unlike bulk authoritative-server queries, which amount to zone enumeration
//! and are rate-limited.
//!
//! Other channels (bulk registrar API, CZDS zone file) can be added by
//! implementing the same trait.

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    Free,
    Taken,
    /// Unusable response (outage, quota, unexpected format). Never treated as
    /// available: a false positive here would be costly at the end of the pipeline.
    Unknown,
}

#[async_trait]
pub trait AvailabilityChecker: Send + Sync {
    async fn check(&self, name: &str) -> Availability;
}

pub struct Rdap {
    http: reqwest::Client,
}

impl Rdap {
    pub fn new() -> Result<Rdap> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("bitter-dotcom-generator (domain name research)")
            .build()
            .context("building the HTTP client")?;
        Ok(Rdap { http })
    }
}

#[async_trait]
impl AvailabilityChecker for Rdap {
    async fn check(&self, name: &str) -> Availability {
        let url = format!("https://rdap.verisign.com/com/v1/domain/{name}.com");
        for attempt in 0..3 {
            match self.http.get(&url).send().await {
                Ok(r) if r.status() == 404 => return Availability::Free,
                Ok(r) if r.status().is_success() => return Availability::Taken,
                Ok(r) if r.status() == 429 || r.status().is_server_error() => {
                    // Exponential backoff: the quota may reopen, so do not retry quickly.
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt + 1))).await;
                }
                Ok(_) => return Availability::Unknown,
                Err(_) => tokio::time::sleep(Duration::from_secs(1 << attempt)).await,
            }
        }
        Availability::Unknown
    }
}

/// Checks a list of names at a bounded rate.
///
/// Throughput is controlled by a token producer rather than concurrency alone:
/// concurrency bounds the number of requests *in flight*, not their frequency.
/// Without this gate, fast responses would produce bursts.
pub async fn check_all<C: AvailabilityChecker + 'static>(
    checker: Arc<C>,
    names: Vec<String>,
    per_second: u32,
    concurrency: usize,
    mut on_result: impl FnMut(&str, Availability) -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(per_second > 0, "rate must be greater than zero");
    anyhow::ensure!(concurrency > 0, "concurrency must be greater than zero");
    let (tx, mut rx) = mpsc::channel::<()>(1);

    let ticker = tokio::spawn(async move {
        let mut iv = tokio::time::interval(Duration::from_secs_f64(1.0 / per_second as f64));
        iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            iv.tick().await;
            if tx.send(()).await.is_err() {
                break; // all consumers have finished
            }
        }
    });

    let mut names = names.into_iter();
    let mut tasks = JoinSet::new();
    let mut exhausted = false;

    loop {
        // Keep only a small window in flight. Once full, consume and journal a
        // result before proceeding.
        while !exhausted && tasks.len() < concurrency {
            let Some(name) = names.next() else {
                exhausted = true;
                break;
            };
            if rx.recv().await.is_none() {
                exhausted = true;
                break;
            }
            let checker = checker.clone();
            tasks.spawn(async move {
                let verdict = checker.check(&name).await;
                (name, verdict)
            });
        }

        let Some(result) = tasks.join_next().await else {
            break;
        };
        let (name, verdict) = result.context("availability-check task interrupted")?;
        on_result(&name, verdict)?;
    }
    ticker.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Immediate {
        started: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AvailabilityChecker for Immediate {
        async fn check(&self, _name: &str) -> Availability {
            self.started.fetch_add(1, Ordering::SeqCst);
            Availability::Free
        }
    }

    #[tokio::test]
    async fn delivers_results_before_scheduling_the_entire_list() {
        let started = Arc::new(AtomicUsize::new(0));
        let checker = Arc::new(Immediate {
            started: started.clone(),
        });
        let names = ["alpha", "beta", "gamma"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut delivered = 0;

        check_all(checker, names, 10_000, 2, |_, _| {
            delivered += 1;
            if delivered == 1 {
                assert!(started.load(Ordering::SeqCst) < 3);
            }
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(delivered, 3);
    }

    #[tokio::test]
    async fn rejects_zero_concurrency() {
        let checker = Arc::new(Immediate {
            started: Arc::new(AtomicUsize::new(0)),
        });
        assert!(
            check_all(checker, Vec::new(), 1, 0, |_, _| Ok(()))
                .await
                .is_err()
        );
    }
}
