# bitter-dotcom-generator

A Rust command-line pipeline for generating short, pronounceable `.com` domain
names from a domain-specific vocabulary.

The program explores possible names, filters out weak or overly familiar ones,
checks `.com` availability through RDAP, and can use OpenAI or local Ollama
models to rank and describe the remaining candidates against a positioning
brief.

## How it works

The pipeline has three main stages:

1. **Generate and filter** — an order-3 character Markov model learns the sound
   of the seed corpus. The scanner explores `[a-z]^L`, prunes low-probability
   prefixes, rejects names that fail a syllable-based pronunciation test, and
   removes names that are too close to a seed.
2. **Check availability** — the checker queries Verisign's public RDAP service
   at a bounded rate. Definite results are cached so interrupted or repeated
   runs do not request the same domain again.
3. **Judge candidates** — an LLM first assigns fit scores in batches. A second
   pass writes taglines, rationales, and concerns for a smaller,
   diversity-filtered set of finalists. The LLM can run through OpenAI or Ollama.

The Markov score is intentionally hidden from the LLM. Generation quality and
business relevance are evaluated independently.

## Requirements

- A recent stable Rust toolchain with Cargo
- Network access for RDAP checks
- An OpenAI API key or a local Ollama installation for the optional `judge` stage

Install Rust through [rustup](https://rustup.rs/), clone this repository, then
build the optimized binary from its directory:

```bash
cd bitter-dotcom-generator
cargo build --release
```

You can run the binary directly from
`target/release/bitter-dotcom-generator`, or use `cargo run --release --` as in
the examples below.

## Quick start

Create local directories for generated data:

```bash
mkdir -p cache out
```

Generate up to 5,000 six-letter candidates:

```bash
cargo run --release -- scan \
  --len 6 \
  --top 5000 \
  --out out/candidates.jsonl
```

Check their `.com` availability:

```bash
cargo run --release -- check out/candidates.jsonl \
  --out out/available.jsonl
```

Rank and describe the available names:

```bash
export OPENAI_API_KEY="your-api-key"

cargo run --release -- judge out/available.jsonl \
  --out out/judgments.jsonl
```

Both `cache/` and `out/` are ignored by Git because they may contain private,
costly, or project-specific run artifacts.

## OpenAI API key

The recommended setup is the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

The variable applies to the current shell and is read automatically by the
program. Never put a real key in source code, committed configuration, examples,
or command-line options. OpenAI recommends environment variables or a secret
manager for API keys; see the
[official API key safety guidance](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

On Linux and macOS, the program also supports a persistent local key file:

```bash
install -d -m 700 ~/.config/bitter-dotcom-generator
read -rsp "OpenAI API key: " key; echo
printf '%s\n' "$key" > ~/.config/bitter-dotcom-generator/openai-key
chmod 600 ~/.config/bitter-dotcom-generator/openai-key
unset key
```

This file lives outside the repository. `OPENAI_API_KEY` takes precedence when
both methods are present.

## Ollama

[Ollama](https://ollama.com/) can run the entire `judge` stage locally through
its OpenAI-compatible Chat Completions endpoint. No OpenAI key is needed.

Install Ollama, pull the default model, and make sure its local service is
running:

```bash
ollama pull gpt-oss:20b
ollama serve
```

In another terminal, select the Ollama provider:

```bash
cargo run --release -- judge out/available.jsonl \
  --provider ollama \
  --out out/judgments.jsonl
```

The Ollama defaults are:

- API base: `http://localhost:11434/v1`
- triage model: `gpt-oss:20b`
- description model: `gpt-oss:20b`

Use any installed model by overriding both model options:

```bash
cargo run --release -- judge out/available.jsonl \
  --provider ollama \
  --triage-model qwen3:8b \
  --model qwen3:8b \
  --out out/judgments.jsonl
```

For a remote Ollama server, pass its OpenAI-compatible base URL with
`--api-base`. If that server requires Bearer authentication, set
`OLLAMA_API_KEY`:

```bash
export OLLAMA_API_KEY="your-ollama-key"

cargo run --release -- judge out/available.jsonl \
  --provider ollama \
  --api-base https://ollama.com/v1 \
  --out out/judgments.jsonl
```

Ollama documents its Chat Completions compatibility and local endpoint in its
[OpenAI compatibility guide](https://ollama.com/blog/openai-compatibility).

## Commands

### `scan`

Exhaustively scans six- or seven-letter names and writes JSONL records containing
`name` and `score`.

```bash
cargo run --release -- scan [OPTIONS]
```

Important options:

- `--len <6|7>`: candidate length; default `6`
- `--threshold <FLOAT>`: minimum average log probability; higher is stricter
- `--top <N>`: maximum number of retained candidates; default `50000`
- `--min-syl <N>` / `--max-syl <N>`: accepted syllable range
- `--novelty <N>`: reject names sharing an N-gram with the seed corpus
- `--min-edit <N>`: reject names fewer than N edits away from a seed
- `--out <FILE>`: write JSONL to a file instead of standard output

Scanning is CPU-intensive. Release mode is strongly recommended.

### `score`

Prints Markov scores for specific lowercase names. This is useful for calibrating
`scan --threshold`.

```bash
cargo run --release -- score centroid voronoi
```

### `check`

Reads `scan` JSONL, checks `.com` availability, and writes only available names.

```bash
cargo run --release -- check out/candidates.jsonl [OPTIONS]
```

Important options:

- `--rate <N>`: maximum request rate; default `20` requests per second
- `--concurrency <N>`: maximum simultaneous requests; default `8`
- `--limit <N>`: check only the first N candidates
- `--cache <FILE>`: persistent RDAP journal; default `cache/rdap.jsonl`
- `--out <FILE>`: output JSONL file

`free` and `taken` responses are cached. Indeterminate responses are retried on
later runs rather than being treated as available.

### `judge`

Reads available-name JSONL and runs a two-pass OpenAI evaluation.

```bash
cargo run --release -- judge out/available.jsonl [OPTIONS]
```

Important options:

- `--provider <openai|ollama>`: LLM backend; default `openai`
- `--api-base <URL>`: override the selected provider's API base URL
- `--brief <FILE>`: company positioning brief; default `data/brief.txt`
- `--triage-model <MODEL>`: model used for bulk fit scoring; provider-dependent default
- `--model <MODEL>`: model used for finalist descriptions; provider-dependent default
- `--batch <N>`: names per triage request; default `100`
- `--describe-batch <N>`: names per description request; default `5`
- `--keep <N>`: number of finalists to describe; default `40`
- `--diversity <N>`: n-gram diversity constraint between finalists
- `--concurrency <N>`: concurrent API requests; default `4`
- `--triage-only`: stop after scoring and write the ranked selection
- `--fit-cache <FILE>`: pass-1 cache; default `cache/fit.jsonl`
- `--judged-cache <FILE>`: description cache; default
  `cache/judgments.jsonl`
- `--out <FILE>`: output JSONL file

Model names can be overridden because availability depends on the provider and
may change over time. OpenAI API usage is billed to the key owner; local Ollama
runs consume the user's compute resources. Review batch sizes, finalist count,
and selected models before a large run.

## Customizing the inputs

`data/seeds.txt` contains one lowercase seed per line. It defines the vocabulary
whose character patterns the Markov model learns. Use another file globally with
`--corpus`:

```bash
cargo run --release -- --corpus path/to/seeds.txt scan --len 6
```

`data/brief.txt` describes the company and products for the LLM evaluation. Edit
it directly or select another brief with `judge --brief`.

Changing the corpus changes generation. Changing the brief, description model,
or description prompt invalidates the corresponding cached judgments by design.

## Output and caches

All pipeline files use one JSON object per line. This makes long runs appendable,
streamable, and recoverable after interruption.

- `cache/rdap.jsonl`: known domain-availability results
- `cache/fit.jsonl`: LLM triage scores, keyed by model
- `cache/judgments.jsonl`: finalist descriptions, keyed by model and prompt
- `out/`: disposable or private run outputs

Do not publish these directories unless you have deliberately reviewed their
contents.

## Development

Run the project checks with:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

## Limitations

- An available domain is not necessarily legally safe to use. Perform a proper
  trademark and prior-rights search before adopting a name.
- RDAP availability can change at any time and should be confirmed with a
  registrar before purchase.
- LLM judgments are subjective and may be incomplete or wrong.
- The syllable filter is heuristic and favors English-like written forms.
- The LLM client targets the OpenAI Chat Completions protocol as implemented by
  OpenAI and Ollama; provider-specific APIs are not used.
