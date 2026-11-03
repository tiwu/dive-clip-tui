# Architecture Plan: [Issue #66] feat: support cloud VLM models

- **Issue**: [#66](https://github.com/tiwu/dive-clip-tui/issues/66)
- **Status**: Approved
- **Author**: tiwu
- **Date**: 2026-11-03
- **Target Branch**: `feature/issue-66-cloud-vlm-support`

---

## 1. Problem Statement & Motivation

The scan pipeline (`scripts/02_scan_footage.py`) requires a local Ollama server for all
VLM calls: Pass 0 (clip classification), Pass 1 (coarse scan) and Pass 2 (fine scan).
Local-first stays the default, but users should be able to send frames to a cloud VLM
instead, primarily through OpenRouter (e.g. `nvidia/nemotron-3.5-super-vl:free`).

`scripts/04_plan_storyline.py` is out of scope. It sends text-only detections to a text
LLM and never sends images, so it is not a VLM call site. Adding OpenRouter as a text
provider there can be a separate issue.

---

## 2. Requirements & Acceptance Criteria

- [ ] A `vlm` config section selects the scan backend: `ollama` (default) or `openrouter`
- [ ] `utils/vlm_client.py` defines a `VLMClient` protocol, with `OllamaClient` conforming to it and a new `OpenRouterClient`
- [ ] `OpenRouterClient` reads `OPENROUTER_API_KEY` from the environment and fails fast with a clear error when it is missing
- [ ] `scripts/02_scan_footage.py` builds its client through one factory and uses it for Pass 0, 1 and 2
- [ ] Per-phase metrics (calls, successes, failures, elapsed, tokens) work for both backends
- [ ] Client-side rate limit (`requests_per_minute`) and retry with backoff on HTTP 429/5xx
- [ ] Non-JSON replies are recovered by extracting the first JSON object; otherwise the call counts as a failure and returns `None`
- [ ] Automated tests cover new behavior with ≥90% statement coverage
- [ ] Documentation updated (`config.yaml` comments, `README.md`)
- [ ] `CHANGELOG.md` entry under `[Unreleased]`

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add OpenRouter as a cloud VLM backend for the scan pipeline, selected with the new `vlm` config section (`provider: ollama | openrouter`). Ollama stays the default.

---

## 4. Technical Design

### 4.1 Client abstraction (`utils/vlm_client.py`)

`OllamaClient` and `OllamaMetrics` in `utils/ollama_client.py` already carry per-phase
stats that `02_scan_footage.py` and the stats output depend on. Keep that shape:

- `VLMClient` (`typing.Protocol`): `generate(image_path, prompt, phase="unspecified", system_prompt=None) -> dict | None`, plus a `metrics` attribute with `record(...)` and `format_summary()`.
- `OllamaClient` is unchanged and already satisfies the protocol.
- `OpenRouterClient` reuses `OllamaMetrics` and `OllamaRequestStats`, mapping the OpenRouter `usage` token counts onto the existing token counters. No cost tracking.
- `create_vlm_client(config) -> VLMClient` picks the backend from `vlm.provider`. The three `OllamaClient(...)` call sites in `02_scan_footage.py` (lines ~112, ~752, ~962, ~2187) go through it.

`generate` takes a file path because frames are extracted to the work dir (`/dev/shm`) before inference. No in-memory input path is needed.

### 4.2 Config (`config.yaml`)

The existing `ollama` section stays as is, so current configs keep working. A missing `vlm` section means `provider: ollama`.

```yaml
vlm:
  provider: "ollama"                      # "ollama" or "openrouter"
  # OpenRouter only. The API key comes from the OPENROUTER_API_KEY env var.
  openrouter:
    model: "nvidia/nemotron-3.5-super-vl:free"
    timeout_seconds: 120
    requests_per_minute: 20
    max_retries: 3
```

Ollama host, model and timeout keep coming from the `ollama` section.

### 4.3 OpenRouter request

- `POST https://openrouter.ai/api/v1/chat/completions`, header `Authorization: Bearer $OPENROUTER_API_KEY`.
- Body: `model`, `messages` (optional system message, then a user message with a text part and an `image_url` part holding a `data:image/jpeg;base64,...` URL), `response_format: {"type": "json_object"}`.
- Frames are the JPEGs already scaled by `coarse_scale` / `fine_scale`, which keeps payloads small. Do not re-encode or upscale them.
- Read `choices[0].message.content` and parse it as JSON. If parsing fails, extract the first `{...}` block and retry the parse once. If that also fails, log a warning, record a failure and return `None`, matching `OllamaClient`.
- `response_format` is a request, not a guarantee. Some models ignore it, which is why the extraction fallback exists.
- Rate limit: a sliding-window limiter allowing `requests_per_minute` calls. On 429 or 5xx, retry up to `max_retries` times with exponential backoff, honoring `Retry-After` when present.
- Concurrency stays sequential, as in the current scan loop. Parallel calls are out of scope.

---

## 5. Test Plan

Write the tests first. HTTP is mocked; no test touches the network.

`tests/test_vlm_client.py`:
- Factory returns `OllamaClient` when `vlm` is absent or `provider: ollama`, and `OpenRouterClient` for `openrouter`.
- Factory raises a clear error for an unknown provider and for `openrouter` with no `OPENROUTER_API_KEY`.
- Request payload: data URL image part, `response_format`, optional system message, auth header.
- Valid JSON reply is parsed and returned; metrics record one success with token counts.
- JSON wrapped in prose or a code fence is recovered by the extraction fallback.
- Unparseable reply, unreadable image, and HTTP error each return `None` and record a failure.
- 429 then 200 succeeds after one retry; repeated 429 gives up after `max_retries`; `Retry-After` is honored (patch `time.sleep`).
- Rate limiter delays the call that would exceed `requests_per_minute` (patch the clock).

`tests/test_scan_footage.py` (or the existing scan tests):
- A scan run with `provider: openrouter` and a fake client passes Pass 0, 1 and 2 through that client.
- Existing Ollama-based tests pass unchanged.

---

## 6. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR has no conflicting SemVer labels (unlabeled defaults to minor)
- [ ] PR submitted with `Closes #66` and plan linked
- [ ] GitHub issue label updated to `agent:plan-review`

---

## 7. Resolved Questions

1. **Frame privacy**: document it explicitly in the README and log a startup warning when a cloud provider is active.
2. **Free-tier limits**: the user handles quotas. No extra call budget or guards.
3. **Prompt fit**: confirm the JSON schema from `normalize_vlm_frame_response` holds on one OpenRouter model. Needs a real `OPENROUTER_API_KEY`, so it is a manual check before release.
4. **Cost visibility**: out of scope.
5. **Other providers**: Gemini, Anthropic and OpenAI direct clients are deferred. The protocol and factory make them additive.
