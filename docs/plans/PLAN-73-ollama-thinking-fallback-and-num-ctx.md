# Architecture Plan: [Issue #73] feat: resilient Ollama thinking fallback and configurable context length (num_ctx)

- **Issue**: [#73](https://github.com/tiwu/dive-clip-tui/issues/73)
- **Status**: Plan Review
- **Author**: Antigravity
- **Date**: 2026-11-04
- **Target Branch**: `feature/issue-73-ollama-thinking-fallback-and-num-ctx`

---

## 1. Problem Statement & Motivation

When using reasoning-capable vision-language models in Ollama (such as `qwen3-vl:4b-64k`, `deepseek-r1` derivatives, or models built from custom thinking Modelfiles), inference runs encounter two main pain points:

### 1. JSON Parse Errors on Thinking Models
When requesting structured JSON with `"format": "json"`, models with reasoning templates often output their entire generated JSON response into Ollama's `"thinking"` key while returning an empty string `""` in `"response"`:
```json
{
  "model": "qwen3-vl:4b-64k",
  "response": "",
  "thinking": "{\n  \"domain\": \"underwater\",\n  \"confidence\": 0.98,\n  \"usable\": true\n}",
  "done": true
}
```
In `utils/ollama_client.py` (lines 204-205):
```python
raw = result.get("response", "{}")
parsed = json.loads(raw)
```
Because `"response"` is present in the dictionary as an empty string `""`, `result.get("response", "{}")` evaluates to `""`, causing `json.loads("")` to raise `json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` and abort the inference pass.

Furthermore, testing against Ollama reveals that passing `"think": false` in the request payload does not prevent models with hardcoded thinking templates from routing tokens to `"thinking"`. A parser fallback is necessary.

### 2. Context Length Tuning Requires Modelfile Duplication
Experimenting with different context window sizes (e.g. 16k, 32k, 64k) to optimize memory usage and avoid VRAM exhaustion currently requires creating and maintaining separate Ollama Modelfiles for every variation (e.g. `PARAMETER num_ctx 65536`). Ollama natively supports passing runtime options including `num_ctx` in the `/api/generate` payload:
```json
{
  "options": {
    "num_ctx": 32768
  }
}
```
Currently, `dive-clip` has no configuration setting for `num_ctx` in `config.yaml`, nor does `OllamaClient` pass options to `/api/generate`.

### 3. VLM and Ollama Client Synergy
`utils/ollama_client.py` and `utils/vlm_client.py` share duplicate code patterns:
- Image file reading and base64-encoding (`with open(image_path, "rb") ...`).
- Session management, pooling, context managers (`__enter__`, `__exit__`, `close()`), and metrics initialization.
- JSON extraction from LLM prose/markdown: `vlm_client.py` has `extract_json_object()`, while `ollama_client.py` only did raw `json.loads(raw)`.
Unifying these helpers into a shared interface eliminates code duplication and makes JSON parsing resilient across both local Ollama models and cloud OpenRouter models.

---

## 2. Requirements & Acceptance Criteria

### Shared VLM Foundation & Resilient JSON Parsing
- [ ] **Generic Multi-Candidate `extract_json_object(*candidates)`**:
  - Generalize `extract_json_object(*candidates: str | None) -> dict[str, Any] | None` to inspect multiple candidate strings in priority order (e.g. `response`, then `thinking`).
  - For each non-empty string candidate, attempt direct `json.loads()`.
  - If direct `json.loads()` fails (e.g. model output wrapped in markdown code blocks ` ```json ... ``` ` or accompanied by conversational reasoning), extract the outer `{...}` block using regex matching.
  - Return the first successfully parsed dictionary, or `None` if all candidates fail.
  - Re-export `extract_json_object` from `utils/vlm_client.py` to maintain backwards compatibility.
- [ ] **Shared Image Base64 Encoder (`encode_image_b64`)**:
  - Implement a common `encode_image_b64(image_path: str) -> str` helper shared by all VLM backends.
- [ ] **Base VLM Client (`BaseVLMClient`)**:
  - Define a common base class for `OllamaClient` and `OpenRouterClient` handling `model`, `timeout`, `metrics`, `session`, `close()`, `__enter__`, and `__exit__`.
- [ ] **Send `think: false` in Ollama Request Payload**:
  - Include `"think": false` in the Ollama `/api/generate` payload to disable unnecessary reasoning tokens on models that support it.

### Configurable Model Context Length (`num_ctx`)
- [ ] **Config Schema**:
  - Add optional `num_ctx` parameter under `ollama:` in `config.yaml` (documented with comments).
- [ ] **`OllamaClient` Initialization**:
  - Accept `num_ctx: int | None = None` in `OllamaClient.__init__()`.
- [ ] **Forward to `/api/generate`**:
  - When `self.num_ctx` is set, populate `"options": {"num_ctx": self.num_ctx}` in the POST payload sent to `/api/generate`.
  - When `self.num_ctx` is `None`, omit the `"options"` key.
- [ ] **Factory & Pipeline Wiring**:
  - Update `create_vlm_client(config)` in `utils/vlm_client.py` to forward `ollama_cfg.get("num_ctx")` to `OllamaClient`.
  - Update compatibility wrapper `call_ollama()` in `scripts/02_scan_footage.py` to accept and forward `num_ctx`.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add resilient Ollama JSON parsing with automatic fallback to the `thinking` field, unified multi-candidate JSON extraction across VLM clients, shared `BaseVLMClient` foundation, and configurable model context length via `ollama.num_ctx` in `config.yaml`.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - In `tests/test_vlm_client.py`:
     - Test `extract_json_object()` with multiple candidate strings (e.g. empty response + valid thinking; invalid response + valid thinking; markdown-fenced JSON in thinking).
     - Test that `create_vlm_client({"ollama": {"num_ctx": 16384}})` passes `num_ctx=16384` to `OllamaClient`.
   - In `tests/test_ollama_client.py`:
     - Test that `generate()` successfully parses JSON returned in `"thinking"` when `"response"` is empty string `""`.
     - Test that `generate()` parses JSON when `"response"` is whitespace or invalid JSON and `"thinking"` contains JSON.
     - Test that `generate()` extracts JSON embedded within markdown code blocks inside `"thinking"`.
     - Test that `generate()` passes `"think": false` in the `/api/generate` payload.
     - Test that `generate()` includes `"options": {"num_ctx": 32768}` when initialized with `num_ctx=32768`.
     - Test that `generate()` omits `"options"` when `num_ctx` is `None`.
     - Test `BaseVLMClient` session pooling and context manager inheritance.

2. **Green (Implementation)**:
   - Implement `BaseVLMClient`, `encode_image_b64`, and generalized `extract_json_object` in `utils/ollama_client.py` / `utils/vlm_client.py`.
   - Refactor `OllamaClient` and `OpenRouterClient` to inherit from `BaseVLMClient` and use the shared helpers.
   - Update `OllamaClient.generate()` to use `extract_json_object(result.get("response"), result.get("thinking"))` and pass `num_ctx` options.
   - Update `create_vlm_client()` in `utils/vlm_client.py`.
   - Update `scripts/02_scan_footage.py` compatibility wrapper.
   - Update `config.yaml` and `README.md`.

3. **Refactor & Quality**:
   - Run focused pytest suite: `uv run python -m pytest tests/test_ollama_client.py tests/test_vlm_client.py`.
   - Run full test suite: `uv run python -m pytest`.
   - Run Ruff check and format: `uv run ruff check .` and `uv run ruff format --check .`.

---

## 5. Verification & Rollout Checklist

- [ ] Unit tests pass: `uv run python -m pytest tests/test_ollama_client.py tests/test_vlm_client.py`
- [ ] Full regression suite passes: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks: `uv run ruff check .` and `uv run ruff format --check .`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Documentation updated in `config.yaml` and `README.md`
- [ ] PR opened with `semver:minor` closing issue #73
