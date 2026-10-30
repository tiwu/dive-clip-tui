#!/usr/bin/env python3
"""
Minimal Ollama connection test script.

Loads config from config.yaml and sends a simple "hello world" prompt
to verify the Ollama server is reachable and the model is loaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ollama_client import resolve_host


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_ollama_connection(host: str, model: str, timeout: int = 30) -> bool:
    """
    Test Ollama connection with a simple generate request.

    Returns True if successful, False otherwise.
    """
    # Simple hello world prompt to test basic connectivity and model loading
    prompt = "Say hello in one short sentence."

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    url = f"{host.rstrip('/')}/api/generate"

    print("Testing Ollama connection...")
    print(f"  Host: {host}")
    print(f"  Model: {model}")
    print(f"  Prompt: {prompt!r}")
    print()

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        elapsed = time.perf_counter() - started

        # Extract the generated text
        generated_text = result.get("response", "").strip()

        print(f"✓ Connection successful! ({elapsed:.2f}s)")
        print(f"  Response: {generated_text!r}")

        # Also show some server metrics if available
        if "eval_count" in result:
            print(f"  Generated tokens: {result['eval_count']}")
        if "total_duration" in result:
            print(f"  Total duration: {result['total_duration'] / 1_000_000_000:.2f}s")

        return True

    except requests.exceptions.ConnectionError as exc:
        elapsed = time.perf_counter() - started
        print(f"✗ Connection failed after {elapsed:.2f}s")
        print(f"  Error: Could not connect to Ollama at {host}")
        print(f"  Details: {exc}")
        return False

    except requests.exceptions.Timeout as exc:
        elapsed = time.perf_counter() - started
        print(f"✗ Request timed out after {elapsed:.2f}s (timeout={timeout}s)")
        print(f"  Error: {exc}")
        return False

    except requests.exceptions.HTTPError as exc:
        elapsed = time.perf_counter() - started
        print(f"✗ HTTP error after {elapsed:.2f}s")
        print(f"  Status: {exc.response.status_code}")
        print(f"  Response: {exc.response.text[:500]}")
        return False

    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"✗ Unexpected error after {elapsed:.2f}s")
        print(f"  Error: {exc}")
        return False


def main() -> int:
    """Main entry point."""
    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: config.yaml not found in current directory")
        return 1
    except yaml.YAMLError as exc:
        print(f"Error: Failed to parse config.yaml: {exc}")
        return 1

    ollama_config = config.get("ollama", {})
    host = resolve_host(ollama_config.get("host"))
    model = ollama_config.get("model", "qwen2.5vl:3b")
    timeout = ollama_config.get("timeout_seconds", 30)

    if not model:
        print("Error: 'ollama.model' not configured in config.yaml")
        return 1

    success = check_ollama_connection(host, model, timeout)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
