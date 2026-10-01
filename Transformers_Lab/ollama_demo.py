"""Query a local Ollama server from Python.

Worksheet: Run_Ollama.ipynb.

Requirements (one-time):
  1. Install Ollama from https://ollama.com/download
  2. Pull a model, e.g.    ollama pull mistral
  3. Make sure the Ollama server is running (it starts automatically on
     Mac after the first launch; otherwise run `ollama serve`).

Usage:
  python3 ollama_demo.py "Who was Sir Isaac Newton?"
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"


def ask_ollama(question: str, model: str = DEFAULT_MODEL) -> str:
    body = json.dumps({
        "model": model,
        "prompt": f"Question: {question}\n\nAnswer: Let's think step by step.",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL}: {exc}.\n"
            f"Is `ollama serve` running and have you pulled `{model}`?"
        )
    return payload["response"].strip()


if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Who was Sir Isaac Newton?"
    )
    print(f"question: {question}")
    print()
    print(ask_ollama(question))
