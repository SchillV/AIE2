"""
Chatbot backend for the IDR/RON Forecaster.

Two LLM backends:
  - Claude : Anthropic API  (requires anthropic package + API key)
  - Local  : Ollama          (requires ollama package + running Ollama daemon)

The chatbot is strictly scoped to app data — no external access.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd

# ── Default model identifiers ─────────────────────────────────────────────────

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
]
DEFAULT_OLLAMA_MODEL = "llama3.2"

# Tool name constants — display
TOOL_SHOW_FORECAST = "show_forecast_chart"
TOOL_SHOW_COMPARISON = "show_model_comparison"
TOOL_SHOW_DIAGNOSTICS = "show_diagnostic_plots"
TOOL_GET_DATA = "get_data_table"
# Tool name constants — app actions
TOOL_FETCH_DATA = "fetch_bnr_data"
TOOL_RETRAIN_ALL = "retrain_all_models"
TOOL_RETRAIN_MODEL = "retrain_model"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    actions: list[dict] = field(default_factory=list)


@dataclass
class ChatResponse:
    text: str
    actions: list[dict] = field(default_factory=list)


# ── System prompt / context builder ──────────────────────────────────────────

_DISP = {
    "ARIMA": "ARIMA",
    "ExponentialSmoothing": "Exponential Smoothing",
    "Naive": "Naive (last value)",
    "NaiveDrift": "Naive + Drift",
}

_MODEL_KEYS = ("ARIMA", "ExponentialSmoothing", "Naive", "NaiveDrift")


def build_app_context(
    series: pd.Series | None,
    all_results: dict | None,
) -> str:
    """
    Build the system-prompt context block from current app data.
    Called once per chat response; data is read from cached session state.
    """
    lines: list[str] = [
        "You are an expert assistant embedded in an IDR/RON exchange rate forecasting app.",
        "The app scrapes official BNR (National Bank of Romania) rates and trains four",
        "time-series models: ARIMA, Exponential Smoothing, Naive, and Naive+Drift.",
        "",
        "CONSTRAINTS:",
        "  - Do NOT search the internet or fabricate data not present in this context.",
        "  - Do NOT call external services other than the app tools listed at the end.",
        "  - If information is not in this context and no tool can retrieve it, say so.",
        "  - You CAN call app tools to display charts, fetch BNR data, and retrain models.",
        "  - Only call action tools (fetch_bnr_data, retrain_*) when explicitly asked.",
        "",
        "CRITICAL — tool use is mandatory for operations:",
        "  Responding with text alone (e.g. 'I have fetched the data') does NOT perform",
        "  any operation. The fetch_bnr_data, retrain_all_models, and retrain_model tools",
        "  cause real side effects; they MUST be called via tool_use for anything to happen.",
        "  Never claim to have fetched data or retrained a model without calling the tool.",
        "",
        "═══ EXCHANGE RATE DATA ══════════════════════════════════════════════════════",
    ]

    if series is None:
        lines.append("Status: NOT AVAILABLE — data has not been fetched yet.")
    else:
        n = len(series)
        first = series.index[0].date()
        last = series.index[-1].date()
        latest = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if n >= 2 else latest
        daily_chg = latest - prev
        week_ago = float(series.iloc[-6]) if n >= 6 else float("nan")
        week_chg = (latest - week_ago) if not np.isnan(week_ago) else float("nan")

        lines += [
            f"Unit        : 100 IDR → RON (Indonesian Rupiah to Romanian Leu)",
            f"Observations: {n:,}  ({first} to {last})",
            f"Latest rate : {latest:.4f} RON per 100 IDR  (date: {last})",
            f"Daily change: {daily_chg:+.4f} RON",
            f"7-day change: {week_chg:+.4f} RON" if not np.isnan(week_chg) else "",
            f"All-time avg: {series.mean():.4f}",
            f"All-time min: {series.min():.4f}  (on {series.idxmin().date()})",
            f"All-time max: {series.max():.4f}  (on {series.idxmax().date()})",
            "",
            "Last 15 business days (date → rate):",
        ]
        for dt, v in series.iloc[-15:].items():
            lines.append(f"  {dt.date()}  →  {v:.4f}")

    lines += [
        "",
        "═══ MODEL RESULTS ═══════════════════════════════════════════════════════════",
    ]

    if all_results is None:
        lines.append("Status: NOT AVAILABLE — models have not been trained yet.")
    else:
        best_name = all_results.get("best", {}).get("model", "")
        ts = all_results.get("timestamp", "unknown")
        n_obs = all_results.get("n_observations", "?")
        lines += [f"Trained: {ts}  on {n_obs} observations", ""]

        for key in _MODEL_KEYS:
            res = all_results.get(key)
            if res is None:
                continue
            star = " ★ BEST" if key == best_name else ""
            mae = res.get("mean_mae", float("nan"))
            std = res.get("std_mae", float("nan"))
            disp = _DISP.get(key, key)
            extra = ""
            if key == "ARIMA":
                aic = res.get("aic")
                extra = f"  order={res.get('order')}  AIC={aic:.2f}" if aic else f"  order={res.get('order')}"
            elif key == "ExponentialSmoothing":
                extra = f"  trend={res.get('trend')}  damped={res.get('damped_trend')}"
            lines.append(f"  {disp}{extra}")
            lines.append(f"    CV MAE: {mae:.6f} ± {std:.6f}{star}")
            folds = res.get("fold_maes", [])
            if folds:
                lines.append(f"    Per-fold: {', '.join(f'{v:.5f}' for v in folds)}")

        lines += ["", f"Best model: {best_name} ({_DISP.get(best_name, best_name)})"]

    lines += [
        "",
        "═══ TOOLS ═══════════════════════════════════════════════════════════════════",
        "Visualisation tools (render inside the chat):",
        "  show_forecast_chart(model_name)      – interactive forecast + CI chart",
        "  show_model_comparison()              – model performance comparison table",
        "  show_diagnostic_plots(model_name)    – residuals / CV fold diagnostics",
        "  get_data_table(n_rows)               – recent exchange rate observations",
        "",
        "App action tools (perform real operations on the app's data):",
        "  fetch_bnr_data()          – scrape and save the latest IDR/RON rates from BNR",
        "  retrain_all_models()      – retrain all four models (takes 15–30 minutes)",
        "  retrain_model(model_name) – retrain one model only",
        "",
        "Valid model_name values: ARIMA, ExponentialSmoothing, Naive, NaiveDrift",
    ]

    return "\n".join(lines)


# ── Claude tool definitions ───────────────────────────────────────────────────

CLAUDE_TOOLS: list[dict] = [
    {
        "name": TOOL_SHOW_FORECAST,
        "description": (
            "Render an interactive forecast chart for one model. Shows historical data, "
            "a retroactive rolling one-step-ahead prediction, and a 95% confidence band."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "enum": list(_MODEL_KEYS),
                    "description": "Which model's forecast to display.",
                }
            },
            "required": ["model_name"],
        },
    },
    {
        "name": TOOL_SHOW_COMPARISON,
        "description": "Render the model performance comparison table (all four models, ranked by CV MAE).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": TOOL_SHOW_DIAGNOSTICS,
        "description": (
            "Render a 2×2 diagnostic figure for one model: residuals histogram, "
            "residuals over time, MAE per CV fold, actual vs predicted scatter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "enum": list(_MODEL_KEYS),
                }
            },
            "required": ["model_name"],
        },
    },
    {
        "name": TOOL_GET_DATA,
        "description": "Return the last N exchange rate observations as a readable table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_rows": {
                    "type": "integer",
                    "description": "How many recent rows to return (1–100).",
                    "default": 20,
                }
            },
        },
    },
    {
        "name": TOOL_FETCH_DATA,
        "description": (
            "Fetch the latest IDR/RON exchange rate data from the BNR website (cursbnr.ro) "
            "and save it to the local CSV file. Use this when the user asks to update, "
            "refresh, or download exchange rate data. This makes a real network request."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": TOOL_RETRAIN_ALL,
        "description": (
            "Retrain all four forecasting models (ARIMA, Exponential Smoothing, Naive, "
            "Naive+Drift) using the current data. This is a long operation (15–30 min). "
            "Only call when the user explicitly requests a full retrain."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": TOOL_RETRAIN_MODEL,
        "description": (
            "Retrain a single forecasting model. Faster than retraining all models. "
            "Only call when the user explicitly requests retraining a specific model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "enum": list(_MODEL_KEYS),
                    "description": "Which model to retrain.",
                }
            },
            "required": ["model_name"],
        },
    },
]


# ── Tool executor (shared by both backends) ───────────────────────────────────

def _execute_tool(
    name: str,
    inputs: dict,
    series: pd.Series | None,
    all_results: dict | None,
) -> tuple[dict | None, str]:
    """
    Execute a tool call. Returns (action_dict | None, result_text_for_llm).
    action_dict is rendered by the Streamlit layer; result_text is fed back to the LLM.
    """
    if name == TOOL_SHOW_FORECAST:
        model_name = inputs.get("model_name", "ARIMA")
        if all_results and model_name in all_results:
            return {"type": "show_forecast", "model_name": model_name}, \
                   f"Forecast chart for {model_name} rendered."
        return None, f"No trained results found for {model_name}."

    if name == TOOL_SHOW_COMPARISON:
        if all_results:
            return {"type": "show_comparison"}, "Model comparison table rendered."
        return None, "No model results available yet."

    if name == TOOL_SHOW_DIAGNOSTICS:
        model_name = inputs.get("model_name", "ARIMA")
        if all_results and model_name in all_results:
            return {"type": "show_diagnostics", "model_name": model_name}, \
                   f"Diagnostic plots for {model_name} rendered."
        return None, f"No trained results found for {model_name}."

    if name == TOOL_GET_DATA:
        if series is None:
            return None, "No exchange rate data available."
        n = max(1, min(int(inputs.get("n_rows", 20)), 100))
        rows = [f"{dt.date()}: {v:.4f}" for dt, v in series.iloc[-n:].items()]
        return None, "\n".join(rows)

    if name == TOOL_FETCH_DATA:
        result = _run_fetch_data()
        return {"type": "did_run"}, result

    if name == TOOL_RETRAIN_ALL:
        result = _run_retrain_all()
        return {"type": "did_run"}, result

    if name == TOOL_RETRAIN_MODEL:
        result = _run_retrain_model(inputs.get("model_name", ""))
        return {"type": "did_run"}, result

    return None, f"Unknown tool: {name}"


# ── App action executors ──────────────────────────────────────────────────────

def _run_fetch_data() -> str:
    """Scrape BNR and save to CSV. Returns a result string."""
    from pathlib import Path
    csv_path = Path("resources") / "data" / "idr_exchange_rates.csv"
    try:
        from .scraper import fetch_exchange_rates, save_to_csv
        headers, rows = fetch_exchange_rates()
        if not rows:
            return "BNR scraper returned no data rows — CSV not updated."
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_csv(headers, rows, str(csv_path))
        # Verify the file was actually written before claiming success
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return (
                f"Scraper returned {len(rows)} rows but the file was not written "
                f"to {csv_path.resolve()}. Check disk permissions."
            )
        return (
            f"Fetched {len(rows)} exchange rate observations from BNR. "
            f"Saved to {csv_path.resolve()}."
        )
    except Exception as exc:
        return f"Failed to fetch BNR data: {exc}"


def _run_retrain_all() -> str:
    """Run the full retraining pipeline. Returns a result string."""
    try:
        from .pipeline import retrain_pipeline
        retrain_pipeline()
        return "All models (ARIMA, Exponential Smoothing, Naive, Naive+Drift) retrained successfully."
    except Exception as exc:
        return f"Retraining failed: {exc}"


def _run_retrain_model(model_name: str) -> str:
    """Retrain a single model. Returns a result string."""
    if not model_name:
        return "No model name provided."
    try:
        from .pipeline import retrain_single_model
        retrain_single_model(model_name)
        return f"Model '{model_name}' retrained successfully."
    except Exception as exc:
        return f"Retraining {model_name} failed: {exc}"


def execute_run_action(action: dict) -> str:
    """
    Execute a pending 'run_*' action (Ollama path — not yet executed).
    Returns a human-readable result string to append to the response.
    """
    atype = action.get("type", "")
    if atype == "run_fetch_data":
        return _run_fetch_data()
    if atype == "run_retrain_all":
        return _run_retrain_all()
    if atype == "run_retrain_model":
        return _run_retrain_model(action.get("model_name", ""))
    return ""


# ── Claude backend ────────────────────────────────────────────────────────────

def get_response_claude(
    messages: list[dict],
    system: str,
    api_key: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    series: pd.Series | None = None,
    all_results: dict | None = None,
) -> ChatResponse:
    """
    Send the message history to Claude with tool use.
    Runs an agentic tool loop until stop_reason is 'end_turn'.
    """
    try:
        import anthropic
    except ImportError:
        return ChatResponse(
            text="The `anthropic` package is not installed. Run:  pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    current_messages = list(messages)
    actions: list[dict] = []
    final_text = ""

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=current_messages,
            tools=CLAUDE_TOOLS,
        )

        text_parts = [
            b.text for b in response.content if hasattr(b, "text") and b.text
        ]
        final_text = " ".join(text_parts).strip()

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue
            action, result_text = _execute_tool(
                block.name, block.input, series, all_results
            )
            if action:
                actions.append(action)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        current_messages = current_messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

    return ChatResponse(text=final_text, actions=actions)


# ── Ollama backend ────────────────────────────────────────────────────────────

_OLLAMA_HINTS = (
    "\n\n[AVAILABLE ACTION TAGS — append at the very end of your response, one per line, "
    "only when genuinely needed. Never invent data; only use these tags to trigger real app operations.]\n"
    "Visualisation: [SHOW:ARIMA]  [SHOW:ExponentialSmoothing]  [SHOW:Naive]  [SHOW:NaiveDrift]  "
    "[SHOW:comparison]  [SHOW:diagnostics:ARIMA]  [SHOW:diagnostics:ExponentialSmoothing]  "
    "[SHOW:diagnostics:Naive]  [SHOW:diagnostics:NaiveDrift]\n"
    "App actions:   [RUN:fetch_data]  [RUN:retrain_all]  "
    "[RUN:retrain:ARIMA]  [RUN:retrain:ExponentialSmoothing]  [RUN:retrain:Naive]  [RUN:retrain:NaiveDrift]"
)


def get_response_ollama(
    messages: list[dict],
    system: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    series: pd.Series | None = None,
    all_results: dict | None = None,
) -> ChatResponse:
    """Send the message history to a local Ollama instance."""
    try:
        import ollama
    except ImportError:
        return ChatResponse(
            text="The `ollama` package is not installed. Run:  pip install ollama"
        )

    ollama_msgs = [{"role": "system", "content": system}]
    for i, msg in enumerate(messages):
        m = dict(msg)
        if i == len(messages) - 1 and m["role"] == "user":
            m["content"] = m["content"] + _OLLAMA_HINTS
        ollama_msgs.append(m)

    try:
        resp = ollama.chat(model=model, messages=ollama_msgs)
        raw_text: str = resp["message"]["content"]
    except Exception as exc:
        return ChatResponse(text=f"Ollama error: {exc}")

    actions, clean_text = _parse_ollama_tags(raw_text)
    return ChatResponse(text=clean_text, actions=actions)


def _parse_ollama_tags(text: str) -> tuple[list[dict], str]:
    """Extract [SHOW:...] and [RUN:...] tags; return (actions, cleaned_text)."""
    actions: list[dict] = []

    def _handle_show(m: re.Match) -> str:
        tag = m.group(1)
        if tag == "comparison":
            actions.append({"type": "show_comparison"})
        elif tag.startswith("diagnostics:"):
            mn = tag.split(":", 1)[1]
            if mn in _MODEL_KEYS:
                actions.append({"type": "show_diagnostics", "model_name": mn})
        elif tag in _MODEL_KEYS:
            actions.append({"type": "show_forecast", "model_name": tag})
        return ""

    def _handle_run(m: re.Match) -> str:
        tag = m.group(1)
        if tag == "fetch_data":
            actions.append({"type": "run_fetch_data"})
        elif tag == "retrain_all":
            actions.append({"type": "run_retrain_all"})
        elif tag.startswith("retrain:"):
            mn = tag.split(":", 1)[1]
            if mn in _MODEL_KEYS:
                actions.append({"type": "run_retrain_model", "model_name": mn})
        return ""

    clean = re.sub(r"\[SHOW:([^\]]+)\]", _handle_show, text)
    clean = re.sub(r"\[RUN:([^\]]+)\]", _handle_run, clean).strip()
    return actions, clean


# ── Ollama model management ───────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Return True if the local Ollama daemon is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Return names of locally available Ollama models."""
    try:
        import ollama
        result = ollama.list()
        return [m["name"] for m in result.get("models", [])]
    except Exception:
        return []


def download_ollama_model(model_name: str) -> Generator[str, None, None]:
    """
    Pull an Ollama model, yielding progress lines as they arrive.
    Requires the 'ollama' CLI to be installed.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip()
        proc.wait()
        if proc.returncode != 0:
            yield f"[ERROR] ollama pull exited with code {proc.returncode}."
        else:
            yield f"[DONE] '{model_name}' is ready."
    except FileNotFoundError:
        yield (
            "[ERROR] 'ollama' command not found. "
            "Install Ollama from https://ollama.com first."
        )
    except Exception as exc:
        yield f"[ERROR] {exc}"
