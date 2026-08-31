"""Claude Sonnet 5 client for the agent's researcher / reflector LLM calls.

Uses the official ``anthropic`` SDK. The stable system prompt (metric scope + headroom
map + EDA summary + anchors) is one cached block, so it is paid once across the whole
run; the volatile per-turn state (best metrics, recent run-log, tried configs) rides in
the user message. Token usage (input / output / cache-write / cache-read) is returned
with every call so the orchestrator can enforce the feasibility budget.
"""
from __future__ import annotations

import os
import sys

import dotenv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Read-only parse (no os.environ mutation): dotenv.load_dotenv defaults to
# override=False, so an already-set env var would silently win. We want the
# .env key verbatim, and only the .env key.
_ENV = dotenv.dotenv_values(os.path.join(_ROOT, ".env"))

import anthropic  # noqa: E402

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
# Pin the real Anthropic endpoint. The Claude Code / editor environment injects
# ANTHROPIC_BASE_URL (a DeepSeek gateway) and ANTHROPIC_AUTH_TOKEN; the SDK
# would otherwise prefer those and route this agent off the user's own
# Anthropic account. Passing api_key + base_url explicitly overrides that.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def usage_tokens(usage: dict) -> int:
    """Total token volume for one call (input + output + cache-write + cache-read).

    The four usage fields are disjoint and together cover every token the call moved; the
    old ``input + output`` sum silently dropped cached tokens, undercounting a long
    persistent conversation's real spend.
    """
    return (int(usage.get("input", 0)) + int(usage.get("output", 0))
            + int(usage.get("cache_write", 0)) + int(usage.get("cache_read", 0)))


def usage_inout(usage: dict) -> tuple[int, int]:
    """(input-side, output) token split for one call, for the D4 resource split.

    input-side = input + cache-write + cache-read (everything the model read);
    output = generated tokens. ``sum(usage_inout(u)) == usage_tokens(u)``, so the
    existing total accounting is unchanged.
    """
    inp = (int(usage.get("input", 0)) + int(usage.get("cache_write", 0))
           + int(usage.get("cache_read", 0)))
    return inp, int(usage.get("output", 0))


class ClaudeClient:
    def __init__(self, model: str = MODEL):
        api_key = (_ENV or {}).get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set — check .env")
        self.model = model
        # Explicit api_key + base_url (not the SDK's env fallback): see the
        # ANTHROPIC_BASE_URL note above — this keeps the agent on the user's
        # own Anthropic account instead of the injected DeepSeek gateway.
        workspace_id = (_ENV or {}).get("ANTHROPIC_WORKSPACE_ID")
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=ANTHROPIC_BASE_URL,
            default_headers=({"anthropic-workspace-id": workspace_id}
                             if workspace_id else None),
        )
        self.system: list[dict] = []

    def set_system(self, text: str) -> None:
        """Set the (stable) system prompt, marked for prompt caching."""
        self.system = [{
            "type": "text", "text": text,
            "cache_control": {"type": "ephemeral"},
        }]

    def complete(self, user_text: str, tools: list[dict] | None = None,
                 tool_choice=None):
        """Send one message; returns (content_blocks, usage_dict).

        ``claude-sonnet-5`` defaults to extended thinking, which both forbids forced
        ``tool_choice`` and bills extra thinking tokens. This agent's turns (propose a
        config / write a lesson) are cheap, so we disable thinking explicitly — it
        re-enables forced tool selection and keeps the feasibility budget tight.
        """
        # Accept either a plain user string or a full message list (for multi-turn
        # tool loops where the caller appends assistant tool_use + user tool_result).
        messages = user_text if isinstance(user_text, list) else [
            {"role": "user", "content": user_text}]
        kwargs = dict(
            model=self.model, max_tokens=MAX_TOKENS,
            system=self.system, messages=messages,
            thinking={"type": "disabled"},
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        resp = self.client.messages.create(**kwargs)
        u = resp.usage
        usage = {
            "input": u.input_tokens,
            "output": u.output_tokens,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        return resp.content, usage

    @staticmethod
    def tool_use(content) -> dict:
        """First tool_use input in a content list, else {}."""
        for block in content:
            if getattr(block, "type", "") == "tool_use":
                return block.input
        return {}

    @staticmethod
    def tool_uses(content) -> list[dict]:
        """All tool_use blocks as ``[{'id', 'name', 'input'}]`` (multi-tool turns)."""
        return [{"id": b.id, "name": b.name, "input": b.input}
                for b in content if getattr(b, "type", "") == "tool_use"]

    @staticmethod
    def text(content) -> str:
        return "".join(getattr(b, "text", "") for b in content
                       if getattr(b, "type", "") == "text")
