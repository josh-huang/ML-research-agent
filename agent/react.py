"""The ReAct engine: one persistent LLM that sees results and proposes the next step.

A single agent (no separate researcher/reflector) holds a conversation that survives across
iterations. Each episode is a bounded tool loop within which the agent may ground a
hypothesis (search_arxiv / fetch_paper / read_run_log), run exactly one experiment
(run_experiment), see its result, and reflect (finish_episode -> lesson). After each episode
the turn block is collapsed to a one-line summary (``history``) so the context stays bounded
— the Messages API re-sends the whole list every call, so unbounded history is quadratic.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent import memory, prompts, tools
from agent.llm import usage_inout, usage_tokens
from agent.logger import tail

MAX_EPISODE_TURNS = 6
HISTORY_CAP = 20


@dataclass
class EpisodeResult:
    hypothesis: str = ""
    config: dict | None = None
    metrics: dict | None = None
    verdict: str = ""
    lesson: str = ""
    tokens: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_h: float = 0.0
    error: str | None = None
    recovery: str = ""
    ran: bool = False     # whether an experiment executed this episode
    failed: bool = False  # whether the LLM itself raised (API down) — triggers no-LLM


def _content_to_dicts(content) -> list[dict]:
    out = []
    for b in content:
        t = getattr(b, "type", "")
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


class ReActAgent:
    def __init__(self, llm, tool_schemas: list[dict]):
        self.llm = llm
        self.tool_schemas = tool_schemas
        self.history: list[str] = []  # one line per finished episode (compressed memory)

    def run_episode(self, ctx: tools.Ctx) -> EpisodeResult:
        state = ctx.state
        ctx.episode_runs = 0
        ctx.run_record = None
        ctx.lesson = ""
        ctx.finish_seen = False

        playbook = memory.load_playbook()
        lessons = memory.lessons_digest(tail(30), k=8)
        preamble = memory.build_preamble(state, tail(5), lessons, playbook)
        if self.history:
            preamble = ("## History (one line per past episode, newest last)\n"
                        + "\n".join(self.history) + "\n\n" + preamble)
        messages = [{"role": "user", "content": preamble}]

        tokens = 0
        tokens_in = 0
        tokens_out = 0
        failed = False
        try:
            for _ in range(MAX_EPISODE_TURNS):
                content, usage = self.llm.complete(messages, tools=self.tool_schemas)
                tokens += usage_tokens(usage)
                ti, to = usage_inout(usage)
                tokens_in += ti
                tokens_out += to
                uses = self.llm.tool_uses(content)
                messages.append({"role": "assistant", "content": _content_to_dicts(content)})
                if not uses:
                    break  # text-only turn: reasoning done, no further act
                results = [{"type": "tool_result", "tool_use_id": u["id"],
                            "content": tools.dispatch(u["name"], u["input"], ctx)}
                           for u in uses]
                messages.append({"role": "user", "content": results})
                if ctx.finish_seen:
                    break
        except Exception:  # noqa: BLE001 — LLM/API failure; still finalize any executed run
            failed = True

        if ctx.run_record is None:
            # No experiment ran (stall or early failure). Consume no iteration; the
            # orchestrator's no-LLM fallback guarantees forward progress.
            return EpisodeResult(tokens=tokens, tokens_in=tokens_in,
                                 tokens_out=tokens_out, failed=failed)

        if not ctx.finish_seen:
            try:
                ti, to = self._force_finish(ctx, messages)
                tokens += ti + to
                tokens_in += ti
                tokens_out += to
            except Exception:  # noqa: BLE001 — a run must never be orphaned
                pass
        verdict, vp, tp = tools.finalize_run(state, ctx.run_record, tokens, ctx.lesson,
                                             tokens_in, tokens_out)
        self._remember(memory.episode_line(ctx.run_record["hypothesis"],
                                           ctx.run_record["cfg"], vp, tp,
                                           verdict, ctx.lesson))
        r = ctx.run_record
        return EpisodeResult(hypothesis=r["hypothesis"], config=r["cfg"],
                             metrics=r["metrics"], verdict=verdict, lesson=ctx.lesson,
                             tokens=tokens, tokens_in=tokens_in, tokens_out=tokens_out,
                             gpu_h=r["gpu_h"], error=r["error"],
                             recovery=r["recovery"], ran=True, failed=failed)

    def _force_finish(self, ctx: tools.Ctx, messages: list[dict]) -> tuple[int, int]:
        """Force finish_episode after the turn cap so an executed run is never orphaned."""
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content":
                             "Now finish the episode via finish_episode with a lesson "
                             "grounded in the primary result."})
        content, usage = self.llm.complete(
            messages, tools=[prompts.FINISH_EPISODE_TOOL],
            tool_choice={"type": "tool", "name": "finish_episode"})
        inp = self.llm.tool_use(content)
        ctx.lesson = str(inp.get("lesson") or "").strip()
        ctx.finish_seen = True
        return usage_inout(usage)

    def _remember(self, line: str) -> None:
        self.history.append(line)
        if len(self.history) > HISTORY_CAP:
            self.history = self.history[-HISTORY_CAP:]
