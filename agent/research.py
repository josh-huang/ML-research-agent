"""Literature retrieval for the agent's "deep research" capability.

The researcher can call these tools when it is stuck, or when it wants to ground a new
hypothesis in published work — the organizer explicitly rewards agents that "draw on
whatever published methods it can find" (Innovation & Problem Insight). arXiv is free,
keyless, and the relevant recsys papers (DIN, ESMM, CWM, DCN-v2, …) are all on it.

Stdlib only (``urllib`` + ``xml.etree``) so no new dependency is required. Every function
returns a *string* and never raises: a network failure degrades to a short message so the
research loop can continue without a human.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_TIMEOUT = 15


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ml-research-agent/0.1"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", _ATOM):
        aid = e.findtext("a:id", default="", namespaces=_ATOM).split("/abs/")[-1]
        title = " ".join(e.findtext("a:title", default="", namespaces=_ATOM).split())
        summary = " ".join(e.findtext("a:summary", default="", namespaces=_ATOM).split())
        out.append({"id": aid, "title": title, "summary": summary})
    return out


def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for ``query``; returns a compact ranked list of titles + abstracts."""
    try:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}", "max_results": int(max_results),
            "sortBy": "relevance"})
        entries = _parse_entries(_get(f"{_ARXIV_API}?{params}"))
        if not entries:
            return f"No arXiv results for query: {query}"
        lines = [f"arXiv search for `{query}`:"]
        for i, e in enumerate(entries):
            lines.append(f"[{i}] {e['id']} — {e['title']}\n    {e['summary'][:350]}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — degrade to a message, never crash
        return f"search_arxiv failed: {exc}"


def fetch_paper(arxiv_id: str) -> str:
    """Fetch the title + abstract of one arXiv paper by id (e.g. ``2404.05870``)."""
    try:
        url = f"{_ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}"
        entries = _parse_entries(_get(url))
        if not entries:
            return f"No arXiv entry for id: {arxiv_id}"
        e = entries[0]
        return f"{e['id']} — {e['title']}\n{e['summary'][:1200]}"
    except Exception as exc:  # noqa: BLE001
        return f"fetch_paper failed: {exc}"
