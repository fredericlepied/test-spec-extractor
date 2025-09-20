# match/llm_rerank.py
import json, time, os
from typing import Dict, Any, List

import requests

SYSTEM_PROMPT = (
    "You compare two Kubernetes/OpenShift test specs for semantic overlap across languages and levels. "
    "Return STRICT JSON with keys: overlap (full|partial|none), score (0-100), "
    "evidence (list of short strings), differences (short string). "
    "Judgement criteria: GVK+verbs, selectors/fields, conditions (Ready/Available/etc.), events, timeouts, externals (HTTP/CLI), "
    "OpenShift-vs-upstream equivalents (Route~Ingress, SCC~PSA). Be conservative; avoid overclaiming."
)

USER_TMPL = """\
Spec A (repo={repo_a}):
{spec_a}

Spec B (repo={repo_b}):
{spec_b}

Return ONLY the JSON.
"""


def _chat(messages: list, *, model: str, api_key: str, base: str) -> str:
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def rerank_pair(spec_a: Dict[str, Any], spec_b: Dict[str, Any], repo_a="A", repo_b="B"):
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")

    user = USER_TMPL.format(
        repo_a=repo_a,
        repo_b=repo_b,
        spec_a=json.dumps(spec_a, ensure_ascii=False),
        spec_b=json.dumps(spec_b, ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    for attempt in range(3):
        try:
            txt = _chat(messages, model=model, api_key=api_key, base=base)
            obj = json.loads(txt)
            overlap = str(obj.get("overlap", "none")).lower()
            if overlap not in {"full", "partial", "none"}:
                overlap = "none"
            score = float(obj.get("score", 0))
            evidence = [str(x) for x in obj.get("evidence", [])][:6]
            differences = str(obj.get("differences", ""))[:500]
            return overlap, score, evidence, differences
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (2**attempt))


def rerank_batch(
    pairs: List[dict], specs_a: List[Dict[str, Any]], specs_b: List[Dict[str, Any]]
) -> List[dict]:
    out = []
    for p in pairs:
        i, j = p["idx_a"], p["idx_b"]
        overlap, score, evidence, differences = rerank_pair(
            specs_a[i],
            specs_b[j],
            repo_a=specs_a[i].get("_repo", "A"),
            repo_b=specs_b[j].get("_repo", "B"),
        )
        final = dict(p)
        final["llm_overlap"] = overlap
        final["llm_score"] = float(score)
        final["llm_evidence"] = evidence
        final["llm_diff"] = differences
        final["blended_score"] = 0.7 * (float(score) / 100.0) + 0.3 * max(
            0.0, min(1.0, p.get("base_score", 0))
        )
        out.append(final)
    return out
