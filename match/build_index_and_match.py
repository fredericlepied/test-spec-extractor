# match/build_index_and_match.py
import argparse, json, re
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def load_specs(path: str) -> List[Dict[str, Any]]:
    specs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                specs.append(json.loads(line))
            except Exception:
                pass
    return specs

def spec_to_text(spec: Dict[str, Any]) -> str:
    # Don't include test_id in the content - it's used as document ID
    parts = []
    
    # Handle None values by converting to empty lists
    preconditions = spec.get('preconditions') or []
    actions = spec.get('actions') or []
    expectations = spec.get('expectations') or []
    externals = spec.get('externals') or []
    openshift_specific = spec.get('openshift_specific') or []
    concurrency = spec.get('concurrency') or []
    artifacts = spec.get('artifacts') or []
    
    parts += preconditions
    for a in actions:
        gvk = a.get('gvk','')
        kind_hint = (a.get('fields') or {}).get('kind_hint','')
        verb = a.get('verb','')
        if gvk and kind_hint and '/' not in gvk:
            gvk = f"{gvk}/{kind_hint}"
        if gvk and verb:
            parts.append(f"{gvk}:{verb}")
        elif gvk:
            parts.append(gvk)
        elif verb:
            parts.append(f"verb:{verb}")
    parts += [f"expect:{e.get('target','')}={e.get('condition','')}" for e in expectations]
    parts += [f"ext:{x}" for x in externals]
    parts += openshift_specific
    parts += concurrency
    parts += artifacts
    return '\n'.join(map(str, parts))

def expand_equivalents(tokens: set) -> set:
    toks = set(tokens)
    has_route = any(('route.openshift.io' in t) or ('/Route:' in t) for t in toks)
    has_ing = any(('networking.k8s.io' in t) or ('/Ingress:' in t) for t in toks)
    if has_route and not has_ing:
        toks.add('networking.k8s.io/v1/Ingress:create')
    if has_ing and not has_route:
        toks.add('route.openshift.io/v1/Route:create')
    return toks

def build_embeddings(specs: List[Dict[str,Any]], model) -> np.ndarray:
    texts = [spec_to_text(s) for s in specs]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    return np.array(embs, dtype='float32')

def tokens_from_spec(s: Dict[str,Any]) -> set:
    toks = set()
    actions = s.get('actions') or []
    for act in actions:
        gvk = act.get('gvk','')
        kind_hint = (act.get('fields') or {}).get('kind_hint','')
        verb = act.get('verb','')
        if gvk and kind_hint and '/' not in gvk:
            gvk = f"{gvk}/{kind_hint}"
        if gvk and verb:
            toks.add(f"{gvk}:{verb}")
    return toks

def shared_signals(a: Dict[str,Any], b: Dict[str,Any]) -> str:
    ta, tb = expand_equivalents(tokens_from_spec(a)), expand_equivalents(tokens_from_spec(b))
    inter = sorted(ta & tb)
    return ';'.join(inter)

def cross_match(specs_a, embs_a, specs_b, embs_b, topk=5):
    idx = faiss.IndexFlatIP(embs_b.shape[1])
    idx.add(embs_b)
    sims, nbrs = idx.search(embs_a, topk)
    pairs = []
    for i, (scores, nbr) in enumerate(zip(sims, nbrs)):
        for j, sc in zip(nbr, scores):
            pairs.append({
                'idx_a': i, 'idx_b': int(j), 'base_score': float(sc),
                'a_test': specs_a[i]['test_id'],
                'b_test': specs_b[j]['test_id'],
                'shared_signals': shared_signals(specs_a[i], specs_b[j]),
            })
    return pairs

def write_report(pairs_ab, pairs_ba, out_csv):
    import pandas as pd
    df = pd.DataFrame(pairs_ab + pairs_ba)
    score_col = 'blended_score' if 'blended_score' in df.columns else 'base_score'
    df.sort_values(score_col, ascending=False, inplace=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

def coverage_matrix(specs, repo_label):
    from collections import Counter
    cv = Counter()
    for s in specs:
        actions = s.get('actions') or []
        for a in actions:
            gvk = a.get('gvk','')
            kind_hint = (a.get('fields') or {}).get('kind_hint','')
            verb = a.get('verb','')
            if gvk and kind_hint and '/' not in gvk:
                gvk = f"{gvk}/{kind_hint}"
            if gvk and verb:
                cv[(gvk, verb)] += 1
    rows = []
    for (gvk, verb), cnt in sorted(cv.items()):
        rows.append({'repo': repo_label, 'gvk': gvk, 'verb': verb, 'count': cnt})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--go', required=True, help='JSONL from Go extractor')
    ap.add_argument('--py', required=True, help='JSONL from Python extractor')
    ap.add_argument('--out', default='report.csv')
    ap.add_argument('--cov', default='coverage_matrix.csv')
    ap.add_argument('--llm', action='store_true', help='use LLM re-ranking (env vars required)')
    args = ap.parse_args()

    go_specs = load_specs(args.go)
    for s in go_specs: s['_repo'] = 'go'
    py_specs = load_specs(args.py)
    for s in py_specs: s['_repo'] = 'py'

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    go_embs = build_embeddings(go_specs, model)
    py_embs = build_embeddings(py_specs, model)

    pairs_ab = cross_match(go_specs, go_embs, py_specs, py_embs, topk=5)
    pairs_ba = cross_match(py_specs, py_embs, go_specs, go_embs, topk=5)

    if args.llm:
        from llm_rerank import rerank_batch
        print('Re-ranking A->B with LLM...')
        pairs_ab = rerank_batch(pairs_ab, go_specs, py_specs)
        print('Re-ranking B->A with LLM...')
        pairs_ba = rerank_batch(pairs_ba, py_specs, go_specs)

    write_report(pairs_ab, pairs_ba, args.out)

    df_go = coverage_matrix(go_specs, 'go')
    df_py = coverage_matrix(py_specs, 'py')
    cov = pd.concat([df_go, df_py], ignore_index=True)
    cov.to_csv(args.cov, index=False)
    print(f"Wrote {args.cov} ({len(cov)} rows)")

if __name__ == '__main__':
    main()
