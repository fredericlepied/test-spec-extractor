# match/build_index_and_match.py
import argparse, json, re
from typing import List, Dict, Any, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Operation categories for better similarity matching
OPERATION_CATEGORIES = {
    'read': ['get', 'list', 'watch'],
    'write': ['create', 'update', 'patch', 'apply'],
    'delete': ['delete'],
}

# Verb groups for similarity detection
VERB_GROUPS = {
    'read_operations': ['get', 'list', 'watch'],
    'write_operations': ['create', 'update', 'patch', 'apply'],
    'delete_operations': ['delete'],
}

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

def is_utility_test(spec: Dict[str, Any]) -> bool:
    """Check if a test spec is a utility/helper test (not a real test case)."""
    test_id = spec.get('test_id', '')
    actions = spec.get('actions') or []
    expectations = spec.get('expectations') or []
    
    # Check for utility test patterns in test_id
    utility_patterns = [
        'test_test',  # TestTestsToScript, test_test_*
        'helper',     # helper functions
        'util',       # utility functions
        'ref.go',     # reference/utility files
        'common.go',  # common utility files
        'setup',      # setup functions
        'teardown',   # teardown functions
        'mock',       # mock functions
        'stub',       # stub functions
        'fixture',    # test fixtures
        'helper_test', # helper test files
        'util_test',   # utility test files
    ]
    
    # Check if test_id contains utility patterns
    for pattern in utility_patterns:
        if pattern in test_id.lower():
            return True
    
    # Check for utility test content patterns
    # 1. No actions (no Kubernetes operations)
    # 2. Only generic test_condition expectations
    # 3. Very simple expectation conditions (single strings, basic values)
    if len(actions) == 0 and len(expectations) > 0:
        # Check if all expectations are generic test_condition with simple values
        all_generic = True
        for exp in expectations:
            target = exp.get('target', '')
            condition = exp.get('condition', '')
            
            # Not a utility test if it has specific targets
            if target not in ['test_condition']:
                all_generic = False
                break
                
            # Not a utility test if condition is complex (contains variables, function calls, etc.)
            if any(char in condition for char in ['(', ')', '[', ']', '{', '}', '==', '!=', '>', '<', 'len(', 'count']):
                all_generic = False
                break
                
            # Not a utility test if condition is too long (likely complex logic)
            if len(condition) > 50:
                all_generic = False
                break
        
        if all_generic:
            return True
    
    return False

def is_empty_test(spec: Dict[str, Any]) -> bool:
    """Check if a test spec is empty (has no meaningful content)."""
    # Handle None values by converting to empty lists
    actions = spec.get('actions') or []
    expectations = spec.get('expectations') or []
    preconditions = spec.get('preconditions') or []
    openshift_specific = spec.get('openshift_specific') or []
    concurrency = spec.get('concurrency') or []
    artifacts = spec.get('artifacts') or []
    
    # A test is considered empty if it has:
    # 1. No actions AND no expectations
    # 2. No other meaningful content (preconditions, openshift_specific, concurrency, artifacts)
    has_actions = len(actions) > 0
    has_expectations = len(expectations) > 0
    has_other_content = (len(preconditions) > 0 or 
                        len(openshift_specific) > 0 or 
                        len(concurrency) > 0 or 
                        len(artifacts) > 0)
    
    # Empty if no actions, no expectations, and no other content
    return not (has_actions or has_expectations or has_other_content)

def filter_empty_tests(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out empty tests and utility tests from the specs list."""
    filtered = []
    empty_count = 0
    utility_count = 0
    
    for spec in specs:
        if is_empty_test(spec):
            empty_count += 1
        elif is_utility_test(spec):
            utility_count += 1
        else:
            filtered.append(spec)
    
    print(f"Filtered out {empty_count} empty tests and {utility_count} utility tests, kept {len(filtered)} meaningful tests")
    return filtered

def spec_to_text(spec: Dict[str, Any]) -> str:
    # Don't include test_id in the content - it's used as document ID
    parts = []
    
    # Include level field to differentiate empty tests
    level = spec.get('level', 'unknown')
    if level:
        parts.append(f"level:{level}")
    
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

def get_resource_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract resource-level tokens (just GVK) from a spec's actions."""
    actions = s.get('actions') or []
    tokens = set()
    for a in actions:
        gvk = a.get('gvk', '')
        if gvk:
            tokens.add(gvk)
    return tokens

def get_category_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract category-level tokens (resource:category) from a spec's actions."""
    actions = s.get('actions') or []
    tokens = set()
    for a in actions:
        gvk = a.get('gvk', '')
        verb = a.get('verb', '')
        if gvk and verb:
            # Find the category for this verb
            category = None
            for cat, verbs in OPERATION_CATEGORIES.items():
                if verb in verbs:
                    category = cat
                    break
            if category:
                tokens.add(f"{gvk}:{category}")
    return tokens

def get_verb_group_tokens(s: Dict[str, Any]) -> Set[str]:
    """Extract verb group tokens (resource:verb_group) from a spec's actions."""
    actions = s.get('actions') or []
    tokens = set()
    for a in actions:
        gvk = a.get('gvk', '')
        verb = a.get('verb', '')
        if gvk and verb:
            # Find the verb group for this verb
            verb_group = None
            for group, verbs in VERB_GROUPS.items():
                if verb in verbs:
                    verb_group = group
                    break
            if verb_group:
                tokens.add(f"{gvk}:{verb_group}")
    return tokens

def shared_signals(a: Dict[str,Any], b: Dict[str,Any]) -> str:
    """Enhanced shared signals detection with multiple similarity levels."""
    signals = []
    
    # 1. Exact operation matches (GVK:verb)
    ta, tb = expand_equivalents(tokens_from_spec(a)), expand_equivalents(tokens_from_spec(b))
    exact_matches = sorted(ta & tb)
    if exact_matches:
        signals.extend([f"exact:{match}" for match in exact_matches])
    
    # 2. Resource-level matches (same GVK, different verbs)
    ra, rb = get_resource_tokens(a), get_resource_tokens(b)
    resource_matches = sorted(ra & rb)
    if resource_matches:
        signals.extend([f"resource:{match}" for match in resource_matches])
    
    # 3. Category-level matches (same GVK and operation category)
    ca, cb = get_category_tokens(a), get_category_tokens(b)
    category_matches = sorted(ca & cb)
    if category_matches:
        signals.extend([f"category:{match}" for match in category_matches])
    
    # 4. Verb group matches (same GVK and verb group)
    va, vb = get_verb_group_tokens(a), get_verb_group_tokens(b)
    verb_group_matches = sorted(va & vb)
    if verb_group_matches:
        signals.extend([f"verb_group:{match}" for match in verb_group_matches])
    
    return ';'.join(signals)

def cross_match(specs_a, embs_a, specs_b, embs_b, topk=5):
    idx = faiss.IndexFlatIP(embs_b.shape[1])
    idx.add(embs_b)
    sims, nbrs = idx.search(embs_a, topk)
    pairs = []
    for i, (scores, nbr) in enumerate(zip(sims, nbrs)):
        for j, sc in zip(nbr, scores):
            # Calculate shared signals
            shared = shared_signals(specs_a[i], specs_b[j])
            
            # Enhanced scoring based on different types of shared signals
            boosted_score = float(sc)
            if shared:
                # Count different types of shared signals
                exact_count = len([s for s in shared.split(';') if s.startswith('exact:')])
                resource_count = len([s for s in shared.split(';') if s.startswith('resource:')])
                category_count = len([s for s in shared.split(';') if s.startswith('category:')])
                verb_group_count = len([s for s in shared.split(';') if s.startswith('verb_group:')])
                
                # Boost based on signal types (exact > category > verb_group > resource)
                boost = 0.0
                if exact_count > 0:
                    boost += 0.15 * exact_count  # Highest boost for exact matches
                if category_count > 0:
                    boost += 0.12 * category_count  # High boost for category matches
                if verb_group_count > 0:
                    boost += 0.10 * verb_group_count  # Medium boost for verb group matches
                if resource_count > 0:
                    boost += 0.08 * resource_count  # Lower boost for resource matches
                
                boosted_score = min(1.0, float(sc) + boost)
            
            pairs.append({
                'idx_a': i, 'idx_b': int(j), 'base_score': float(sc),
                'blended_score': boosted_score,
                'a_test': specs_a[i]['test_id'],
                'b_test': specs_b[j]['test_id'],
                'shared_signals': shared,
            })
    return pairs

def validate_high_similarity_matches(pairs, threshold=0.8):
    """Validate that high-similarity matches actually share operations."""
    high_sim_matches = [p for p in pairs if p['base_score'] >= threshold]
    shared_ops_matches = [p for p in high_sim_matches if p['shared_signals']]
    
    # Count different types of shared signals
    exact_matches = [p for p in high_sim_matches if any(s.startswith('exact:') for s in p['shared_signals'].split(';'))]
    resource_matches = [p for p in high_sim_matches if any(s.startswith('resource:') for s in p['shared_signals'].split(';'))]
    category_matches = [p for p in high_sim_matches if any(s.startswith('category:') for s in p['shared_signals'].split(';'))]
    verb_group_matches = [p for p in high_sim_matches if any(s.startswith('verb_group:') for s in p['shared_signals'].split(';'))]
    
    print(f"High similarity matches (>{threshold}): {len(high_sim_matches)}")
    print(f"  - Exact operation matches: {len(exact_matches)}")
    print(f"  - Resource-level matches: {len(resource_matches)}")
    print(f"  - Category-level matches: {len(category_matches)}")
    print(f"  - Verb group matches: {len(verb_group_matches)}")
    print(f"  - Any shared signals: {len(shared_ops_matches)}")
    
    if len(high_sim_matches) > 0:
        validation_rate = len(shared_ops_matches)/len(high_sim_matches)*100
        print(f"Validation rate: {validation_rate:.1f}%")
        
        if validation_rate < 50:
            print(f"⚠️  {len(high_sim_matches) - len(shared_ops_matches)} high-similarity matches lack shared operations!")
            print("   These may be false positives based on text similarity alone.")
        else:
            print("✅ Good validation rate - most high-similarity matches have shared operations!")
    
    return shared_ops_matches

def write_report(pairs_ab, pairs_ba, out_csv):
    import pandas as pd
    df = pd.DataFrame(pairs_ab + pairs_ba)
    score_col = 'blended_score' if 'blended_score' in df.columns else 'base_score'
    df.sort_values(score_col, ascending=False, inplace=True)
    
    # Validate high similarity matches
    print("\n=== VALIDATION RESULTS ===")
    validate_high_similarity_matches(pairs_ab + pairs_ba)
    
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
    
    # Filter out empty tests
    print("Filtering Go specs...")
    go_specs = filter_empty_tests(go_specs)
    print("Filtering Python specs...")
    py_specs = filter_empty_tests(py_specs)

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
