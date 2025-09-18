# py-extractor/extract_kubespec.py
import argparse, ast, json, os, re
from pathlib import Path

VERB_PREFIXES = ('create_', 'patch_', 'delete_', 'read_', 'list_', 'replace_', 'watch_')
CLI_RE = re.compile(r"\b(oc|kubectl)\b")
GOLDEN_RE = re.compile(r"(?i)testdata/[^\\"']+")

PSA_KEYS = [
    'pod-security.kubernetes.io/enforce',
    'pod-security.kubernetes.io/audit',
    'pod-security.kubernetes.io/warn',
]
SCC_CLI_PATTERNS = [
    'oc adm policy add-scc-to-user',
    'oc adm policy add-scc-to-group',
]

class TestVisitor(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.specs = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name.startswith('test'):
            return
        spec = {
            'test_id': f"{self.path}:{node.name}",
            'level': 'unknown',
            'preconditions': [],
            'actions': [],
            'expectations': [],
            'externals': [],
            'openshift_specific': [],
            'concurrency': [],
            'artifacts': [],
        }
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, 'attr', '') == 'parametrize':
                spec['preconditions'].append('parametrized')
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func = n.func
                if isinstance(func, ast.Attribute):
                    mname = func.attr or ''
                    low = mname.lower()
                    if low.startswith(VERB_PREFIXES):
                        verb = low.split('_', 1)[0]
                        kind = low.split('_')[-1]
                        spec['actions'].append({
                            'gvk': guess_gvk_from_attr(func),
                            'verb': verb.replace('read', 'get'),
                            'fields': {'kind_hint': kind.capitalize()},
                        })
                if isinstance(func, ast.Attribute) and func.attr in {'run','check_call','check_output'}:
                    if getattr(func.value, 'id', '') == 'subprocess':
                        args = []
                        for a in n.args:
                            if isinstance(a, ast.List):
                                for el in a.elts:
                                    if isinstance(el, ast.Constant):
                                        args.append(str(el.value))
                            elif isinstance(a, ast.Constant):
                                args.append(str(a.value))
                        cmd = ' '.join(args)
                        if CLI_RE.search(cmd):
                            spec['externals'].append(cmd)
                            low = cmd.lower()
                            if ' oc ' in (' '+low+' ') and ' create ' in low and ' route ' in low:
                                spec['actions'].append({'gvk': 'route.openshift.io/v1/Route', 'verb': 'create'})
                            if ((' kubectl ' in (' '+low+' ')) or (' oc ' in (' '+low+' '))) and ' create ' in low and ' ingress ' in low:
                                spec['actions'].append({'gvk': 'networking.k8s.io/v1/Ingress', 'verb': 'create'})
                            if (' label ' in low and ' ns ' in low) and any(k in low for k in PSA_KEYS):
                                for k in PSA_KEYS:
                                    if k in low:
                                        m = re.search(k + r'=([^\s]+)', low)
                                        if m:
                                            spec['preconditions'].append(f"psa:{k}={m.group(1)}")
                            if any(p in low for p in [s.lower() for s in SCC_CLI_PATTERNS]):
                                spec['preconditions'].append('equiv:scc~psa')
                if isinstance(func, ast.Attribute) and func.attr == 'raises':
                    if getattr(func.value, 'id', '') == 'pytest':
                        spec['expectations'].append({'target': 'exception', 'condition': 'raises'})
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                m = GOLDEN_RE.search(n.value)
                if m:
                    spec['artifacts'].append(m.group(0))
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                low = (n.func.attr or '').lower()
                if 'ingress' in low:
                    verb = low.split('_', 1)[0]
                    spec['actions'].append({'gvk': 'networking.k8s.io/v1/Ingress', 'verb': verb.replace('read','get')})

        if any(a.get('verb') in {'create','delete','patch','replace','watch'} for a in spec['actions']):
            spec['level'] = 'integration'

        bridges = set()
        for a in spec['actions']:
            g = (a.get('gvk') or '').lower()
            if 'route.openshift.io' in g and '/route' in g: bridges.add('equiv:route~ingress')
            if 'networking.k8s.io' in g and '/ingress' in g: bridges.add('equiv:route~ingress')
            if 'security.openshift.io' in g: bridges.add('equiv:scc~psa')
        for p in spec['preconditions']:
            if p.startswith('psa:'): bridges.add('equiv:scc~psa')
        for b in sorted(bridges):
            spec['preconditions'].append(b)

        self.specs.append(spec)

def guess_gvk_from_attr(attr: ast.Attribute) -> str:
    root = attr
    while isinstance(root, ast.Attribute):
        if isinstance(root.value, ast.Call) and isinstance(root.value.func, ast.Attribute):
            api = root.value.func.attr
            group, version = api_to_group_version(api)
            if group or version:
                return f"{group+'/'+version if group else version}"
        root = root.value if isinstance(root.value, ast.Attribute) else getattr(root, 'value', None)
        if root is None:
            break
    return ''

def api_to_group_version(api: str):
    m = (api or '').lower()
    if 'appsv1' in m: return ('apps','v1')
    if m == 'corev1api' or m == 'v1api' or 'corev1' in m: return ('','v1')
    if 'batchv1' in m: return ('batch','v1')
    if 'rbacauthorizationv1' in m or 'rbacv1' in m: return ('rbac.authorization.k8s.io','v1')
    if 'networkingv1' in m: return ('networking.k8s.io','v1')
    return ('','')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()

    for root, _, files in os.walk(args.root):
        for fn in files:
            if not (fn.startswith('test_') and fn.endswith('.py')):
                continue
            path = os.path.join(root, fn)
            try:
                src = Path(path).read_text(encoding='utf-8')
                tree = ast.parse(src, filename=path)
                v = TestVisitor(path)
                v.visit(tree)
                for spec in v.specs:
                    print(json.dumps(spec, ensure_ascii=False))
            except Exception:
                continue

if __name__ == '__main__':
    main()
