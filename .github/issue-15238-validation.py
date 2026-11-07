"""Fork-only, bounded CPU validation. Not part of the proposed upstream diff."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = 'bb7b0087f0aec41089f540e0f3f13d9e6db74d37'
SOURCE = Path('integration_tests/src/main/python/conftest.py')
TEST = Path('integration_tests/src/main/python/allow_non_gpu_validation_test.py')
OUT = Path('.validation')
OUT.mkdir(exist_ok=True)
assert subprocess.check_output(['git', 'show', f'{BASE}:{SOURCE}'], text=True) == SOURCE.read_text()

audit = []
for name in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], text=True).splitlines():
    if not name.endswith('.py'):
        continue
    try:
        tree = ast.parse(Path(name).read_text())
    except (UnicodeError, SyntaxError):
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        label = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
        if label not in ('allow_non_gpu', 'allow_non_gpu_databricks'):
            continue
        kwargs = []
        for kw in node.keywords:
            value = ast.literal_eval(kw.value) if isinstance(kw.value, ast.Constant) else '<dynamic>'
            kwargs.append({'name': kw.arg, 'value': value})
            assert kw.arg == 'any' and type(value) is bool, (name, node.lineno, kwargs)
        audit.append({'path': name, 'line': node.lineno, 'marker': label, 'kwargs': kwargs})
(OUT / 'usage-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print('AUDIT', json.dumps({'calls': len(audit), 'with_kwargs': sum(bool(x['kwargs']) for x in audit), 'invalid': 0}), flush=True)

runner = OUT / 'run_hooks.py'
runner.write_text('''import importlib.util, pathlib, socket, sys
root = pathlib.Path.cwd()
source = root / 'integration_tests/src/main/python/conftest.py'
spec = importlib.util.spec_from_file_location('conftest', source)
module = importlib.util.module_from_spec(spec)
sys.modules['conftest'] = module
spec.loader.exec_module(module)
def blocked(*args, **kwargs):
    raise AssertionError('External connections are forbidden during marker tests')
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
''')

def run(label):
    xml_path = OUT / f'{label}.xml'
    cmd = [sys.executable, str(runner), str(TEST), '--noconftest', '-c', '/dev/null',
           '-q', '-p', 'no:cacheprovider', '--junitxml=' + str(xml_path),
           '-o', 'markers=allow_non_gpu: CPU allowance\nallow_non_gpu_databricks: DBR allowance\nallow_non_gpu_conditional: conditional allowance']
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    (OUT / f'{label}.log').write_text(result.stdout)
    counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
    for suite in ET.parse(xml_path).getroot().iter('testsuite'):
        for key in counts:
            counts[key] += int(suite.get(key, '0'))
    print(label.upper(), json.dumps({'exit': result.returncode, **counts}), flush=True)
    print('\n'.join(result.stdout.splitlines()[-6:]), flush=True)
    return result.returncode, counts

code, counts = run('baseline')
assert code == 1 and counts == {'tests': 84, 'failures': 64, 'errors': 0, 'skipped': 0}, counts
source = SOURCE.read_text()
helper = '''def _get_allow_non_gpu_any(mark):
    """Validate legacy marker keywords even on an inactive runtime branch."""
    unknown_kwargs = set(mark.kwargs) - {'any'}
    if unknown_kwargs:
        raise TypeError(
            f"{mark.name} got unexpected keyword argument(s) "
            f"{sorted(unknown_kwargs)}; only 'any' is supported.")
    allow_any = mark.kwargs.get('any', False)
    if not isinstance(allow_any, bool):
        raise TypeError(
            f"The 'any' parameter of '{mark.name}' must be a Boolean.")
    return allow_any


'''
changes = [
    ('def pytest_runtest_setup(item):', helper + 'def pytest_runtest_setup(item):'),
    ("    if non_gpu_databricks:\n        if is_databricks_runtime():\n            if non_gpu_databricks.kwargs and non_gpu_databricks.kwargs['any']:",
     "    if non_gpu_databricks:\n        allow_any_databricks = _get_allow_non_gpu_any(non_gpu_databricks)\n        if is_databricks_runtime():\n            if allow_any_databricks:"),
    ("        if non_gpu.kwargs and non_gpu.kwargs['any']:", "        if _get_allow_non_gpu_any(non_gpu):"),
    ("    for item in items:\n        extras = []", "    for item in items:\n        # Reject malformed keywords before formatting node IDs or running tests.\n        for marker_name in ('allow_non_gpu', 'allow_non_gpu_databricks'):\n            marker = item.get_closest_marker(marker_name)\n            if marker:\n                _get_allow_non_gpu_any(marker)\n        extras = []"),
]
for old, new in changes:
    expected = 2 if old == "        if non_gpu.kwargs and non_gpu.kwargs['any']:" else 1
    assert source.count(old) == expected, (old, source.count(old))
    source = source.replace(old, new)
SOURCE.write_text(source)
code, counts = run('fixed')
assert code == 0 and counts == {'tests': 84, 'failures': 0, 'errors': 0, 'skipped': 0}, counts
subprocess.run(['git', 'diff', '--check'], check=True)
patch = subprocess.check_output(['git', 'diff', '--', str(SOURCE)], text=True)
(OUT / 'source.patch').write_text(patch)
print(patch, flush=True)
old_tree = ast.parse(subprocess.check_output(['git', 'show', f'{BASE}:{SOURCE}'], text=True))
new_tree = ast.parse(SOURCE.read_text())
allowed = {'_get_allow_non_gpu_any', 'pytest_runtest_setup', 'pytest_collection_modifyitems'}
normalize = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if getattr(n, 'name', None) not in allowed]
assert normalize(old_tree) == normalize(new_tree)
print('UNCHANGED_OTHER_DEFINITIONS confirmed', flush=True)
print('VALIDATION_COMPLETE', flush=True)
