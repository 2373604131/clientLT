import hashlib
import json
from pathlib import Path
import shutil

source = Path('/data/yzh/clientLT')
target = source/'tmp/ab_parameter_search_20261014_01'
if target.exists():
    raise RuntimeError('Isolated source already exists; do not overwrite')
target.mkdir(parents=True)
for name in ('utils', 'tools', 'scripts', 'trainers', 'tests', 'clip', 'Dassl', 'datasets', 'configs', 'loss'):
    shutil.copytree(source/name, target/name, ignore=shutil.ignore_patterns('__pycache__', '.git'))
for path in source.glob('*.py'):
    shutil.copy2(path, target/path.name)
for name in ('third_party', 'references', 'DATA'):
    (target/name).symlink_to(source/name, target_is_directory=True)
changes = {
 'utils/method_a_margin_math.py': [
  ('def algorithm_config(**parameters):', 'def algorithm_config(correction_step_size=.1, **parameters):'),
  ("    return dict(schema_version=VARIANT, variant=VARIANT, **values,", "    if not math.isfinite(correction_step_size) or correction_step_size <= 0:\n        raise ValueError('Correction step size must be finite and positive')\n    return dict(schema_version=VARIANT, variant=VARIANT, **values,"),
  ('correction_steps=3, correction_step_size=.1,', 'correction_steps=3, correction_step_size=correction_step_size,')],
 'utils/cliplora_method_a_margin.py': [
  ("        config = algorithm_config(**{k: getattr(self.args, 'a_margin_'+k, v) for k, v in DEFAULTS.items()})", "        config = algorithm_config(\n            correction_step_size=getattr(self.args, 'a_margin_correction_step_size', .1),\n            **{k: getattr(self.args, 'a_margin_'+k, v) for k, v in DEFAULTS.items()})")],
 'tools/client_aggregation/protocol.py': [
  ("    if set(parameters) != set(DEFAULTS):\n        raise ValueError('New A requires all four registered margin parameters')", "    if set(parameters) not in (set(DEFAULTS), set(DEFAULTS) | {'correction_step_size'}):\n        raise ValueError('New A requires all four registered margin parameters and optional correction_step_size')")],
 'federated_main.py': [
  ("    parser.add_argument('--a_margin_learn_weight', type=float, default=1.)", "    parser.add_argument('--a_margin_learn_weight', type=float, default=1.)\n    parser.add_argument('--a_margin_correction_step_size', type=float, default=.1)")]
}
manifest = {}
for name, replacements in changes.items():
    path = target/name
    original = path.read_text(encoding='utf-8')
    value = original
    for old, new in replacements:
        if value.count(old) != 1:
            raise RuntimeError('Patch context not unique: ' + name)
        value = value.replace(old, new)
    path.write_text(value, encoding='utf-8')
    manifest[name] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                          isolated_sha256=hashlib.sha256(value.encode()).hexdigest())
(target/'deployment_changes.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(source=str(source), isolated=str(target), changed_files=list(manifest)), indent=2))
