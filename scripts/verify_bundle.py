"""Read-only consistency checks for the published inputs, code, evidence and docs."""
from pathlib import Path
import ast,hashlib,json,re,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(x) for x in p.read_text().splitlines()]
prov=json.loads((ROOT/'docs/data/provenance.json').read_text());redacted=set(prov['publication_metadata_redactions']);verified=0
for rec in prov['files']:
 if rec['published'] in redacted:continue
 assert sha(ROOT/rec['published'])==rec['sha256'],rec['published'];verified+=1
for p in (ROOT/'experiment').rglob('*.py'):ast.parse(p.read_text())
tape=ROOT/'data/hand-tape-20261012-000004-472steps-registered';meta=json.loads((tape/'metadata.json').read_text());fr=jl(tape/'frames.jsonl')
assert len(fr)==236 and meta['total_steps']==472
assert sha(tape/'frames.jsonl')==meta['frame_manifest_sha256']
assert [x['map_step'] for x in fr]==list(range(2,473,2))
for r in fr:
 with np.load(tape/r['file']) as z:
  assert z['depth_m'].shape==(480,640)
  assert hashlib.sha256(z['depth_m'].tobytes()).hexdigest()==r['depth_sha256']
  assert not any('rgb' in k.lower() for k in z.files)
scene=json.loads((ROOT/'experiment/scene.json').read_text());target=json.loads((ROOT/'experiment/target.json').read_text());assert meta['scene']==scene and meta['target_config']==target
models={};summaries={};patches={};calc={}
for k in ['R','LOW','HIGH']:
 p=ROOT/'evidence'/k;m=jl(p/'metrics.jsonl');s=json.loads((p/'summary.json').read_text());pol=json.loads((p/'mass-policy.json').read_text())
 assert len(m)==472
 iv=next(x for x in s['goal_intervals'] if x['goal_index']==1);end=m[iv['arrival_step']-1]
 assert end['position_error_m']<=.01 and end['orientation_error_rad']<=.05
 if k!='R':
  mu=lambda h:5*h/(5+h)
  eta=.025 if k=='LOW' else .025*mu(2.3)/mu(.6)
  assert abs(pol['eta_gpu_m']-eta)<1e-8 and pol['alpha_gpu']==1 and pol['weight']==100000
 models[k]=json.loads((p/'control-settings.json').read_text());summaries[k]=s;patches[k]=(p/'patches-per-frame.jsonl').read_bytes()
 assert s['map_steps']==[x['map_step'] for x in fr]
 calc[k]={'B_arrival_s':iv['arrival_sim_s'],'position_mm':end['position_error_m']*1000,'orientation_rad':end['orientation_error_rad']}
assert models['R']==models['LOW']==models['HIGH']
assert patches['R']==patches['LOW']==patches['HIGH']
missing=[]
for p in ROOT.rglob('*.md'):
 for url in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in url or url.startswith('#'):continue
  dest=url.split('#')[0]
  if not (p.parent/dest).exists():missing.append((str(p.relative_to(ROOT)),dest))
assert not missing,missing
for p in (ROOT/'docs/images').glob('*.svg'):ET.parse(p)
print(json.dumps({'copied_files_hash_verified':verified,'metadata_redactions':len(redacted),'depth_frames':len(fr),'steps':472,'shared_control_and_patch_schedules':True,'goal_arrivals':calc,'markdown_links':'valid','python_syntax':'valid','svg_xml':'valid'},indent=2))
