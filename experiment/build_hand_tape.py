"""Build a static replay tape from a hand capture: longest contiguous stable HOLD run, frames in recorded order.
usage: python3 build_hand_tape.py inputs/<capture> [replay_steps] [--register]
Stability (hand-observation.json, fixed before capture): ROI centroid within 0.01 m of the HOLD median and
ROI point count >= 0.5 x HOLD median. Unstable HOLD frames are reported, never used. If replay_steps needs more
frames than the stable run, frames are reused cyclically and the tape is flagged 'looped'."""
import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
argv=[a for a in sys.argv[1:] if a!='--register'];REG='--register' in sys.argv
src=Path(argv[0]);steps=int(argv[1]) if len(argv)>1 else None
md=json.loads((src/'metadata.json').read_text());recs=[json.loads(l) for l in (src/'frames.jsonl').read_text().splitlines()]
assert hashlib.sha256((src/'frames.jsonl').read_bytes()).hexdigest()==md['frame_manifest_sha256']
hand=md.get('hand_observation') or json.loads((ROOT/'hand-observation.json').read_text());rule=hand['capture']['stable_frame_rule']
cfg=md['target_config'];lo=np.array(cfg['manual_target_world_bounds_m']['min']);hi=np.array(cfg['manual_target_world_bounds_m']['max'])
K=np.array(md['intrinsics']);T=md['camera_to_robot_pose_wxyz'];R=Rotation.from_quat([*T[4:],T[3]]).as_matrix();t=np.array(T[:3])
rows=[]
for r in recs:
    with np.load(src/r['file']) as f:d=f['depth_m'].copy()
    assert hashlib.sha256(d.tobytes()).hexdigest()==r['depth_sha256']
    d[:24,:]=0;d[456:,:]=0;d[:,:32]=0;d[:,608:]=0;ys,xs=np.nonzero(d>0);z=d[ys,xs]
    pc=np.column_stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z])@R.T+t;P=pc[np.all((pc>=lo)&(pc<hi),1)]
    rows.append(dict(ordinal=r['ordinal'],stage=r['capture_stage'],sim_time=r['sim_time'],sdk_timestamp_ms=r.get('sdk_timestamp_ms'),wall_utc_ns=r.get('wall_utc_ns'),roi_points=int(len(P)),centroid=P.mean(0) if len(P) else None))
if REG:   # stability of the segmented hand itself (registration moves it into the ROI afterwards)
    from register_hand import Segmenter,register
    seg=Segmenter(src,md,recs);by={r['ordinal']:r for r in recs}
    for x in rows:
        if x['stage']=='HOLD':
            dsc=seg.describe(by[x['ordinal']]);x['desc']=dsc;x['centroid']=None if dsc is None else dsc['centroid'];x['roi_points']=0 if dsc is None else dsc['n']
hold=[x for x in rows if x['stage']=='HOLD' and x['centroid'] is not None]
if not hold:sys.exit('no HOLD frames with ROI points; capture is not usable')
med_c=np.median([x['centroid'] for x in hold],0);med_n=np.median([x['roi_points'] for x in hold])
if REG:   # same running-median rule as the capture GUI counter
    from register_hand import stable_runs
    hr=[x for x in rows if x['stage']=='HOLD'];ids=stable_runs([x['centroid'] if x['roi_points']>=rule['roi_points_min_fraction_of_hold_median']*med_n else None for x in hr],rule['roi_centroid_shift_max_m'])
    for x,i in zip(hr,ids):x['run']=i
    from collections import Counter
    cnt=Counter(i for i in ids if i is not None);bid=cnt.most_common(1)[0][0] if cnt else None
    for x in rows:x['stable']=bool(x.get('run') is not None and x.get('run')==bid)
else:
  for x in rows:
    x['stable']=bool(x['stage']=='HOLD' and x['centroid'] is not None and np.linalg.norm(x['centroid']-med_c)<=rule['roi_centroid_shift_max_m'] and x['roi_points']>=rule['roi_points_min_fraction_of_hold_median']*med_n)
runs=[];cur=[]
for x in rows:
    if x['stable']:cur.append(x)
    elif cur:runs.append(cur);cur=[]
if cur:runs.append(cur)
best=max(runs,key=len) if runs else []
if not best:sys.exit('no stable HOLD frames')
unstable=[dict(ordinal=x['ordinal'],sim_time=x['sim_time'],sdk_timestamp_ms=x['sdk_timestamp_ms'],roi_points=x['roi_points'],centroid_shift_m=None if x['centroid'] is None else float(np.linalg.norm(x['centroid']-med_c))) for x in rows if x['stage']=='HOLD' and not x['stable']]
n=len(best);steps=steps or 2*n;need=steps//2;looped=need>n
reg=register(seg,[x['desc'] for x in best],hand['recommended_hand']) if REG else None
out=ROOT/'inputs'/('hand-tape-%s-%dsteps%s'%(src.name,steps,'-registered' if REG else ''));(out/'frames').mkdir(parents=True,exist_ok=False)
by={r['ordinal']:r for r in recs};new=[]
for i in range(need):
    x=best[i%n];r=dict(by[x['ordinal']]);f='frames/%06d.npz'%i
    shutil.copyfile(src/r['file'],out/f)
    r.update(ordinal=i,map_step=2*(i+1),global_physics_step=2*(i+1),sim_time=2*(i+1)/60,file=f,capture_stage='HOLD',source_ordinal=x['ordinal'],source_sim_time=x['sim_time'],loop_index=i//n)
    new.append(r)
(out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
m=dict(md);m.update(total_steps=steps,frame_count=len(new),map_steps=[r['map_step'] for r in new],goal_events=[],phase_events=[],
  manipulation_events=[dict(stage='HOLD',step=1,sim_s=0.,operator_confirmed=True)],
  hand_tape=dict(source_capture=str(src),source_frames=len(recs),hold_frames=len(hold),stable_hold_frames=sum(x['stable'] for x in rows),
    used_run=dict(first_ordinal=best[0]['ordinal'],last_ordinal=best[-1]['ordinal'],frames=n,sim_time=[best[0]['sim_time'],best[-1]['sim_time']],
      sdk_timestamp_ms=[best[0]['sdk_timestamp_ms'],best[-1]['sdk_timestamp_ms']],wall_utc_ns=[best[0]['wall_utc_ns'],best[-1]['wall_utc_ns']]),
    looped=looped,loops=(need+n-1)//n,unstable_hold_frames=unstable,hold_median_centroid_m=med_c.round(4).tolist(),hold_median_roi_points=float(med_n),rule=rule,stability_basis='segmented hand (top 15 cm of foreground), running-median run rule shared with GUI' if REG else 'ROI points vs HOLD median'))
m['hand_registration']=reg
m['frame_manifest_sha256']=hashlib.sha256((out/'frames.jsonl').read_bytes()).hexdigest()
(out/'metadata.json').write_text(json.dumps(m,indent=1))
print(json.dumps(dict(tape=str(out),**{k:v for k,v in m['hand_tape'].items() if k!='unstable_hold_frames'},unstable_hold_frame_count=len(unstable)),indent=1))
