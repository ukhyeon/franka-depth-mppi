"""Regenerate documentation figures and CSV from archived logs, without simulation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/images';DATA=ROOT/'docs/data'
COL={'R':'#1baf7a','LOW':'#2a78d6','HIGH':'#eda100'}
LABEL={'R':'Baseline MPPI','LOW':'LOW: 0.6 kg, eta 25.0 mm','HIGH':'HIGH: 2.3 kg, eta 73.52 mm (p=1)'}
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained');rows=[];series=[]
for k in COL:
 p=ROOT/'evidence'/k;m=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()];summary=json.loads((p/'summary.json').read_text());iv=next(x for x in summary['goal_intervals'] if x['goal_index']==1)
 selected=[x for x in m if iv['command_step']<=x['step']<=iv['arrival_step']];gap=np.array([np.nan if x['original_surface_clearance_m'] is None else x['original_surface_clearance_m']*1000 for x in selected]);bare=gap+20
 t=np.array([x['sim_s'] for x in selected]);axes[0].plot(t,bare,color=COL[k],label=LABEL[k],lw=2);axes[0].plot(t[-1],bare[-1],marker='o',color=COL[k]);axes[1].plot(t,[x['position_error_m']*1000 for x in selected],color=COL[k],lw=2)
 axes[2].plot(t,[x['orientation_error_rad'] for x in selected],color=COL[k],lw=2)
 for ax in axes:ax.axvline(iv['arrival_sim_s'],color=COL[k],ls=':',alpha=.65)
 axes[1].annotate(f"{k}: B reached\n{iv['arrival_sim_s']:.2f} s",(t[-1],selected[-1]['position_error_m']*1000),xytext=(-75,35 if k=='R' else 75 if k=='LOW' else 115),textcoords='offset points',color=COL[k],arrowprops=dict(arrowstyle='-',color=COL[k]),fontsize=9)
 valid=np.isfinite(bare);mins=int(np.nanargmin(bare));r=dict(condition=k,assumed_hand_mass_kg=None if k=='R' else .6 if k=='LOW' else 2.3,eta_target_mm=None if k=='R' else 25 if k=='LOW' else .025*((5*2.3/(5+2.3))/(5*.6/(5+.6)))*1000,alpha=None if k=='R' else 1,weight=None if k=='R' else 100000,min_bare_sphere_clearance_mm=float(np.nanmin(bare)),min_buffered_sphere_clearance_mm=float(np.nanmin(gap)),bare_negative_samples=int((bare<0).sum()),buffered_negative_samples=int((gap<0).sum()),valid_samples=int(valid.sum()),min_step=selected[mins]['step'],min_link=selected[mins]['nearest_observed_surface_link'],A_arrival_s=next(x['arrival_sim_s'] for x in summary['goal_intervals'] if x['goal_index']==0),B_command_s=iv['command_sim_s'],B_arrival_s=iv['arrival_sim_s'],AB_duration_s=iv['duration_s'],B_position_error_mm=selected[-1]['position_error_m']*1000,B_orientation_error_rad=selected[-1]['orientation_error_rad'])
 rows.append(r)
 for x,d in zip(selected,bare):series.append(dict(condition=k,step=x['step'],sim_s=x['sim_s'],bare_sphere_clearance_mm=float(d) if np.isfinite(d) else None,buffered_sphere_clearance_mm=x['original_surface_clearance_m']*1000 if x['original_surface_clearance_m'] is not None else None,ee_position_error_mm=x['position_error_m']*1000,ee_orientation_error_rad=x['orientation_error_rad']))
axes[0].axhline(0,c='black',lw=1);axes[0].set_ylim(-30,400);axes[0].set_ylabel('Observed ROI surface clearance (mm)\nUnbuffered approximation spheres');axes[0].legend(loc='upper left',fontsize=9)
axes[1].axhline(10,c='black',lw=1,ls='--',label='Position threshold: 10 mm');axes[1].set_ylabel('EE goal position error (mm)');axes[1].set_xlabel('Simulation time from run start (s)');axes[1].legend(loc='upper right')
axes[2].axhline(.05,c='black',lw=1,ls='--',label='Orientation threshold: 0.05 rad');axes[2].set_ylabel('EE goal orientation error (rad)');axes[2].set_xlabel('Simulation time from run start (s)');axes[2].legend(loc='upper right');axes[1].set_xlabel('')
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('Same recorded hand input: separation and actual goal arrival\nB arrival requires position <= 10 mm AND orientation <= 0.05 rad',fontsize=13)
fig.savefig(OUT/'clearance-and-arrival.png',dpi=170);fig.savefig(OUT/'clearance-and-arrival.svg');plt.close(fig)
for name,rr in [('comparison.csv',rows),('timeseries.csv',series)]:
 with (DATA/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
print(json.dumps(rows,indent=2))
