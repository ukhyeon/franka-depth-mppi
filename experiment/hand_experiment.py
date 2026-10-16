"""Real-hand static-obstacle experiment (capture + R/LOW/HIGH replay). Copy of experiment.py (plate version kept unchanged)
with hand-observation.json settings, hand capture stages and hand GUI. Original header:
Static-obstacle ROI comparison (R/LOW/HIGH), derived from ../d435i-mppi-shuttle/experiment.py.
Verified ROI modules (base_target_cost, target_cost, target_margin, policy) are byte-identical copies.
LOW/HIGH map to the verified ETA06/ETA23 sqrt policy keys."""
import argparse,json,time,csv,hashlib
from pathlib import Path
import numpy as np
import torch
import isaacsim
from isaacsim import SimulationApp
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['record','replay','diagnostic'],default='record')
parser.add_argument('--condition',choices=['R','LOW','HIGH'],default='R')
parser.add_argument('--goal-plan',choices=['shuttle','AB','ABA'],default='shuttle',help='reached mode: AB = start->A->B then hold B; ABA = ->A->B->A then hold A')
parser.add_argument('--hand-capture',action='store_true',help='record mode: robot held at a fixed pose (no avoidance needed while the hand is placed); stage buttons BACKGROUND/PLACING/HOLD/REMOVED')
parser.add_argument('--capture-robot-pose',choices=['crossing','retract'],default='crossing',help='pose shown during capture: nominal A->B centre crossing or retract')
parser.add_argument('--hand-gui',action='store_true',help='depth overlay + hand readout window (GUI only)')
parser.add_argument('--portfolio-snapshots',action='store_true',help='opt-in: save RGB + aligned depth PNGs (button, and once 2 s after HOLD) under the session folder')
parser.add_argument('--debug-anchor-sdf',action='store_true',help='log nvblox sdf of every bin candidate (read-only query, same rule as target_margin.py)')
parser.add_argument('--eta-exponent',type=float,default=None,help='sensitivity: eta=0.025*(mu(m_h)/mu(0.6))**p for LOW/HIGH (p=0.5 equals the verified policy); default None = verified policy.py unchanged')
parser.add_argument('--hand-analysis-every',type=int,default=4,help='run the hand analysis/display every N depth frames (all frames are still recorded)')
parser.add_argument('--show-spheres',action='store_true',help='visual-only robot collision spheres (buffered radii)')
parser.add_argument('--patch-display-m',type=float,default=0.012,help='display size of observed patch markers; cost patch stays 1 mm')
parser.add_argument('--sample-steps',type=int,nargs='*',default=[120,360,600,960,1200,1800])
parser.add_argument('--record-label',default='')
parser.add_argument('--input')
parser.add_argument('--state-input',choices=['command','measured'],default='command')
parser.add_argument('--failure-action',choices=['apply','hold-measured'],default='apply')
parser.add_argument('--without-camera-map',action='store_true',help='R diagnostic only: omit depth map and target cost, retain fixed world/self collision')
parser.add_argument('--audit-failures',action='store_true')
parser.add_argument('--drive-velocity-target',choices=['zero','solver'],default='zero',help='zero: position target only (existing). solver: also send the applied command velocity as PhysX drive velocity target; gains unchanged')
parser.add_argument('--model-dt',type=float,default=None,help='Optional model base dt override; does not change physics or map cadence')
parser.add_argument('--goal-mode',choices=['recorded','reached'],default=None,help='Replay recorded R schedule or switch on actual EE arrival')
parser.add_argument('--check-steps',type=int,default=0)
parser.add_argument('--show-window',action='store_true')
parser.add_argument('--max-display-voxels',type=int,default=1000)
parser.add_argument('--voxel-display-spacing',type=float,default=.03)
parser.add_argument('--headless',action='store_true')
parser.add_argument('--record-video',action='store_true',help='replay only, visual only: render every 2nd step and save RGB PNGs from fixed cameras (oblique, side) under the session folder; nvblox voxel boxes shown; controller unchanged')
parser.add_argument('--video-steps',type=int,nargs=2,default=[1,10000000],help='first/last step to save video frames')
parser.add_argument('--light-render',action='store_true',help='Local GUI recovery: 960x540, no DLSS, single GPU; controller unchanged')
parser.add_argument('--manual-stages',action='store_true',help='Show all six manipulation stage buttons instead of simple controls')
args=parser.parse_args()
if args.mode=='replay' and not args.input:parser.error('replay requires a shuttle recording')
POLICY_KEY={'R':'R','LOW':'ETA06','HIGH':'ETA23'}[args.condition]
if args.hand_capture and args.mode!='record':parser.error('--hand-capture is a record-mode option')
if args.goal_plan!='shuttle' and args.goal_mode=='recorded':parser.error('goal-plan requires --goal-mode reached')
if args.mode=='record' and args.condition!='R':parser.error('record shared input and goal schedule with R, then replay policies')
if args.goal_mode is None:args.goal_mode='recorded' if args.mode=='replay' else 'reached'
if args.mode!='replay' and args.goal_mode=='recorded':parser.error('recorded goal mode requires replay')
if args.record_video and args.mode!='replay':parser.error('--record-video requires replay mode')
if args.max_display_voxels<=0 or args.voxel_display_spacing<=0:parser.error('positive display settings required')
if args.without_camera_map and (args.mode!='diagnostic' or args.condition!='R'):parser.error('without-camera-map requires diagnostic R')
if args.failure_action=='hold-measured' and args.state_input!='measured':parser.error('hold-measured requires measured input')
ROOT=Path(__file__).resolve().parent
scene=json.loads((ROOT/'scene.json').read_text());cfg=json.loads((ROOT/'target.json').read_text())
session=ROOT/'results'/(time.strftime('%Y%m%d-%H%M%S')+'-hand-'+args.mode+'-'+args.condition+'-'+args.goal_mode+'-'+args.state_input+('-dt'+str(args.model_dt) if args.model_dt else '')+('-hold' if args.failure_action=='hold-measured' else '')+('-p'+str(args.eta_exponent) if args.eta_exponent is not None else '')+('-'+args.goal_plan if args.goal_plan!='shuttle' else '')+('-no-camera' if args.without_camera_map else '')+('-velff' if args.drive_velocity_target=='solver' else '')+('-video' if args.record_video else ''));session.mkdir()
app_config={'headless':args.headless,'width':1280,'height':720}
if args.light_render:
 app_config.update(width=960,height=540,anti_aliasing=0,multi_gpu=False)
print('GUI_LAUNCH_CONFIG',json.dumps(app_config),flush=True)
app=SimulationApp(app_config)
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from scipy.spatial.transform import Rotation
from curobo.geom.types import WorldConfig,Cuboid
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.camera import CameraObservation
from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.mpc import MpcSolver,MpcSolverConfig
from curobo.util_file import load_yaml,join_path,get_robot_configs_path,get_world_configs_path
from helper import add_robot_to_scene,VoxelManager
from input_source import LiveDepth,TapeDepth
from target_margin import TargetMargin
from target_cost import TargetCostChannel
from camera_panel import LiveCameraPanel
import omni.ui as ui
import random
random.seed(scene['seed']);np.random.seed(scene['seed']);torch.manual_seed(scene['seed']);torch.cuda.manual_seed_all(scene['seed'])
ta=TensorDeviceType();world=World(stage_units_in_meters=1.)
world.set_simulation_dt(physics_dt=scene['physics_dt'],rendering_dt=scene['physics_dt']);world.scene.add_default_ground_plane()
stage=world.stage;stage.SetDefaultPrim(stage.DefinePrim('/World','Xform'))
robot_cfg=load_yaml(join_path(get_robot_configs_path(),'franka.yml'))['robot_cfg'];robot_cfg['kinematics']['collision_sphere_buffer']=.02
robot,path,view=add_robot_to_scene(robot_cfg,world,subroot='/World/world_robot/')
names=robot_cfg['kinematics']['cspace']['joint_names'];q0=robot_cfg['kinematics']['cspace']['retract_config']
worldcfg=WorldConfig.from_dict({'blox':{'world':{'pose':[0,0,0,1,0,0,0],'integrator_type':'occupancy','voxel_size':scene['map_voxel_m']}}})
if args.without_camera_map:worldcfg=WorldConfig()
fixed=WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(),'collision_wall.yml')));fixed.cuboid[0].pose[2]-=.01
for obj in fixed.cuboid[:2]:worldcfg.add_obstacle(obj)
from curobo.util.usd_helper import UsdHelper
usd=UsdHelper();usd.load_stage(stage);usd.add_world_to_stage(fixed.get_mesh_world(),base_frame='/World')
for i in range(cfg['proxy_capacity']):worldcfg.add_obstacle(Cuboid(name='target_margin_%02d'%i,pose=[10,0,0,1,0,0,0],dims=cfg['nominal_point_proxy_dims_m']))
mpc=MpcSolver(MpcSolverConfig.load_from_robot_config(robot_cfg,worldcfg,use_cuda_graph=True,use_cuda_graph_metrics=True,use_cuda_graph_full_step=False,self_collision_check=True,collision_checker_type=CollisionCheckerType.PRIMITIVE if args.without_camera_map else CollisionCheckerType.BLOX,use_mppi=True,use_lbfgs=False,use_es=False,store_rollouts=True,step_dt=scene['solver_dt'] if args.model_dt is None else args.model_dt))
js=JointState.from_position(mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0),joint_names=mpc.joint_names)
goal=Goal(current_state=js,goal_state=js,goal_pose=Pose.from_list(scene['goals_wxyz'][0],ta));gb=mpc.setup_solve_single(goal,1);mpc.update_goal(gb)
checker=mpc.world_collision;channel=TargetCostChannel(mpc,ta,cfg,POLICY_KEY,session);observer=TargetMargin(checker,world,ta,cfg,0.)
if args.eta_exponent is not None and args.condition!='R':
 # Sensitivity override: same target-cost objects and alpha/weight; only the activation tensor value changes (in place, before CUDA graph capture).
 import policy as _pol
 hm=_pol.HAND_MASSES[POLICY_KEY];eta_p=_pol.REFERENCE_ETA_M*(_pol.reduced_mass(hm)/_pol.reduced_mass(_pol.REFERENCE_HAND_MASS_KG))**args.eta_exponent
 for opt,c_,snap,old in channel.optimizers:
  assert not opt.cu_opt_init;ptr=c_.activation_distance.data_ptr();c_.activation_distance.fill_(eta_p);assert c_.activation_distance.data_ptr()==ptr
 channel.policy.update(eta_calculated_m=eta_p,eta_gpu_m=float(channel.optimizers[0][1].activation_distance[0]),eta_formula='0.025*(mu(m_h)/mu(0.6))**%g'%args.eta_exponent,eta_exponent=args.eta_exponent,sensitivity_override=True)
 (session/'mass-policy.json').write_text(json.dumps(channel.policy,indent=2));print('ETA_OVERRIDE',json.dumps({'p':args.eta_exponent,'eta_m':eta_p,'gpu':channel.policy['eta_gpu_m']}),flush=True)
from dataclasses import fields
from curobo.rollout.cost.primitive_collision_cost import PrimitiveCollisionCost,PrimitiveCollisionCostConfig
cost=channel.optimizers[0][1]
kw={f.name:getattr(cost,f.name) for f in fields(PrimitiveCollisionCostConfig)}
kw.update(weight=cost.weight.clone(),activation_distance=cost.activation_distance.clone())
target_diagnostic=PrimitiveCollisionCost(PrimitiveCollisionCostConfig(**kw))
materials=[OmniPBR('/World/looks/goal%d'%i,color=np.array([.2,.8,.2])) for i in range(2)]
visuals=[VisualCuboid('/World/goal%d'%i,position=np.array(g[:3]),orientation=np.array(g[3:]),size=.04,visual_material=materials[i]) for i,g in enumerate(scene['goals_wxyz'])]
viewer=VoxelManager(args.max_display_voxels,size=.03) if (not args.headless or args.record_video) else None
from static_gui import SphereVis,candidate_target_stats
from hand_gui import HandObservation,HandPanel
guide=json.loads((ROOT/'hand-observation.json').read_text())
from hand_gui import CaptureGuide
capture_guide=CaptureGuide(scene,guide) if args.hand_capture else None
hand_obs=HandObservation(ROOT,cfg,scene,guide);hand_panel=(HandPanel('Follow the top line. Green = OK, red = fix. White = detected hand.') if args.hand_capture else HandPanel()) if args.hand_gui and not args.headless else None
hand_log=(session/'hand-frames.jsonl').open('w')
patch_log=(session/'patches-per-frame.jsonl').open('w')
sphere_vis=SphereVis() if args.show_spheres else None
for v in observer.nominal:v.set_local_scale(np.full(3,args.patch_display_m))
set_camera_view(eye=np.array([1.65,1.5,1.2]),target=np.array([.4,0,.35]))
video=[]
if args.record_video:   # visual-only fixed cameras; frames read after world.step, nothing fed back to the controller
 from isaacsim.core.utils.extensions import enable_extension;enable_extension('omni.replicator.core')
 import omni.replicator.core as rep_;from pxr import UsdGeom;import cv2 as cv2_
 VIDEO_VIEWS={'oblique':([1.45,-1.15,.95],[.45,0,.42]),'front':([1.9,0.,.7],[.35,0,.45])}
 for vname,(eye,tgt) in VIDEO_VIEWS.items():
  cp='/World/video_cam_'+vname;UsdGeom.Camera.Define(stage,cp).GetFocalLengthAttr().Set(24.);set_camera_view(eye=np.array(eye),target=np.array(tgt),camera_prim_path=cp)
  rp_=rep_.create.render_product(cp,(1280,720));an_=rep_.AnnotatorRegistry.get_annotator('rgb');an_.attach([rp_]);vd=session/('video-'+vname);vd.mkdir();video.append((vname,an_,vd))
 (session/'video-views.json').write_text(json.dumps(dict(views=VIDEO_VIEWS,focal_length_mm=24,resolution=[1280,720],every_steps=2,steps=args.video_steps,note='visual only; frame saved right after world.step of that step')))
T=scene['camera_to_robot_pose_wxyz'];R=Rotation.from_quat([*T[4:],T[3]]).as_matrix();t=np.array(T[:3]);cam=Pose.from_list(T,ta)
if args.mode=='record':
 (ROOT/'capture-stage.txt').write_text(('BACKGROUND' if args.hand_capture else 'EMPTY')+'\n');source=LiveDepth(ROOT)
 source.metadata.update(scene=scene,target_config=cfg,initial_joint_positions=q0,seed=scene['seed'],camera_to_robot_pose_wxyz=T,record_label=args.record_label,hand_observation=json.loads((ROOT/'hand-observation.json').read_text()),hand_capture=args.hand_capture,capture_robot_pose=args.capture_robot_pose if args.hand_capture else None)
 source.metadata.pop('target_pose_wxyz',None)
elif args.mode=='replay':
 source=TapeDepth(args.input)
 assert source.metadata['scene']==scene and source.metadata['target_config']==cfg
 registration=source.metadata.get('hand_registration')
 if registration:   # virtual-camera translation fixed at tape build time; same for every condition
  T=registration['registered_camera_to_robot_pose_wxyz'];R=Rotation.from_quat([*T[4:],T[3]]).as_matrix();t=np.array(T[:3]);cam=Pose.from_list(T,ta)
  hand_obs.R,hand_obs.t=R,t;print('HAND_REGISTRATION',json.dumps(registration['translation_m']),flush=True)
 assert source.metadata['initial_joint_positions']==q0
 args.check_steps=source.metadata['total_steps']
else:source=None
panel=LiveCameraPanel() if args.show_window and args.mode=='record' else None
stopped=False
auto_stop_at=None
stage_name='BACKGROUND' if args.hand_capture else 'EMPTY'
snap_request=[];snap_hold_at=None;snap_files=[]
latest_frame_stage=stage_name
manipulation_events=[{'stage':'EMPTY','step':1,'sim_s':0.,'operator_confirmed':False}]
def set_stage(s):
 global stage_name,auto_stop_at,snap_hold_at
 auto_stop_at=(world.current_time-globals().get('initial',0.)+10.) if s=='ABSENT' and not args.manual_stages else ((world.current_time-globals().get('initial',0.)+guide['capture']['auto_stop_after_removed_sim_s']) if s=='REMOVED' else None)
 if s=='HOLD' and args.portfolio_snapshots:snap_hold_at=world.current_time-globals().get('initial',0.)+2.
 stage_name=s;(ROOT/'capture-stage.txt').write_text(s+'\n')
 event={'stage':s,'step':globals().get('step',0),'sim_s':world.current_time-globals().get('initial',0.),'wall_utc_ns':time.time_ns(),'operator_confirmed':True};manipulation_events.append(event)
 with (session/'manipulation-events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
def stop():
 global stopped
 stopped=True
window=ui.Window('MPPI shuttle controls',width=510,height=350)
with window.frame:
 with ui.VStack():
  status=ui.Label('Initializing MPPI')
  ui.Label('Manual central ROI only; unknown space is not verified free')
  if args.hand_capture:
   ui.Label('Hand capture: 1 background (no hand) -> 2 placing -> 3 hand fixed (HOLD, keep still ~15 s) -> 4 removed (auto-save after 3 sim s)')
   for s in ['BACKGROUND','PLACING','HOLD','REMOVED']:ui.Button(s,clicked_fn=lambda value=s:set_stage(value))
   if args.portfolio_snapshots:ui.Button('Save portfolio PNG (RGB + depth)',clicked_fn=lambda:snap_request.append('button'))
  elif args.manual_stages:
   for s in ['EMPTY','APPEAR','APPROACH','HOLD','RETREAT','ABSENT']:ui.Button(s,clicked_fn=lambda value=s:set_stage(value))
  else:
   ui.Label('Start -> move/hold/retreat object -> Removed; auto-save after10 sim seconds')
   ui.Button('1. Start test',clicked_fn=lambda:set_stage('TEST'))
   ui.Button('2. Removed / save in 10 sim seconds',clicked_fn=lambda:set_stage('ABSENT'))
  ui.Button('Stop and save',clicked_fn=stop)
capture_q=(guide['nominal_sweep']['crossing_q'] if args.capture_robot_pose=='crossing' else q0[:7]) if args.hand_capture else None
world.reset();world.play();indices=[robot.get_dof_index(n) for n in names];robot.set_joint_positions((list(capture_q)+list(q0[7:])) if args.hand_capture else q0,indices);robot._articulation_view.set_max_efforts(np.full(len(indices),5000),joint_indices=indices)
command_names=mpc.joint_names;command_indices=[robot.get_dof_index(n) for n in command_names]
controller=robot.get_articulation_controller();drive_kp,drive_kd=[x.reshape(-1).tolist() for x in robot._articulation_view.get_gains(joint_indices=np.array([robot.get_dof_index(n) for n in mpc.joint_names]))];active_goal=0;held_from=None;pending_switch=False;command=None;switches=[];events=[{'step':1,'sim_s':0.,'goal_index':0}];metrics=[];sphere_log=[];map_steps=[];diagnostic=args.mode=='diagnostic'
last_spheres=None
capture_js=JointState.from_position(ta.to_device([capture_q]),joint_names=mpc.joint_names) if args.hand_capture else None
if args.hand_capture:assert list(names[:7])==list(mpc.joint_names)
expected_events=source.metadata['goal_events'] if args.mode=='replay' else None
from goal_progress import GoalProgress
progress=GoalProgress(session,args.goal_mode)
progress.command(0,1,0.)
current_state=js.clone()
# Audit existing filter output before command integration can mutate its state.
filtered_start=js.clone()
original_filter=mpc.solver.safety_rollout.filter_robot_state
def audit_filter(raw):
 filtered=original_filter(raw);filtered_start.copy_(filtered);return filtered
mpc.solver.safety_rollout.filter_robot_state=audit_filter
from failure_probe import FailureProbe
probe=FailureProbe(mpc) if args.audit_failures else None
failure_streak=0
failure_events=[]
mf=(session/'metrics.jsonl').open('w');ef=(session/'goal-events.jsonl').open('w');ef.write(json.dumps(events[0])+'\n');ef.flush()
initial=world.current_time;wall0=time.monotonic();preview=source.preview if source else None
# Show the true optical frustum and existing manual target ROI using visualization-only lines.
if preview is not None:
 from isaacsim.util.debug_draw import _debug_draw
 draw=_debug_draw.acquire_debug_draw_interface();K=preview['intrinsics'].numpy();corners=[]
 for z in [.15,.7]:
  for u,v in [(0,0),(639,0),(639,479),(0,479)]:corners.append(t+R@np.array([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]))
 edges=[(i,(i+1)%4) for i in range(4)]+[(i+4,(i+1)%4+4) for i in range(4)]+[(i,i+4) for i in range(4)]
 draw.draw_lines([tuple(corners[a]) for a,b in edges],[tuple(corners[b]) for a,b in edges],[(0,1,1,1)]*12,[2.]*12)
 lo=np.array(cfg['manual_target_world_bounds_m']['min']);hi=np.array(cfg['manual_target_world_bounds_m']['max'])
 rc=[(x,y,z) for z in [lo[2],hi[2]] for x,y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]]
 draw.draw_lines([rc[a] for a,b in edges],[rc[b] for a,b in edges],[(1,.8,0,1)]*12,[2.]*12)
for step in range(1,(args.check_steps or 10000000)+1):
 if not app.is_running() or stopped or (panel and panel.stop_requested):break
 if auto_stop_at is not None and world.current_time-initial>=auto_stop_at:
  print('AUTO_SAVE_AFTER_REMOVAL',world.current_time-initial,flush=True);break
 world.step(render=((not args.headless) and (not args.hand_capture or step%2==0)) or (args.record_video and step%2==0));now=world.current_time-initial
 if video and step%2==0 and args.video_steps[0]<=step<=args.video_steps[1]:
  for vname,an_,vd in video:
   img=an_.get_data()
   if img is not None and getattr(img,'size',0):cv2_.imwrite(str(vd/('%06d.png'%step)),cv2_.cvtColor(np.asarray(img)[:,:,:3],cv2_.COLOR_RGB2BGR))
 if args.mode=='replay':stage_name=source.by_step.get(max([x for x in source.by_step if x<=step],default=0),{}).get('capture_stage','EMPTY')
 if args.mode=='replay' and args.goal_mode=='recorded':
  ev=[x for x in expected_events if x['step']<=step][-1]
  if ev['goal_index']!=active_goal:active_goal=ev['goal_index'];held_from=None;pending_switch=True
 elif pending_switch:active_goal=1-active_goal;held_from=None
 if pending_switch:
  progress.command(active_goal,step,now)
  gb.goal_pose.copy_(Pose.from_list(scene['goals_wxyz'][active_goal],ta));mpc.update_goal(gb)
  e={'step':step,'sim_s':now,'goal_index':active_goal};events.append(e);ef.write(json.dumps(e)+'\n');ef.flush();print('GOAL_SWITCH',json.dumps(e),flush=True);pending_switch=False
 for i,v in enumerate(visuals):materials[i].set_color(np.array([0,1,0]) if i==active_goal else np.array([.25,.25,.25]))
 if source and source.has_update(step,world.current_time_step_index):
  data=source.get_data(step);d=data['depth'];d[:24,:]=0;d[456:,:]=0;d[:,:32]=0;d[:,608:]=0
  checker.decay_layer('world');checker.add_camera_frame(CameraObservation(depth_image=d,intrinsics=data['intrinsics'],pose=cam).to(device=ta.device),'world');checker.process_camera_frames('world',False);torch.cuda.synchronize();checker.update_blox_hashes();torch.cuda.synchronize()
  observer.update(data,checker._blox_mapper,R,t);channel.update(observer.anchors);map_steps.append(step)
  if args.mode=='record':
   source.publish(step,world.current_time_step_index,world.current_time,stage_name);latest_frame_stage=stage_name
  if panel:panel.update(data)
  if args.portfolio_snapshots and args.mode=='record' and (snap_request or (snap_hold_at is not None and now>=snap_hold_at)):
   from portfolio import save_snapshot
   why=snap_request.pop(0) if snap_request else 'auto-2s-after-HOLD'
   if why!='button':snap_hold_at=None
   hand_obs.analyze(d.cpu().numpy(),data['intrinsics'].cpu().numpy(),observer.anchors,stage_name,last_spheres);ov=hand_obs.overlay(d.cpu().numpy(),data['intrinsics'].cpu().numpy())
   f=save_snapshot(session/'portfolio',source.rgb,source.raw,source.scale,ov,dict(step=step,stage=stage_name,trigger=why,sdk=getattr(source,'capture',None)));snap_files.append(f);print('PORTFOLIO_SNAPSHOT',f,flush=True)
  if args.debug_anchor_sdf and len(map_steps)%8==1:
   dd=d.cpu().numpy();Kd=data['intrinsics'].cpu().numpy();yy,xx=np.nonzero(dd>0);zz=dd[yy,xx];pp=np.column_stack([(xx-Kd[0,2])*zz/Kd[0,0],(yy-Kd[1,2])*zz/Kd[1,1],zz])@R.T+t
   lo_=observer.low;hi_=observer.high;pp=pp[np.all((pp>=lo_)&(pp<hi_),1)].astype(np.float32);bb=np.floor((pp[:,1:]-lo_[1:])/cfg['patch_bin_yz_m']).astype(int);gg=bb[:,0]*8+bb[:,1];cand=[]
   for key in sorted(set(gg)):
    grp=pp[gg==key]
    if len(grp)<cfg['min_points_per_bin']:continue
    cen=np.median(grp,axis=0);cand.append((int(key),grp[np.argmin(np.linalg.norm(grp-cen,axis=1))]))
   if cand:
    qq=torch.cat([ta.to_device(np.array([c[1] for c in cand])),torch.zeros((len(cand),1),device='cuda')],-1).contiguous();sd=checker._blox_mapper.query_sdf(qq,mapper_id=0).cpu().numpy()
    with (session/'anchor-sdf-debug.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,candidates=[dict(key=k,y=float(c[1]),z=float(c[2]),x=float(c[0]),sdf=float(v)) for (k,c),v in zip(cand,sd)]))+'\n')
  if capture_guide:
   cg=capture_guide.track(d.cpu().numpy(),data['intrinsics'].cpu().numpy(),stage_name)
   with (session/'capture-guide.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,sdk=getattr(source,'capture',None),**cg))+'\n')
   if hand_panel and len(map_steps)%2==1:hand_panel.update(capture_guide.overlay(d.cpu().numpy()),capture_guide.text())
  patch_log.write(json.dumps(dict(step=step,sim_s=now,stage=stage_name,sdk=getattr(source,'capture',None) if args.mode=='record' else None,count=int(len(observer.anchors)),valid=observer.stats.get('state'),roi_points=observer.stats.get('target_depth_points'),anchors=np.asarray(observer.anchors).round(5).tolist()))+'\n')
  if len(map_steps)%args.hand_analysis_every==1 or args.hand_analysis_every==1:
   ho=hand_obs.analyze(d.cpu().numpy(),data['intrinsics'].cpu().numpy(),observer.anchors,stage_name,last_spheres)
   hand_log.write(json.dumps(dict(step=step,sim_s=now,sdk=getattr(source,'capture',None) if args.mode=='record' else None,**ho))+'\n');hand_log.flush()
   if hand_panel and not capture_guide:hand_panel.update(hand_obs.overlay(d.cpu().numpy(),data['intrinsics'].cpu().numpy()),hand_obs.text(ho))
   if len(map_steps)%(8*args.hand_analysis_every)==1:print('HAND_STATUS',json.dumps({k:ho[k] for k in ['stage','roi_points','anchors','bins_mixed_depth','behind_points','moving','hold_dropout','nominal_path_gap_m','live_robot_gap_m']}),flush=True)
  if viewer and step%6<=1:
   vox=checker.get_voxels_in_bounding_box(Cuboid('display',dims=[1,1.2,1],pose=[.4,0,.5,1,0,0,0]),args.voxel_display_spacing);viewer.update_voxels(vox[:,:3].cpu().numpy())
 state=robot.get_joints_state();actual=JointState(position=ta.to_device(state.positions),velocity=ta.to_device(state.velocities)*0,acceleration=ta.to_device(state.velocities)*0,jerk=ta.to_device(state.velocities)*0,joint_names=robot.dof_names).get_ordered_joint_state(mpc.joint_names)
 measured=actual.clone();measured.velocity.copy_(ta.to_device(state.velocities)[[robot.get_dof_index(n) for n in mpc.joint_names]])
 if args.state_input=='measured':measured=measured.unsqueeze(0)
 if args.state_input=='measured':assert measured.position.shape==current_state.position.shape
 current_state.copy_(measured if args.state_input=='measured' else (actual if command is None else command.get_ordered_joint_state(mpc.joint_names)))
 submitted=current_state.clone()
 if args.state_input=='measured':
  assert torch.equal(submitted.position,measured.position) and torch.equal(submitted.velocity,measured.velocity)
 current=current_state
 kin=mpc.rollout_fn.compute_kinematics(actual);ee=kin.ee_pos_seq.reshape(-1,3)[0];quat=kin.ee_quat_seq.reshape(-1,4)[0];gp=gb.goal_pose.position.reshape(-1,3)[0];gq=gb.goal_pose.quaternion.reshape(-1,4)[0]
 pe=float(torch.linalg.norm(ee-gp));qe=float(2*torch.acos(torch.dot(quat,gq).abs().clamp(0,1)))
 within=pe<=scene['position_tolerance_m'] and qe<=scene['orientation_tolerance_rad']
 if within:
  if held_from is None:held_from=now
 else:held_from=None
 held=now-held_from if held_from is not None else 0.
 if within:progress.reach(step,now,pe,qe)
 plan_len={'shuttle':None,'AB':2,'ABA':3}[args.goal_plan]
 if within and args.goal_mode=='reached' and (plan_len is None or len(progress.arrivals)<plan_len):pending_switch=True
 if probe:probe.begin()
 torch.cuda.synchronize();begin=time.perf_counter();result=mpc.step(current,max_attempts=2);torch.cuda.synchronize();solver_s=time.perf_counter()-begin
 optimizer_start=mpc.solver.optimizers[0].rollout_fn.start_state.clone()
 assert torch.equal(optimizer_start.position,filtered_start.position) and torch.equal(optimizer_start.velocity,filtered_start.velocity)
 solver_command=result.js_action.get_ordered_joint_state(command_names)
 failed=not bool(result.metrics.feasible.all()) or not bool(torch.isfinite(solver_command.position).all())
 if failed:
  failure_streak+=1
  diag=probe.describe(actual,filtered_start,result,observer) if probe else {'constraint_total':result.metrics.constraint.cpu().tolist()}
  diag.update(step=step,sim_s=now,failure_streak=failure_streak,failure_action=args.failure_action)
  with (session/'infeasible-diagnostics.jsonl').open('a') as f:f.write(json.dumps(diag)+'\n')
 elif failure_streak:
  failure_events.append({'event':'recovered','step':step,'sim_s':now,'preceding_failure_steps':failure_streak});failure_streak=0
 command=solver_command
 if failed and args.failure_action=='hold-measured':
  command=measured.clone();command.velocity.zero_();command.acceleration.zero_();command.jerk.zero_()
  assert torch.equal(command.position,measured.position)
  failure_events.append({'event':'hold_measured','step':step,'sim_s':now,'consecutive':failure_streak,'q':command.position.cpu().reshape(-1).tolist(),'native_warm_start_empty':mpc.solver._init_act_seq is None})
 if args.hand_capture:command=capture_js.clone()  # robot held still while the hand is captured; solver output ignored
 velocity_target=command.velocity.reshape(-1).cpu().numpy() if args.drive_velocity_target=='solver' else None
 controller.apply_action(ArticulationAction(command.position.reshape(-1).cpu().numpy(),joint_velocities=velocity_target,joint_indices=command_indices))
 if failed and probe:
  kin=mpc.rollout_fn.compute_kinematics(actual);ee=kin.ee_pos_seq.reshape(-1,3)[0];quat=kin.ee_quat_seq.reshape(-1,4)[0]
 spheres=kin.robot_spheres.reshape(-1,4);spheres=spheres[spheres[:,3]>0]
 last_spheres=spheres.cpu().numpy()
 sphere_log.append(spheres.cpu().numpy().copy())
 clearance=observer.clearance(spheres) if source else None
 sc=mpc.rollout_fn.robot_self_collision_cost.forward(kin.robot_spheres.reshape(1,1,-1,4)) if mpc.rollout_fn.robot_self_collision_cost else None
 current_target_cost=float(target_diagnostic.forward(kin.robot_spheres.reshape(1,1,-1,4)).sum())
 nearest_link=None
 if len(observer.points):
  pts=ta.to_device(observer.points);dist=torch.linalg.norm(pts[None,:,:]-spheres[:,None,:3],dim=-1)-spheres[:,None,3];near=int(dist.argmin())//len(pts)
  model=mpc.rollout_fn.dynamics_model.robot_model;indices_sphere=torch.nonzero(kin.robot_spheres.reshape(-1,4)[:,3]>0).reshape(-1)
  li=int(model.kinematics_config.link_sphere_idx_map[indices_sphere[near]])
  nearest_link={v:k for k,v in model.kinematics_config.link_name_to_idx_map.items()}[li]
 row={'target_cost_current':current_target_cost,'nearest_observed_surface_link':nearest_link,'step':step,'sim_s':now,'wall_s':time.monotonic()-wall0,'stage':latest_frame_stage if args.mode=='record' else stage_name,'goal_index':active_goal,'goal_position':gp.cpu().tolist(),'goal_quaternion_wxyz':gq.cpu().tolist(),'position_error_m':pe,'orientation_error_rad':qe,'goal_hold_sim_s':held,'solver_wall_s':solver_s,'actual_q':actual.position.cpu().reshape(-1).tolist(),'command_q':command.position.cpu().reshape(-1).tolist(),'input_actual_q_max_rad':float((current.position-actual.position).abs().max()),'ee_position':ee.cpu().tolist(),'ee_quaternion_wxyz':quat.cpu().tolist(),'original_surface_clearance_m':clearance,'self_collision_cost_max':float(sc.max()) if sc is not None else None,'feasible':result.metrics.feasible.cpu().tolist(),**observer.stats}
 row.update(failure_action=args.failure_action,hold_applied=failed and args.failure_action=='hold-measured',failure_streak=failure_streak,attempts=probe.attempts if probe else None,solver_command_q=solver_command.position.cpu().reshape(-1).tolist(),solver_command_velocity=solver_command.velocity.cpu().reshape(-1).tolist())
 row.update(optimizer_start_q=optimizer_start.position.cpu().reshape(-1).tolist(),optimizer_start_velocity=optimizer_start.velocity.cpu().reshape(-1).tolist(),state_input=args.state_input,actual_velocity=measured.velocity.cpu().reshape(-1).tolist(),submitted_q=submitted.position.cpu().reshape(-1).tolist(),submitted_velocity=submitted.velocity.cpu().reshape(-1).tolist(),filtered_start_q=filtered_start.position.cpu().reshape(-1).tolist(),filtered_start_velocity=filtered_start.velocity.cpu().reshape(-1).tolist(),filtered_start_acceleration=filtered_start.acceleration.cpu().reshape(-1).tolist(),filtered_actual_q_max_rad=float((filtered_start.position-actual.position).abs().max()),filtered_actual_velocity_max_rad_s=float((filtered_start.velocity-measured.velocity).abs().max()),command_actual_q_max_rad=float((command.position-actual.position).abs().max()),command_velocity=command.velocity.cpu().reshape(-1).tolist(),drive_velocity_target=args.drive_velocity_target,applied_velocity_target=velocity_target.tolist() if velocity_target is not None else None,goal_mode=args.goal_mode,completed_round_trips=progress.cycles,solver_input_q=current.position.cpu().reshape(-1).tolist())
 row.update(condition_label=args.condition,policy_key=POLICY_KEY,goal_plan=args.goal_plan,**candidate_target_stats(channel))
 metrics.append(row);mf.write(json.dumps(row)+'\n');mf.flush()
 if step in args.sample_steps:channel.save_sample(step,row['stage'])
 if sphere_vis and step%3==0:sphere_vis.update(kin.robot_spheres.reshape(-1,4))
 if step%60==0:
  status.text='%s | goal %d | %.2fs | EE %.1fmm %.3frad\nIn tolerance %.2fs (no wait) | points %d | patches %d | eta %.2fmm'%(args.condition,active_goal,now,pe*1000,qe,held,observer.stats.get('target_depth_points',0),channel.enabled,channel.policy['eta_calculated_m']*1000)+ '\nTarget cost %.1f | nearest %s'%(current_target_cost,nearest_link)
  print('SHUTTLE_STATUS',json.dumps({k:row[k] for k in ['step','sim_s','goal_index','position_error_m','orientation_error_rad','goal_hold_sim_s','original_surface_clearance_m','target_depth_points']}),flush=True)
 if step==1:
  maps=Path('/proc/self/maps').read_text();native=sorted(set(x.split()[-1] for x in maps.splitlines() if 'libnvblox_lib.so' in x));wrapper=sorted(set(x.split()[-1] for x in maps.splitlines() if 'libpy_nvblox.so' in x));assert (native or args.without_camera_map) and all('abi0-recovery/install/lib' in x for x in native);assert (wrapper or args.without_camera_map) and all('abi0-recovery/nvblox_torch/' in x for x in wrapper)
  (session/'runtime.json').write_text(json.dumps({'native':native,'wrapper':wrapper,'torch':torch.__version__,'abi':torch._C._GLIBCXX_USE_CXX11_ABI,'scene':scene,'mode':args.mode,'condition':args.condition,'policy_key':POLICY_KEY,'goal_plan':args.goal_plan,'obstacle_guide':guide,'camera_pose_used':T,'hand_registration':source.metadata.get('hand_registration') if source else None,'failure_action':args.failure_action,'drive_velocity_target':args.drive_velocity_target,'without_camera_map':args.without_camera_map,'fixed_world':[{'name':o.name,'pose':o.pose,'dims':o.dims} for o in fixed.cuboid[:2]],'particles':400,'horizon':30,'iterations':5,'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}},indent=2))
model=mpc.solver.safety_rollout.dynamics_model
(session/'control-settings.json').write_text(json.dumps({'state_input':args.state_input,'drive_velocity_target':args.drive_velocity_target,'drive_stiffness':drive_kp,'drive_damping':drive_kd,'drive_damping_over_stiffness_s':[d/k if k else None for k,d in zip(drive_kp,drive_kd)],'physics_dt':scene['physics_dt'],'model_base_dt':model.dt,'trajectory_dt':model.traj_dt.cpu().reshape(-1).tolist(),'trajectory_duration_s':float(model.traj_dt.sum()),'filter_enabled':model.state_filter.enable,'filter_coeff':vars(model.state_filter.filter_coeff),'filter_dt':model.state_filter.dt,'control_space':str(model.control_space),'joint_names':mpc.joint_names,'command_index':0,'shift_steps':1,'acceleration_input':'zero placeholder for measured ACCELERATION mode; acceleration is next control, no new estimator'},indent=2))
(session/'failure-events.json').write_text(json.dumps(failure_events,indent=2))
# Normal cleanup, no fast-exit or object lifetime bypass.
progress.finish(metrics[-1] if metrics else None)
(session/'eta-after-run.json').write_text(json.dumps([dict(captured=bool(o.cu_opt_init),activation_m=float(c_.activation_distance[0]),ptr=c_.activation_distance.data_ptr()) for o,c_,sn,ol in channel.optimizers]))
hand_log.close();patch_log.close();mf.close();ef.close();channel.finish()
np.savez_compressed(session/'robot-spheres.npz',spheres=np.array(sphere_log),sim_s=np.array([x['sim_s'] for x in metrics]))
if source:
 if args.mode=='record':source.metadata.update(goal_events=events,map_steps=map_steps,manipulation_events=manipulation_events)
 source.finish(len(metrics),session/'unused-phase-events.jsonl');source.stop_device()
summary={'goal_mode':args.goal_mode,'arrival_sequence':progress.arrivals,'completed_round_trips':progress.cycles,'goal_intervals':progress.intervals,'steps':len(metrics),'mode':args.mode,'condition':args.condition,'input':str(source.archive) if source else None,'goal_events':events,'map_steps':map_steps,'goal_switch_count':len(events)-1,'sim_elapsed_s':metrics[-1]['sim_s'] if metrics else 0,'physical_scene_confirmed_empty':False,'dynamic_avoidance_verified':False,'diagnostic_only':diagnostic,'distance_scope':'visible original ROI depth surface to 61 buffered spheres, not cost/proxy geometry or full occluded object','record_interface':'manual_stages' if args.manual_stages else 'two_buttons_TEST_ABSENT','goal_switch_rule':('actual EE <=10mm and <=0.05rad; zero dwell; switch next step' if args.goal_mode=='reached' else 'recorded R goal schedule'),'unknown_space':'not verified free'}
(session/'summary.json').write_text(json.dumps(summary,indent=2));print('SHUTTLE_NORMAL_RETURN',json.dumps(summary),flush=True);app.close()
