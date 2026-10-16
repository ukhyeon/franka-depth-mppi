"""Hand registration: translate the *virtual* camera pose so the observed hand lands at the recommended pose.
The camera-to-robot transform is a virtual placement (not a calibration). The translation is computed from the
capture alone (never from LOW/HIGH results) and applied identically to every replay. Rotation is unchanged.
Segmentation: per-pixel BACKGROUND median with no-return pixels treated as far (inf); foreground = HOLD pixels
>3 cm nearer than background. Hand = top 15 cm of the foreground (fingers up). Descriptor: palm front = 5th pct of
the depth-axis coordinate, centre = median lateral, fingertip = 97th pct height, centroid = mean of hand points."""
import numpy as np
from scipy.spatial.transform import Rotation

def load(src,r):
    with np.load(src/r['file']) as f:d=f['depth_m'].copy()
    d[:24,:]=0;d[456:,:]=0;d[:,:32]=0;d[:,608:]=0;return d

class Segmenter:
    def __init__(self,src,md,recs):
        self.src=src;self.K=np.array(md['intrinsics']);T=md['camera_to_robot_pose_wxyz']
        self.R=Rotation.from_quat([*T[4:],T[3]]).as_matrix();self.t=np.array(T[:3]);self.T=T
        bgr=[r for r in recs if r['capture_stage']=='BACKGROUND'];self.bg_frames=len(bgr)
        if bgr:
            b=np.stack([load(src,r) for r in bgr[::max(1,len(bgr)//60)]]);b[b<=0]=np.inf;self.bg=np.median(b,0)
        else:self.bg=np.full((480,640),np.inf)
    def describe(self,r):
        d=load(self.src,r);fg=(d>0)&(d<self.bg-.03);ys,xs=np.nonzero(fg);z=d[ys,xs]
        if len(z)<500:return None
        K=self.K;pc=np.column_stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z])@self.R.T+self.t
        top=float(np.percentile(pc[:,2],97));H=pc[pc[:,2]>top-.15]
        return dict(front=float(np.percentile(H[:,0],5)),y=float(np.median(H[:,1])),top=top,surface_x=float(np.median(H[:,0])),centroid=H.mean(0),n=int(len(H)))

def register(seg,descs,target):
    o=np.median([[x['front'],x['y'],x['top']] for x in descs],0);s=np.std([[x['front'],x['y'],x['top']] for x in descs],0)
    want=np.array([target['palm_face_x_m'],target['center_y_m'],target['fingertip_z_m']]);delta=want-o;T=seg.T
    # nvblox voxel-phase adjustment (|adj| <= half voxel): put the registered median hand surface 60% into its 3 cm voxel,
    # so the voxel holding the surface is observed and the verified patch filter (observed, |sdf|<=6 cm) can accept it.
    V=.03;sx=float(np.median([x['surface_x'] for x in descs]))+delta[0];phase=(sx/V)%1;adj=((.6-phase+.5)%1-.5)*V;delta[0]+=adj
    return dict(original_camera_to_robot_pose_wxyz=T,registered_camera_to_robot_pose_wxyz=[*(seg.t+delta).tolist(),*T[3:]],translation_m=delta.round(4).tolist(),voxel_phase=dict(voxel_m=V,surface_phase_before=round(phase,3),target_phase=.6,depth_adjust_m=round(adj,4)),
                observed_hand_m=dict(palm_front_x=float(o[0]),center_y=float(o[1]),fingertip_z=float(o[2]),frame_std_m=s.round(4).tolist(),frames_used=len(descs)),
                target_m=dict(zip(['palm_front_x','center_y','fingertip_z'],want.tolist())),background_frames=seg.bg_frames,
                rule='virtual camera translation only; computed from the capture before any replay; identical for R/LOW/HIGH')

def stable_runs(centroids,tol=.01,warm=5):
    """Shared by GUI counter and tape builder: a run continues while each centroid stays within tol of the median
    of the run so far (first `warm` frames form the reference); None or a jump starts a new run. Returns run id per frame."""
    ids=[];run=[];rid=-1
    for c in centroids:
        if c is None:ids.append(None);run=[];continue
        c=np.asarray(c)
        if run and np.linalg.norm(c-np.median(np.array(run[:max(warm,1)] if len(run)<warm else run),0))>tol:run=[]
        if not run:rid+=1
        run.append(c);ids.append(rid)
    return ids
