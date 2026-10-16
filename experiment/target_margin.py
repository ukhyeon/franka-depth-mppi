import json,hashlib,time
import numpy as np
import torch
from curobo.types.math import Pose
from isaacsim.core.api.objects import VisualCuboid

class TargetMargin:
    def __init__(self,world,sim_world,ta,config,margin):
        self.world,self.ta,self.cfg,self.margin=world,ta,config,margin
        self.low=np.array(config["manual_target_world_bounds_m"]["min"]);self.high=np.array(config["manual_target_world_bounds_m"]["max"])
        self.base=np.array(config["nominal_point_proxy_dims_m"]);self.dims=self.base+2*margin;self.capacity=config["proxy_capacity"]
        self.names=["target_margin_%02d"%i for i in range(self.capacity)]
        self.extra=[];self.nominal=[]
        for i,name in enumerate(self.names):
            world.enable_obstacle(name,False)
            a=VisualCuboid("/World/target_extra/%s"%name,position=np.array([10,0,0]),scale=self.dims,size=1.,color=np.array([1.,.2,.7]));a.set_visibility(False);self.extra.append(a)
            b=VisualCuboid("/World/target_measured/%s"%name,position=np.array([10,0,0]),scale=self.base,size=1.,color=np.array([1.,.8,0.]));b.set_visibility(False);self.nominal.append(b)
        self.points=np.empty((0,3),np.float32);self.stats={"enabled_count":0,"target_depth_points":0}
    def update(self,data,mapper,R,t):
        start=time.perf_counter();depth=data["depth"].numpy();K=data["intrinsics"].numpy();ys,xs=np.nonzero(depth>0);z=depth[ys,xs]
        pc=np.column_stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z])@R.T+t
        mask=np.all((pc>=self.low)&(pc<self.high),axis=1);points=pc[mask].astype(np.float32)
        self.points=points[::max(1,len(points)//2000)]
        anchors=[]
        if len(points):
            bins=np.floor((points[:,1:]-self.low[1:])/self.cfg["patch_bin_yz_m"]).astype(int)
            groups=bins[:,0]*8+bins[:,1]
            for key in sorted(set(groups)):
                group=points[groups==key]
                if len(group)<self.cfg["min_points_per_bin"]:continue
                center=np.median(group,axis=0);anchor=group[np.argmin(np.linalg.norm(group-center,axis=1))];anchors.append(anchor)
        anchors=np.asarray(anchors,np.float32).reshape(-1,3)
        if len(anchors):
            query=torch.cat([self.ta.to_device(anchors),torch.zeros((len(anchors),1),device="cuda")],dim=-1).contiguous();d=mapper.query_sdf(query,mapper_id=0)
            anchors=anchors[((d>-99)&(d.abs()<=.06)).cpu().numpy()]
        self.anchors=anchors.copy()
        assert len(anchors)<=self.capacity
        for i,name in enumerate(self.names):
            observed=i<len(anchors);enabled=observed and self.margin>0
            self.world.update_obb_dims(self.ta.to_device(self.dims),name=name)
            if observed:self.world.update_obb_pose(w_obj_pose=Pose.from_list([*anchors[i],1,0,0,0],self.ta),name=name)
            self.world.enable_obstacle(name,enabled)
            self.extra[i].set_visibility(enabled);self.nominal[i].set_visibility(observed)
            if observed:self.extra[i].set_world_pose(position=anchors[i]);self.nominal[i].set_world_pose(position=anchors[i])
        self.stats={"enabled_count":len(anchors) if self.margin>0 else 0,"observed_anchor_count":len(anchors),"target_depth_points":len(points),"margin_m":self.margin,"base_proxy_dims_m":self.base.tolist(),"expanded_dims_m":self.dims.tolist(),"observed_anchor_sha256":hashlib.sha256(anchors.tobytes()).hexdigest(),"policy_wall_s":time.perf_counter()-start,"state":"observed_target" if len(anchors) else "no_supported_target_observation"}
        # Read back actual native primitive cache dimensions and enable flags.
        if len(anchors):
            index=self.world.get_obb_idx(self.names[0]);dims=self.world._cube_tensor_list[0][0,index,:3].cpu().numpy()
            assert np.allclose(dims,self.dims,atol=1e-7),(dims,self.dims)
        return self.stats
    def clearance(self,spheres):
        if not len(self.points):return None
        points=self.ta.to_device(self.points)
        return float((torch.linalg.norm(points[None,:,:]-spheres[:,None,:3],dim=-1)-spheres[:,None,3]).min())
