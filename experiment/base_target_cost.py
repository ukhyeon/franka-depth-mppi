import json,types
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from curobo.geom.types import Cuboid,WorldConfig
from curobo.geom.sdf.world import WorldCollisionConfig,WorldPrimitiveCollision
from curobo.rollout.cost.primitive_collision_cost import PrimitiveCollisionCost,PrimitiveCollisionCostConfig
from curobo.types.math import Pose

class TargetCostChannel:
    def __init__(self,mpc,ta,cfg,condition,session):
        self.mpc,self.ta,self.session,self.condition=mpc,ta,session,condition
        self.alpha=torch.tensor({'R':0.,'Z':0.,'L':1.,'H':2.}[condition],device=ta.device,dtype=ta.dtype)
        self.names=['human_patch_%02d'%i for i in range(cfg['proxy_capacity'])]
        self.world=WorldPrimitiveCollision(WorldCollisionConfig(tensor_args=ta,world_model=WorldConfig(cuboid=[Cuboid(name=n,pose=[10,0,0,1,0,0,0],dims=cfg['nominal_point_proxy_dims_m']) for n in self.names]),cache={'obb':len(self.names)}))
        for n in self.names:self.world.enable_obstacle(n,False)
        self.fixed_dims=np.array(cfg['nominal_point_proxy_dims_m']);self.enabled=0;self.optimizers=[];self.info=[]
        for i,opt in enumerate(mpc.solver.optimizers):
            assert not opt.cu_opt_init,'hook must precede optimizer capture'
            env=opt.rollout_fn.primitive_collision_cost
            kw={f.name:getattr(env,f.name) for f in fields(PrimitiveCollisionCostConfig)}
            kw.update(world_coll_checker=self.world,weight=env.weight.clone(),activation_distance=env.activation_distance.clone(),return_loss=True)
            assert float(env.activation_distance[0])>0
            cost=PrimitiveCollisionCost(PrimitiveCollisionCostConfig(**kw))
            b,h=opt.total_num_particles,opt.horizon
            snap={k:torch.zeros((b,h),device=ta.device,dtype=ta.dtype) for k in ['base','target','scaled','total','used']}
            snap.update(discounted=torch.zeros((opt.n_problems,opt.particles_per_problem),device=ta.device,dtype=ta.dtype),weights=torch.zeros((opt.n_problems,opt.particles_per_problem),device=ta.device,dtype=ta.dtype),calls=torch.zeros((),device=ta.device,dtype=torch.int64))
            original=opt.rollout_fn.cost_fn
            if condition!='R':
                def hooked(rollout,state,action_batch=None,_old=original,_cost=cost,_snap=snap):
                    base=_old(state,action_batch);target=_cost.forward(state.robot_spheres)
                    scaled=self.alpha*target;total=base+scaled
                    _snap['base'].copy_(base);_snap['target'].copy_(target);_snap['scaled'].copy_(scaled);_snap['total'].copy_(total);_snap['calls'].add_(1)
                    return total
                opt.rollout_fn.cost_fn=types.MethodType(hooked,opt.rollout_fn)
            distribution=opt._update_distribution
            def audited(optimizer,trajectories,_old=distribution,_snap=snap):
                costs=trajectories.costs
                _snap['used'].copy_(costs.reshape(_snap['used'].shape))
                _snap['discounted'].copy_(optimizer._compute_total_cost(costs));_snap['weights'].copy_(optimizer._exp_util_from_costs(costs))
                if self.condition=='R':
                    _snap['base'].copy_(costs.reshape(_snap['base'].shape));_snap['total'].copy_(_snap['base']);_snap['calls'].add_(1)
                return _old(trajectories)
            opt._update_distribution=types.MethodType(audited,opt)
            self.optimizers.append((opt,cost,snap,original))
            self.info.append({'index':i,'optimizer_type':type(opt).__name__,'connected':condition!='R','not_aux_rollout':opt.rollout_fn is not mpc.rollout_fn,'pre_capture':not opt.cu_opt_init,'shape':[b,h],'weight':cost.weight.cpu().tolist(),'activation_m':cost.activation_distance.cpu().tolist(),'use_sweep':cost.use_sweep,'sweep_steps':cost.sweep_steps,'speed_metric':cost.use_speed_metric,'return_loss':cost.return_loss})
        assert self.optimizers and all(x['optimizer_type']=='ParallelMPPI' and x['not_aux_rollout'] for x in self.info)
        self.addresses=self.pointers();self.samples=(session/'candidate-summaries.jsonl').open('w')
        (session/'cost-connection.json').write_text(json.dumps({'condition':condition,'alpha':float(self.alpha),'a':1,'mass_used':False,'shape_dims_m':self.fixed_dims.tolist(),'slots':len(self.names),'original_enlarged_geometry_enabled':False,'optimizers':self.info,'pointer_addresses':self.addresses,'aux_and_safety_costs':'unchanged; reported constraints do not include target penalty','internal_weight_role':'environment-matched proximity cost scale','external_alpha_role':'additional target penalty multiplier only','same_sampling_and_settings':True},indent=2))
    def pointers(self):return [self.alpha.data_ptr()]+[x.data_ptr() for x in self.world._cube_tensor_list[:3]]
    def update(self,anchors):
        assert self.pointers()==self.addresses
        for i,n in enumerate(self.names):
            active=i<len(anchors)
            if active:self.world.update_obb_pose(w_obj_pose=Pose.from_list([*anchors[i],1,0,0,0],self.ta),name=n)
            self.world.enable_obstacle(n,active)
        self.enabled=len(anchors)
        indices=[self.world.get_obb_idx(n) for n in self.names]
        assert int(self.world._cube_tensor_list[2][0,indices].sum())==self.enabled
        assert np.allclose(self.world._cube_tensor_list[0][0,indices,:3].cpu().numpy(),self.fixed_dims)
    def save_sample(self,step,phase):
        torch.cuda.synchronize()
        for i,(opt,cost,snap,original) in enumerate(self.optimizers):
            arrays={k:v.detach().cpu().numpy().copy() for k,v in snap.items()}
            for k in ['base','target','scaled','total','used','discounted','weights']:assert np.isfinite(arrays[k]).all(),k
            assert np.array_equal(arrays['total'],arrays['used'])
            assert np.allclose(arrays['total'],arrays['base']+arrays['scaled'],rtol=1e-6,atol=1e-4)
            assert np.allclose(arrays['scaled'],float(self.alpha)*arrays['target'],rtol=1e-6,atol=1e-4)
            summary={'step':step,'phase':phase,'alpha':float(self.alpha),'enabled_target_patches':self.enabled,'optimizer':i,'calls':int(arrays['calls']),'capture_ready':bool(opt.cu_opt_init),'pointers_unchanged':self.pointers()==self.addresses,'selected_index_from_actual_weights':int(arrays['weights'][0].argmax()),'weights_sum':float(arrays['weights'].sum()),'target_cost_available':self.condition!='R'}
            for k in ['base','target','scaled','total','discounted','weights']:
                v=arrays[k];summary[k]={'min':float(v.min()),'mean':float(v.mean()),'max':float(v.max()),'std':float(v.std())}
            self.samples.write(json.dumps(summary)+'\n');self.samples.flush()
            # 32 candidates at representative times, plus all actual utility weights.
            ids=np.linspace(0,opt.total_num_particles-1,32,dtype=int)
            np.savez_compressed(self.session/('cost-sample-%04d-%d.npz'%(step,i)),candidate_ids=ids,**{k:arrays[k][ids] for k in ['base','target','scaled','total','used']},discounted=arrays['discounted'],weights=arrays['weights'])
    def finish(self):
        self.samples.close();torch.cuda.synchronize()
        (self.session/'cost-final.json').write_text(json.dumps({'alpha':float(self.alpha),'enabled_patches':self.enabled,'pointer_stability':self.pointers()==self.addresses,'optimizers':[{'captured':bool(o.cu_opt_init),'calls':int(s['calls']),'connected':self.condition!='R'} for o,c,s,f in self.optimizers]},indent=2))
