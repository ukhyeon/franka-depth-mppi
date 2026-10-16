import json
import torch
from base_target_cost import TargetCostChannel as VerifiedTargetCostChannel
from policy import mass_policy
class TargetCostChannel(VerifiedTargetCostChannel):
    def __init__(self,mpc,ta,cfg,condition,session):
        super().__init__(mpc,ta,cfg,'R' if condition=='R' else 'L',session)
        self.condition=condition
        self.policy=mass_policy(condition)
        self.alpha.fill_(0 if condition=='R' else 1)
        for (opt,cost,snap,old),info in zip(self.optimizers,self.info):
            assert not opt.cu_opt_init
            env=opt.rollout_fn.primitive_collision_cost
            assert abs(float(env.activation_distance[0])-.025)<1e-8
            assert float(env.weight[0])==100000
            cost.activation_distance.fill_(self.policy['eta_calculated_m'])
            info['activation_m']=cost.activation_distance.cpu().tolist()
            info['environment_activation_m']=env.activation_distance.cpu().tolist()
            info['environment_weight']=env.weight.cpu().tolist()
            assert float(cost.weight[0])==100000
        assert self.pointers()==self.addresses
        self.policy.update(alpha_gpu=float(self.alpha),eta_gpu_m=float(self.optimizers[0][1].activation_distance[0]),weight=100000,slots=len(self.names),patch_dims_m=self.fixed_dims.tolist())
        (session/'mass-policy.json').write_text(json.dumps(self.policy,indent=2))
        p=session/'cost-connection.json';data=json.loads(p.read_text());data.update(condition=condition,alpha=float(self.alpha),mass_used=condition!='R',policy=self.policy,optimizers=self.info);p.write_text(json.dumps(data,indent=2))
        print('MASS_ETA_POLICY',json.dumps(self.policy),flush=True)
