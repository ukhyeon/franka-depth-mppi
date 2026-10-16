"""Read-only Python-side MPC failure audit; never changes results or constraints."""
class FailureProbe:
    def __init__(self, mpc):
        self.mpc = mpc
        self.attempts = 0
        self.attempt_results = []
        self.sequence = None
        self.start = None
        original_once = mpc._step_once
        def once(*args, **kwargs):
            self.attempts += 1
            result = original_once(*args, **kwargs)
            self.attempt_results.append(dict(feasible=result.metrics.feasible.cpu().tolist(),constraint=result.metrics.constraint.cpu().tolist(),q=result.action.position.cpu().tolist()))
            return result
        mpc._step_once = once
        original_command = mpc.solver.safety_rollout.get_robot_command
        def command(state, sequence, *args, **kwargs):
            self.sequence = sequence.detach().clone()
            self.start = state.clone()
            return original_command(state, sequence, *args, **kwargs)
        mpc.solver.safety_rollout.get_robot_command = command
    def begin(self):
        self.attempts = 0
        self.attempt_results = []
    def components(self, kin):
        rollout = self.mpc.rollout_fn
        state = kin.state_seq
        if state.position.ndim == 2:state = state.unsqueeze(1)
        spheres = kin.robot_spheres
        if spheres.ndim == 3:spheres = spheres.unsqueeze(1)
        return dict(bound=rollout.bound_constraint.forward(state).cpu().tolist(),environment=rollout.primitive_collision_constraint.forward(spheres).cpu().tolist(),self=rollout.robot_self_collision_constraint.forward(spheres).cpu().tolist())
    def describe(self, actual, filtered, result, observer):
        rollout = self.mpc.rollout_fn
        current = rollout.compute_kinematics(actual)
        spheres=current.robot_spheres.reshape(-1,4)
        current_components=self.components(current)
        current_clearance=observer.clearance(spheres[spheres[:,3]>0])
        command=rollout.compute_kinematics(result.action)
        out=dict(attempts=self.attempts,attempts_detail=self.attempt_results,returned_constraint=result.metrics.constraint.cpu().tolist(),current=current_components,returned_command=self.components(command),native_warm_start_empty=self.mpc.solver._init_act_seq is None,returned_q=result.action.position.cpu().tolist(),measured_q=actual.position.cpu().tolist(),current_surface_clearance_m=current_clearance)
        predicted=rollout.dynamics_model.forward(self.start,self.sequence)
        out['predicted']=self.components(predicted)
        out['prediction_scope']='30 modeled states of final optimized sequence; audit only, not returned feasible scope'
        return out
