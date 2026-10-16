"""Display-only helpers and read-only per-step candidate statistics (no effect on cost geometry or solver)."""
import numpy as np


def add_obstacle_guide(guide):
    from isaacsim.core.api.objects import VisualCuboid
    b = guide['board']; lo = np.array(b['min_m']); hi = np.array(b['max_m'])
    # VisualCuboid has no collider and is never added to a cuRobo world.
    return VisualCuboid('/World/obstacle_guide', position=(lo + hi) / 2, scale=hi - lo, size=1., color=np.array([.8, .2, .9]))


class SphereVis:
    """Buffered robot collision spheres (radius > 0) as visual-only prims."""
    def __init__(self):
        self.prims = None

    def update(self, spheres):
        from isaacsim.core.api.objects import VisualSphere
        s = spheres.detach().cpu().numpy(); s = s[s[:, 3] > 0]
        if self.prims is None:
            self.prims = [VisualSphere('/World/robot_spheres/s%02d' % i, position=x[:3], radius=float(x[3]), color=np.array([.1, .6, 1.])) for i, x in enumerate(s)]
        for p, x in zip(self.prims, s):
            p.set_world_pose(position=x[:3])


def placement_readout(observer, guide):
    b = guide['board']; lo = np.array(b['min_m']); hi = np.array(b['max_m'])
    st = observer.stats; pts = observer.points
    if not len(pts):
        return 'ROI points 0 | patches 0 | board not observed in ROI'
    on = np.all((pts >= lo - [.015, .015, .015]) & (pts <= hi + [.015, .015, .015]), 1)
    med = np.median(pts[on], 0) if on.any() else np.median(pts, 0)
    want = np.array([b['front_face_x_m'], *b['center_m'][1:]])
    d = med - want
    # Camera view = standing behind the camera, looking along the lens axis. Robot +y = camera-view left.
    hint = lambda v, pos, neg: ('%s %.1f cm' % (pos if v > 0 else neg, abs(100 * v))) if abs(v) > .01 else 'ok'
    return ('ROI points %d | patches %d | on-board %.0f%% off-board %.0f%%\n'
            'board - guide: x %+.1f y %+.1f z %+.1f cm  ->  depth: %s | side: %s | height: %s'
            % (st.get('target_depth_points', 0), st.get('observed_anchor_count', 0), 100 * on.mean(), 100 * (1 - on.mean()),
               100 * d[0], 100 * d[1], 100 * d[2], hint(d[0], 'move toward lens', 'move away from lens'),
               hint(d[1], 'move right (camera view)', 'move left (camera view)'), hint(d[2], 'lower', 'raise')))


def candidate_target_stats(channel):
    """Last MPPI iteration of this step: candidates with added target cost > 0 and their utility-weight mass."""
    if channel.condition == 'R':
        return dict(target_nonzero_candidates=None, target_weight_mass=None, best_candidate_target=None)
    opt, cost, snap, old = channel.optimizers[0]
    target = snap['target'].sum(-1); w = snap['weights'].reshape(-1)
    nz = target > 0
    return dict(target_nonzero_candidates=int(nz.sum()), target_weight_mass=float(w[nz].sum() / w.sum()) if bool(w.sum() > 0) else None,
                best_candidate_target=float(target[int(w.argmax())]))
