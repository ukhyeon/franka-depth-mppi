"""Real-hand observation: per-frame analysis, depth overlay and GUI text.
Display/logging only: never changes ROI, patch extraction (target_margin.py), costs or the solver."""
import json
from collections import deque
from pathlib import Path
import numpy as np

FACES = ['x-(lens side)', 'x+(far side)', 'y-(camera right)', 'y+(camera left)', 'z-(bottom)', 'z+(top)']


class HandObservation:
    def __init__(self, root, cfg, scene, hand):
        self.cfg, self.hand = cfg, hand
        self.lo = np.array(cfg['manual_target_world_bounds_m']['min']); self.hi = np.array(cfg['manual_target_world_bounds_m']['max'])
        self.bin = cfg['patch_bin_yz_m']; self.minpts = cfg['min_points_per_bin']
        T = scene['camera_to_robot_pose_wxyz']
        from scipy.spatial.transform import Rotation
        self.R = Rotation.from_quat([*T[4:], T[3]]).as_matrix(); self.t = np.array(T[:3])
        # Nominal obstacle-free A->B sweep (buffered spheres) near the ROI: "expected gap" if the robot did not avoid.
        sw = hand['nominal_sweep']; run = (Path(root) / sw['run']).resolve()
        S = np.load(run / 'robot-spheres.npz')['spheres'][sw['AB_steps'][0] - 1:sw['AB_steps'][1]]
        links = np.array([x['link'] for x in json.loads((Path(root) / 'analysis/sphere-links.json').read_text())])
        c = (self.lo + self.hi) / 2; near = np.linalg.norm(S[..., :3] - c, axis=-1) < .35
        self.sweep = S[near]; self.sweep_link = np.broadcast_to(links, S.shape[:2])[near]
        self.sweep_all = S.reshape(-1, 4)[::3]   # whole nominal A->B sweep, for background-on-path check
        self.hist = deque(maxlen=15); self.counts = deque(maxlen=60); self.prev_keys = None; self.hold_ref = []
        self.last = None

    def points(self, depth, K):
        ys, xs = np.nonzero(depth > 0); z = depth[ys, xs]
        pc = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z]) @ self.R.T + self.t
        return pc, xs, ys

    def analyze(self, depth, K, anchors, stage, robot_spheres=None):
        pc, xs, ys = self.points(depth, K)
        roi = np.all((pc >= self.lo) & (pc < self.hi), 1)
        grown = np.all((pc >= self.lo - .05) & (pc < self.hi + .05), 1) & ~roi
        spill = {}
        for i, f in enumerate(FACES):
            a = i // 2; s = pc[grown]
            spill[f] = int(((s[:, a] < self.lo[a]) if i % 2 == 0 else (s[:, a] >= self.hi[a])).sum())
        P = pc[roi]; out = dict(stage=stage, valid_points=int(len(pc)), roi_points=int(len(P)), spill_by_face=spill)
        front = float(np.percentile(P[:, 0], 5)) if len(P) else None
        behind = (P[:, 0] > front + self.hand['behind_hand_m']) if len(P) else np.zeros(0, bool)
        out.update(front_x_p5=front, behind_points=int(behind.sum()), centroid=P.mean(0).round(4).tolist() if len(P) else None,
                   extent_min=P.min(0).round(3).tolist() if len(P) else None, extent_max=P.max(0).round(3).tolist() if len(P) else None)
        # Same bins as target_margin.py (y,z 2 cm; >=12 points); anchors are the module's post-filter output.
        grid = np.zeros((8, 8), dtype='<U1'); grid[:] = ' '; mixed = 0; selected = 0
        if len(P):
            b = np.floor((P[:, 1:] - self.lo[1:]) / self.bin).astype(int)
            for (iy, iz) in {tuple(x) for x in b}:
                m = (b[:, 0] == iy) & (b[:, 1] == iz); n = int(m.sum())
                spread = float(np.ptp(P[m, 0])) if n else 0.
                if n < self.minpts:grid[iz, iy] = '.'; continue
                selected += 1; grid[iz, iy] = 'x'
                if spread > self.hand['bin_mixed_depth_m']:grid[iz, iy] = 'm'; mixed += 1
        keys = set()
        for a in np.asarray(anchors).reshape(-1, 3):
            iy, iz = np.floor((a[1:] - self.lo[1:]) / self.bin).astype(int); keys.add((int(iy), int(iz)))
            if 0 <= iy < 8 and 0 <= iz < 8:grid[iz, iy] = 'M' if grid[iz, iy] == 'm' else '#'
        jac = None if self.prev_keys is None else (len(keys & self.prev_keys) / max(1, len(keys | self.prev_keys)))
        self.prev_keys = keys; self.counts.append(len(keys))
        moving = None
        if out['centroid'] is not None:
            self.hist.append(np.array(out['centroid']))
            if len(self.hist) >= 5:moving = bool(np.linalg.norm(np.array(out['centroid']) - np.median(np.array(self.hist), 0)) > .01)
        if stage == 'HOLD' and out['centroid'] is not None:self.hold_ref.append(len(P))
        dropout = bool(stage == 'HOLD' and self.hold_ref and len(P) < .5 * np.median(self.hold_ref))
        gap = gap_link = None
        if len(P) and len(self.sweep):
            q = P[::max(1, len(P) // 800)]; best = np.inf
            for i in range(0, len(self.sweep), 512):
                s = self.sweep[i:i + 512]; d = np.linalg.norm(q[None, :, :] - s[:, None, :3], axis=-1).min(1) - s[:, 3]
                j = int(d.argmin())
                if d[j] < best:best, gap_link = float(d[j]), str(self.sweep_link[i + j])
            gap = best
        live = None
        if robot_spheres is not None and len(P):
            s = robot_spheres[robot_spheres[:, 3] > 0]; q = P[::max(1, len(P) // 800)]
            live = float((np.linalg.norm(q[None] - s[:, None, :3], axis=-1).min(1) - s[:, 3]).min())
        bg = []
        if len(pc):
            H, W = depth.shape
            for r in range(3):
                for c in range(3):
                    m = (ys >= r * H / 3) & (ys < (r + 1) * H / 3) & (xs >= c * W / 3) & (xs < (c + 1) * W / 3)
                    if m.sum() > 500:bg.append(dict(region=['top', 'middle', 'bottom'][r] + '-' + ['left', 'center', 'right'][c], points=int(m.sum()), median_depth_m=round(float(np.median(depth[ys[m], xs[m]])), 3)))
        path_bg = None
        Q = pc[~roi]
        if len(Q) and (stage == 'BACKGROUND' or len(P) < 200):
            q = Q[::max(1, len(Q) // 800)]; best = (np.inf, None)
            for i in range(0, len(self.sweep_all), 1024):
                s = self.sweep_all[i:i + 1024]; d = np.linalg.norm(q[None] - s[:, None, :3], axis=-1) - s[:, 3:4]
                j = np.unravel_index(int(d.argmin()), d.shape)
                if d[j] < best[0]:best = (float(d[j]), q[j[1]])
            path_bg = dict(gap_m=best[0], depth_from_lens_m=round(float(best[1][0] - self.t[0]), 3), camera_left_m=round(float(best[1][1]), 3), above_lens_m=round(float(best[1][2] - self.t[2]), 3))
        out['hint'] = self.hint(pc, out, len(keys), moving)
        out.update(background_vs_nominal_path=path_bg,bins_selected=selected, bins_mixed_depth=mixed, anchors=int(len(keys)), anchor_keys=sorted(keys), anchors_xyz=np.asarray(anchors).reshape(-1, 3).round(4).tolist(),
                   patch_grid=[''.join(r) for r in grid[::-1]], jaccard_prev=jac, moving=moving, hold_dropout=dropout,
                   nominal_path_gap_m=gap, nominal_gap_link=gap_link, live_robot_gap_m=live, background_regions=bg,
                   patch_count_window=[min(self.counts), max(self.counts)])
        self.last = (out, pc, xs, ys, roi, grown, behind); return out

    def hint(self, pc, o, npatch, moving):
        r = self.hand['recommended_hand']; tol = .015
        E = pc[np.all((pc >= self.lo - .15) & (pc <= self.hi + .15), 1)]
        if len(E) < 300:
            return dict(ok=False, text='NO HAND NEAR ROI' if o['stage'] != 'BACKGROUND' else ('BACKGROUND OK' if o['roi_points'] <= self.hand['background_limits']['roi_points_max'] else 'CLEAR ROI'))
        top = float(np.percentile(E[:, 2], 97)); Hn = E[E[:, 2] > top - .15]
        front = float(np.percentile(Hn[:, 0], 5)); yc = float(np.median(Hn[:, 1]))
        d = dict(depth=front - r['palm_face_x_m'], side=yc - r['center_y_m'], height=top - r['fingertip_z_m'])
        act = []
        if d['depth'] < -tol:act.append('MOVE AWAY %d cm' % round(-100 * d['depth']))
        if d['depth'] > tol:act.append('MOVE CLOSER %d cm' % round(100 * d['depth']))
        if d['side'] > tol:act.append('RIGHT %d cm' % round(100 * d['side']))      # robot +y = camera-view left
        if d['side'] < -tol:act.append('LEFT %d cm' % round(-100 * d['side']))
        if d['height'] > tol:act.append('DOWN %d cm' % round(100 * d['height']))
        if d['height'] < -tol:act.append('UP %d cm' % round(-100 * d['height']))
        if not act and moving:act.append('HAND MOVING - REST ELBOW')
        if not act and npatch < 20:act.append('FEW PATCHES (%d)' % npatch)
        ok = not act
        return dict(ok=ok, text='OK - HOLD STILL' if ok else ' | '.join(act), offsets_m={k: round(v, 3) for k, v in d.items()})

    def text(self, o):
        lines = ['>>> ' + o['hint']['text'] + '   (directions as seen standing behind the camera)', '']
        lines += ['stage %s | valid %d | ROI %d | patches %d (last %d frames: %d-%d) | Jaccard %s' % (o['stage'], o['valid_points'], o['roi_points'], o['anchors'], len(self.counts), *o['patch_count_window'], '-' if o['jaccard_prev'] is None else '%.2f' % o['jaccard_prev'])]
        if o['stage'] == 'BACKGROUND' or o['roi_points'] < 200:
            ok = o['roi_points'] <= self.hand['background_limits']['roi_points_max']
            lines.append('BACKGROUND CHECK: ROI points %d -> %s' % (o['roi_points'], 'OK (ROI empty)' if ok else 'CLEAR objects inside ROI'))
            for b in o['background_regions']:lines.append('  within 0.7 m: image %s  %d pts  depth %.2f m  -> remove' % (b['region'], b['points'], b['median_depth_m']))
            pb = o.get('background_vs_nominal_path')
            if pb and pb['gap_m'] < .10:lines.append('  ON ROBOT PATH (gap %+.0f mm): object at depth %.2f m, %.0f cm %s of axis, %.0f cm %s lens height -> remove' % (1000 * pb['gap_m'], pb['depth_from_lens_m'], 100 * abs(pb['camera_left_m']), 'left' if pb['camera_left_m'] > 0 else 'right', 100 * abs(pb['above_lens_m']), 'above' if pb['above_lens_m'] > 0 else 'below'))
        else:
            sp = {k: v for k, v in o['spill_by_face'].items() if v > 50}
            lines.append('hand in ROI: extent y %.3f..%.3f z %.3f..%.3f m | front x %.3f | outside ROI near faces: %s' % (o['extent_min'][1], o['extent_max'][1], o['extent_min'][2], o['extent_max'][2], o['front_x_p5'], sp or 'none'))
            lines.append('behind-hand points in ROI (wrist/arm/background, >4 cm behind front): %d | mixed-depth bins: %d' % (o['behind_points'], o['bins_mixed_depth']))
        lines.append('expected gap to nominal A->B sweep: %s %s | live robot gap: %s' % ('-' if o['nominal_path_gap_m'] is None else '%+.1f mm' % (1000 * o['nominal_path_gap_m']), o['nominal_gap_link'] or '', '-' if o['live_robot_gap_m'] is None else '%+.1f mm' % (1000 * o['live_robot_gap_m'])))
        flags = [f for f, v in [('MOVING >1 cm', o['moving']), ('HOLD DROPOUT', o['hold_dropout'])] if v]
        if flags:lines.append('WARNING: ' + ', '.join(flags))
        lines.append('patch grid (camera view, top row = high z; # patch, M patch w/ mixed depth, x/m bin>=12 pts w/o patch, . <12 pts):')
        lines += ['  |' + r[::-1] + '|' for r in o['patch_grid']]
        return '\n'.join(lines)

    def overlay(self, depth, K):
        import cv2
        o, pc, xs, ys, roi, grown, behind = self.last
        shade = np.clip(depth * (255 / .7), 0, 255).astype(np.uint8); img = cv2.applyColorMap(shade, cv2.COLORMAP_VIRIDIS); img[depth <= 0] = 0
        ri = np.nonzero(roi)[0]
        img[ys[ri], xs[ri]] = (255, 255, 255)
        if behind.any():img[ys[ri[behind]], xs[ri[behind]]] = (0, 0, 255)
        gi = np.nonzero(grown)[0]; img[ys[gi], xs[gi]] = (0, 140, 255)
        proj = lambda p: (lambda c: (int(K[0, 0] * c[0] / c[2] + K[0, 2]), int(K[1, 1] * c[1] / c[2] + K[1, 2])))(self.R.T @ (np.asarray(p) - self.t))
        C = [[x, y, z] for x in (self.lo[0], self.hi[0]) for y in (self.lo[1], self.hi[1]) for z in (self.lo[2], self.hi[2])]
        for a in range(8):
            for b in range(a + 1, 8):
                if sum(C[a][k] != C[b][k] for k in range(3)) == 1:cv2.line(img, proj(C[a]), proj(C[b]), (0, 220, 255), 1)
        for a in o['anchors_xyz']:cv2.circle(img, proj(a), 3, (0, 255, 0), -1)
        h = o['hint']; col = (0, 200, 0) if h['ok'] else (0, 0, 255)
        cv2.rectangle(img, (0, 0), (639, 40), (0, 0, 0), -1); cv2.putText(img, h['text'], (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .8, col, 2, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


class HandPanel:
    """Isaac Sim UI window: depth overlay (white ROI points, orange spill, red behind-hand, yellow ROI box, green patches) + readout."""
    def __init__(self, title='Depth 0-0.7 m | white ROI points | orange outside-ROI near faces | red >4 cm behind hand | yellow ROI | green patches'):
        import omni.ui as ui
        self.provider = ui.ByteImageProvider(); self.window = ui.Window('D435i hand observation', width=700, height=640)
        with self.window.frame:
            with ui.VStack():
                ui.Label(title)
                ui.ImageWithProvider(self.provider, height=480)
                self.label = ui.Label('Waiting for depth', word_wrap=True)

    def update(self, rgba, text):
        self.provider.set_bytes_data(rgba.ravel().tolist(), [rgba.shape[1], rgba.shape[0]]); self.label.text = text


class CaptureGuide:
    """Per-frame capture instructions (every recorded frame, cheap). Precise placement is NOT required:
    the tape builder registers the hand afterwards. Checks: empty view, hand visible, distance, image edges, stillness."""
    def __init__(self, scene, hand):
        from scipy.spatial.transform import Rotation
        from register_hand import stable_runs
        T = scene['camera_to_robot_pose_wxyz']; self.R = Rotation.from_quat([*T[4:], T[3]]).as_matrix(); self.t = np.array(T[:3])
        c = hand['capture']; self.target = c['hold_frames_target']; g = c['live_guide']; self.empty_max = g['empty_view_max_points']; self.dr = g['hand_depth_range_m']
        self.tol = c['stable_frame_rule']['roi_centroid_shift_max_m']; self.stable_runs = stable_runs
        self.cents = []; self.best = 0; self.state = None

    def track(self, depth, K, stage):
        ys, xs = np.nonzero(depth > 0); z = depth[ys, xs]; n = len(z); hand_px = None; st = dict(stage=stage, points=int(n))
        if stage == 'BACKGROUND':
            if n <= self.empty_max:msg, ok = 'STEP 1  VIEW EMPTY  ->  press PLACING', True
            else:
                r = ['top', 'middle', 'bottom'][min(2, int(np.median(ys) // 160))] + '-' + ['left', 'center', 'right'][min(2, int(np.median(xs) // 214))]
                msg, ok = 'STEP 1  REMOVE OBJECTS within 70 cm (%s of image)' % r, False
        elif stage == 'REMOVED':msg, ok = 'SAVING ... (auto stop in 3 s)', True
        else:
            if n < 1500:msg, ok = 'STEP 2  PUT HAND IN FRONT OF CAMERA (palm to lens, fingers up)', False; self._hold(None, stage)
            else:
                pc = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z]) @ self.R.T + self.t
                import cv2
                top = np.percentile(pc[:, 2], 97); m = pc[:, 2] > top - .15
                img = np.zeros(depth.shape, np.uint8); img[ys[m], xs[m]] = 1; img = cv2.dilate(img, np.ones((5, 5), np.uint8))
                nlab, lab, stats, _ = cv2.connectedComponentsWithStats(img, 8)
                if nlab > 1:   # hand = largest connected blob in the top band (drops body/arm fragments)
                    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); m &= lab[ys, xs] == big
                d = float(np.median(z[m])); hand_px = (ys[m], xs[m])
                edge = bool((ys[m].min() <= 28) or (xs[m].min() <= 36) or (xs[m].max() >= 603))
                st.update(hand_depth_m=round(d, 3))
                if d < self.dr[0]:msg, ok = 'MOVE HAND AWAY: %d cm -> %d-%d cm' % (100 * d, 100 * self.dr[0], 100 * self.dr[1]), False
                elif d > self.dr[1]:msg, ok = 'MOVE HAND CLOSER: %d cm -> %d-%d cm' % (100 * d, 100 * self.dr[0], 100 * self.dr[1]), False
                elif edge:msg, ok = 'HAND CUT AT IMAGE EDGE -> move toward center', False
                else:msg, ok = None, True
                run = self._hold(pc[m].mean(0) if ok else None, stage)
                if msg is None:
                    if stage == 'PLACING':msg = 'STEP 2  HAND OK (%d cm)  ->  press HOLD, then keep still' % (100 * d)
                    elif run >= self.target:msg = 'DONE  %d/%d still frames  ->  remove hand, press REMOVED' % (run, self.target)
                    elif run <= 1 and len(self.cents) > 5:msg, ok = 'MOVED -> keep still (count restarted)  %d/%d' % (run, self.target), False
                    else:msg = 'STEP 3  KEEP STILL  %d/%d frames  (best %d)' % (run, self.target, self.best)
        st.update(msg=msg, ok=ok, still=self.best if stage == 'HOLD' else None); self.state = st; self.hand_px = hand_px; return st

    def _hold(self, c, stage):
        if stage != 'HOLD':self.cents = []; return 0
        self.cents.append(c); ids = self.stable_runs(self.cents, self.tol)
        cur = ids[-1]; run = 0 if cur is None else sum(1 for i in ids if i == cur); self.best = max(self.best, run); return run

    def overlay(self, depth):
        import cv2
        shade = np.clip(depth * (255 / .7), 0, 255).astype(np.uint8); img = cv2.cvtColor(255 - shade, cv2.COLOR_GRAY2BGR); img[depth <= 0] = 0
        if self.hand_px is not None:img[self.hand_px] = (255, 255, 255)
        s = self.state; col = (0, 200, 0) if s['ok'] else (0, 0, 255)
        cv2.rectangle(img, (0, 0), (639, 44), (0, 0, 0), -1)
        cv2.putText(img, s['msg'][:52], (6, 30), cv2.FONT_HERSHEY_SIMPLEX, .72, col, 2, cv2.LINE_AA)
        if len(s['msg']) > 52:cv2.putText(img, s['msg'][52:], (6, 470), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    def text(self):
        s = self.state
        return '>>> %s\n\nstage %s | depth points %d | hand distance %s | still frames best %s / %d\n(white = detected hand; exact position is corrected later by software registration)' % (
            s['msg'], s['stage'], s['points'], '-' if s.get('hand_depth_m') is None else '%d cm' % (100 * s['hand_depth_m']), '-' if s.get('still') is None else s['still'], self.target)
