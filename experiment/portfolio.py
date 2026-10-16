"""Opt-in portfolio snapshots (RGB + aligned depth PNG). Only used with --portfolio-snapshots; the experiment
recording itself still stores no RGB. Files stay local under the session folder."""
import json, time
from pathlib import Path
import numpy as np
import cv2


def _label(img, text):
    out = img.copy(); cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA); return out


def depth_color(depth_m, near=.2, far=1.2):
    v = np.clip((depth_m - near) / (far - near), 0, 1); img = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth_m <= 0] = 0; return img


def save_snapshot(folder, rgb_bgr, raw_z16, scale, overlay_rgba=None, meta=None, far=1.2):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); tag = time.strftime('%Y%m%d-%H%M%S')
    depth = raw_z16.astype(np.float32) * scale                       # aligned to color, not clipped
    dimg = depth_color(depth, far=far)
    cv2.imwrite(str(folder / f'{tag}-rgb.png'), rgb_bgr)
    cv2.imwrite(str(folder / f'{tag}-depth.png'), dimg)
    np.save(folder / f'{tag}-depth-m.npy', depth)
    side = np.hstack([_label(rgb_bgr, 'D435i RGB'), _label(dimg, 'D435i depth (aligned, 0.2-%.1f m, near=red)' % far)])
    cv2.imwrite(str(folder / f'{tag}-rgb-depth.png'), side)
    if overlay_rgba is not None:
        ov = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(folder / f'{tag}-rgb-roi-patches.png'), np.hstack([_label(rgb_bgr, 'D435i RGB'), _label(ov, 'depth <=0.7 m: ROI (yellow), patches (green)')]))
    (folder / f'{tag}-meta.json').write_text(json.dumps(dict(meta or {}, depth_scale_m=scale, depth_display_m=[.2, far]), indent=1))
    return str(folder / f'{tag}-rgb-depth.png')
