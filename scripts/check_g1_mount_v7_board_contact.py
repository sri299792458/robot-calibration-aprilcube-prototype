"""Check the actual carrier mesh against all four exported support pads."""
from pathlib import Path
import json, sys
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'artifacts/g1_mount_v7'
sys.path.insert(0, str(OUT / 'python_packages'))
import numpy as np
import trimesh

D = json.loads((OUT / 'design_inputs.json').read_text(encoding='utf-8'))
T = np.array(D['torso_T_board_mm'])
carrier = trimesh.load_mesh(BASE / 'artifacts/g1_mount_v7/reference_inputs/current_design/cad/carrier_white_body.stl')
carrier.apply_translation([-15,-15,0])
report = {}
for side in ['left', 'right']:
    name = 'V7_side_' + side
    support = trimesh.load_mesh(OUT / 'cad' / (name + '_PRINT.stl'))
    transform = np.eye(4)
    transform[:3] = json.loads((OUT / (name + '_fusion.json')).read_text(encoding='utf-8'))['local_to_torso_mm']
    support.apply_transform(np.linalg.inv(T) @ transform)
    for row in ['upper', 'lower']:
        key = row + '_' + side
        q = trimesh.transform_points([D['board_connector_centers_torso_mm'][key]], np.linalg.inv(T))[0]
        ring = np.array([[q[0] + r*np.cos(a), q[1] + r*np.sin(a), 0]
                         for r in [3.0, 4.5, 6.0] for a in np.linspace(0, 2*np.pi, 24, endpoint=False)])
        carrier_starts = ring.copy(); carrier_starts[:, 2] = 50
        support_starts = ring.copy(); support_starts[:, 2] = -20
        c_hits, c_idx, _ = carrier.ray.intersects_location(carrier_starts, np.tile([0,0,-1], (len(ring),1)), multiple_hits=False)
        s_hits, s_idx, _ = support.ray.intersects_location(support_starts, np.tile([0,0,1], (len(ring),1)), multiple_hits=False)
        c_map = {int(i):float(p[2]) for i,p in zip(c_idx,c_hits)}; s_map = {int(i):float(p[2]) for i,p in zip(s_idx,s_hits)}
        gaps = [s_map[i]-c_map[i] for i in range(len(ring)) if i in c_map and i in s_map]
        axis_start = np.array([[q[0],q[1],-20]])
        axis_dir = np.array([[0,0,1]])
        c_axis = carrier.ray.intersects_location(axis_start, axis_dir, multiple_hits=False)[0]
        s_axis = support.ray.intersects_location(axis_start, axis_dir, multiple_hits=False)[0]
        report[key] = {'board_local_hole_center_mm':q.tolist(), 'pad_samples':len(ring), 'matched_samples':len(gaps),
                       'min_support_minus_carrier_z_mm':float(min(gaps)) if gaps else None, 'max_support_minus_carrier_z_mm':float(max(gaps)) if gaps else None,
                       'shared_screw_centerline_clear':len(c_axis)==0 and len(s_axis)==0}

(OUT / 'board_contact_qa.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert all(r['matched_samples']==r['pad_samples'] and abs(r['min_support_minus_carrier_z_mm'])<0.02
           and abs(r['max_support_minus_carrier_z_mm'])<0.02 and r['shared_screw_centerline_clear'] for r in report.values())
