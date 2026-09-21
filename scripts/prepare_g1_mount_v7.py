"""Prepare frozen interfaces and nominal shell samples for the native Fusion build."""
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'artifacts' / 'g1_mount_v7'
sys.path.insert(0, str(OUT / 'python_packages'))
import numpy as np
import trimesh
from shapely.geometry import Polygon, LineString, MultiPoint

BUNDLE = BASE / 'artifacts/g1_mount_v7/reference_inputs'
manifest = json.loads((BUNDLE / 'current_design/cad/design_manifest.json').read_text(encoding='utf-8'))
angle = np.deg2rad(21.0)
rotation = np.array([[0,-np.sin(angle),np.cos(angle)],[-1,0,0],[0,-np.cos(angle),-np.sin(angle)]])
translation = np.array([245,0,127.5]) - rotation @ np.array([90,135,0])
transform = np.eye(4)
transform[:3,:3] = rotation
transform[:3,3] = translation

def board(y, z):
    p = rotation @ np.array([-8.5,y,z]) + translation
    return [float(p[0]), float(p[2])]

upper = 265.0961682
lower = 70.8961682
by_upper = 22.4925663052
by_lower = 226.6865302522
outer = MultiPoint([[84,lower-20],[84,upper+20],[104,upper+20],
                   board(by_upper-18,9.4),board(by_upper-18,33.4),
                   board(by_lower+18,9.4),board(by_lower+18,33.4)]).convex_hull
inner = outer.buffer(-18, join_style=2)
brace = LineString([[94,lower],board(by_upper,21.4)]).buffer(9, cap_style=2)
windows = inner.difference(brace)
assert len(windows.geoms) == 2

mesh = trimesh.load_mesh(BUNDLE / 'robot_model/torso_link_rev_1_0.STL',process=False)
tri = np.array(mesh.triangles)*1000
a=tri[:,0,1:]
v0=tri[:,1,1:]-a
v1=tri[:,2,1:]-a
den=v0[:,0]*v1[:,1]-v1[:,0]*v0[:,1]
usable=np.abs(den)>1e-10

def shell_x(y,z):
    q=np.array([y,z])-a
    u=np.zeros(len(tri)); v=np.zeros(len(tri))
    u[usable]=(q[usable,0]*v1[usable,1]-v1[usable,0]*q[usable,1])/den[usable]
    v[usable]=(v0[usable,0]*q[usable,1]-q[usable,0]*v0[usable,1])/den[usable]
    inside=usable&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
    assert inside.any(), (y,z)
    x=tri[:,0,0]+u*(tri[:,1,0]-tri[:,0,0])+v*(tri[:,2,0]-tri[:,0,0])
    return float(x[inside].max())

roots={}
for name,p in manifest['nominal_m6_root_centers_torso_mm'].items():
    samples=[]
    for dy in np.linspace(-14,14,9):
        u=p[1]+dy
        section=[]
        for dv in np.linspace(-14,14,9):
            section.append([float(dv),104-shell_x(u,p[2]-dv)])
        samples.append({'u':float(u),'shell_v_depth':section})
    roots[name]={'center':p,'sections':samples,'washer_depth':104-p[0]-12}

data={
    'name':'G1 V7 broad crossbars and flat side plates',
    'scope':'New native Fusion structure; existing carrier and camera layout unchanged.',
    'source_manifest':(BUNDLE/'current_design/cad/design_manifest.json').relative_to(BASE).as_posix(),
    'physical_fit_status_file':'FIT_STATUS.md',
    'torso_T_board_mm':transform.tolist(),
    'carrier_print_to_board_translation_mm':[-15,-15,0],
    'upper_z':upper,'lower_z':lower,'crossbar_front_x':104,'crossbar_depth':20,
    'crossbar_half_width':115,'crossbar_height':40,'plate_outer_y':125,
    'plate_thickness':10,'board_tab_inner_y':90.5,'board_tab_depth':12,
    'side_outer_xz':list(outer.exterior.coords)[:-1],
    'side_windows_xz':[list(w.exterior.coords)[:-1] for w in windows.geoms],
    'board_tabs_xz':[[board(y-14,9.4),board(y+14,9.4),board(y+14,21.4),board(y-14,21.4)] for y in [by_upper,by_lower]],
    'board_connector_centers_torso_mm':manifest['board_connector_centers_torso_mm'],
    'board_y_centers':[by_upper,by_lower],
    'roots':roots,
    'm6_clearance':6.8,'m6_well':16,'m5_clearance':5.4,'m4_clearance':4.5,
    'notes':['Shell contact surfaces are sampled nominal CAD, not verified on this robot.',
             'Physical test: upper M6 x 25 reportedly works; lower M6 x 25 does not engage through R2 coupons. Final M6 hardware is unresolved; see FIT_STATUS.md.',
             'M6 x 20 projects 6.4 mm past the nominal outer shell with a 1.6 mm washer; usable thread engagement depends on the unmeasured insert recess.',
             'Four M4 x 30 retain the existing printed carrier.',
             'Eight M5 x 30, eight washers and eight standard M5 hex nuts join the four structural parts.',
             'Side plates print with their large exterior face on the bed; tabs grow upward.',
             'Crossbars print with the broad forward face on the bed; shell pads grow upward.']
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'design_inputs.json').write_text(json.dumps(data,indent=2))
print(json.dumps({'outer_xz':data['side_outer_xz'],'windows':data['side_windows_xz'],'contact_centers':{k:v['center'] for k,v in roots.items()}},indent=2))
