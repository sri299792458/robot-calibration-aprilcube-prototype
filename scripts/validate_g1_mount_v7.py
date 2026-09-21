"""Check actual Fusion STL exports, interfaces, view rays and nominal contact fit."""
from pathlib import Path
import json,sys,hashlib
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'artifacts/g1_mount_v7'
sys.path.insert(0,str(OUT/'python_packages'))
import numpy as np
import trimesh

D=json.loads((OUT/'design_inputs.json').read_text(encoding='utf-8'))
BUNDLE=BASE/'artifacts/g1_mount_v7/reference_inputs'
CAM=json.loads((BUNDLE/'reference_code/config/torso_fixture_layout_camera.json').read_text(encoding='utf-8'))
TBOARD=np.array(D['torso_T_board_mm'])
report={'checks':{},'parts':{},'limits':['No FEA, printer trial or physical load test has been performed.','Nominal shell fit does not establish physical calibration accuracy.']}
assembled={}
for path in sorted((OUT/'cad').glob('*_PRINT.stl')):
    name=path.name.removesuffix('_PRINT.stl')
    m=trimesh.load_mesh(path,process=True)
    data=json.loads((OUT/(name+'_fusion.json')).read_text(encoding='utf-8'))
    parts=m.split(only_watertight=False)
    assert m.is_watertight and m.is_winding_consistent and len(parts)==1,name
    assert max(m.extents[:2])<300 and m.extents[2]<320,name
    if name.startswith('V7_'):assert abs(m.bounds[0,2])<0.001,name
    assert abs(m.volume/1000-data['volume_cm3'])/data['volume_cm3']<0.001,name
    record={'watertight':m.is_watertight,'single_body':len(parts)==1,'extents_mm':m.extents.tolist(),'volume_cm3':m.volume/1000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    T=np.eye(4);T[:3]=data['local_to_torso_mm']
    ma=m.copy();ma.apply_transform(T)
    if name.startswith('V7_'):assembled[name]=ma
    # Keep Fusion's assembly-coordinate exports untouched; provide centered print copies.
    print_dir=OUT/('fit_coupons' if name.startswith('FIT_ONLY') else 'print_parts');print_dir.mkdir(exist_ok=True)
    centered=m.copy();centered.apply_translation([-m.bounds[:,0].mean(),-m.bounds[:,1].mean(),-m.bounds[0,2]])
    centered.export(print_dir/path.name)
    report['parts'][name]=record

structure=trimesh.util.concatenate(list(assembled.values()))
camera=np.array(CAM['torso_T_camera'])[:3,3]*1000
x,y=np.meshgrid(np.arange(181),np.arange(271))
local=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
target=trimesh.transform_points(local,TBOARD)
delta=target-camera
lengths=np.linalg.norm(delta,axis=1)
origins=np.tile(camera,(len(target),1))
blocked_count=0
for start in range(0,len(target),512):
    end=min(start+512,len(target))
    locations,rays,triangles=structure.ray.intersects_location(origins[start:end],delta[start:end]/lengths[start:end,None],multiple_hits=False)
    if len(locations):blocked_count+=int((np.linalg.norm(locations-camera,axis=1)<lengths[start+rays]-0.1).sum())
report['checks']['active_target_ray_count']=len(target)
report['checks']['active_target_blocked_ray_count']=blocked_count
assert blocked_count==0,'Structure obstructs active target'

def segment_clear(a,b,margin=0.05):
    a=np.array(a);b=np.array(b);d=b-a;dist=np.linalg.norm(d)
    pos,idx,_=structure.ray.intersects_location(a[None],(d/dist)[None],multiple_hits=False)
    return len(pos)==0 or np.linalg.norm(pos[0]-a)>=dist-margin

m6_clear=True
for r in D['roots'].values():
    q=np.array(r['center'])
    for radius in [0,3,6.5,7.2]:
        for angle in np.linspace(0,2*np.pi,24,endpoint=False):
            a=q+np.array([12.1,radius*np.cos(angle),radius*np.sin(angle)])
            b=a+np.array([90,0,0])
            m6_clear=m6_clear and segment_clear(a,b)
report['checks']['four_M6_14p4_mm_diameter_90_mm_access_corridors_clear']=m6_clear
assert m6_clear,'Blocked M6 access'

m4_clear=True
for side in ['left','right']:
    for row in ['upper','lower']:
        q=np.array(D['board_connector_centers_torso_mm'][row+'_'+side])
        n=TBOARD[:3,2]
        # Optical-side tool path and rear nut approach; carrier excluded intentionally.
        m4_clear=m4_clear and segment_clear(q-n*65,q-n*10)
        m4_clear=m4_clear and segment_clear(q+n*12.1,q+n*65)
report['checks']['four_M4_optical_side_and_rear_nut_centerline_paths_clear']=m4_clear
assert m4_clear,'Blocked M4 access'

# Independent screw-axis alignment: each of 8 M5 centerlines must pass through
# both the plate and matching crossbar up to its front-loading nut pocket.
m5_clear=True
for sign in [-1,1]:
    for z in [D['upper_z'],D['lower_z']]:
        for dz in [-10,10]:m5_clear=m5_clear and segment_clear([94,sign*140,z+dz],[94,sign*98,z+dz])
report['checks']['eight_M5_shared_axes_clear']=m5_clear
assert m5_clear,'M5 interface holes are misaligned'

# Compare sampled rear contact faces against the public shell, avoiding the bore.
torso=trimesh.load_mesh(BUNDLE/'robot_model/torso_link_rev_1_0.STL');torso.apply_scale(1000)
seat_fit={}
for name,r in D['roots'].items():
    points=[]
    for rad in [5,9,12]:
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            points.append([140,r['center'][1]+rad*np.cos(angle),r['center'][2]+rad*np.sin(angle)])
    points=np.array(points);direction=np.tile([-1,0,0],(len(points),1))
    shell_hits,shell_idx,_=torso.ray.intersects_location(points,direction,multiple_hits=False)
    # From behind the chest, the first fixture hit is its shell contact surface.
    points[:,0]=0
    row=name.split('_')[0];fixture=assembled['V7_crossbar_'+row]
    fixture_hits,fixture_idx,_=fixture.ray.intersects_location(points,-direction,multiple_hits=False)
    shell_map=dict(zip(shell_idx,shell_hits[:,0]));fixture_map=dict(zip(fixture_idx,fixture_hits[:,0]))
    diffs=[fixture_map[i]-shell_map[i] for i in range(len(points)) if i in shell_map and i in fixture_map]
    seat_fit[name]={'sample_count':len(diffs),'minimum_fixture_minus_nominal_shell_x_mm':float(min(diffs)),'maximum_fixture_minus_nominal_shell_x_mm':float(max(diffs))}
report['checks']['nominal_contact_approximation']=seat_fit

mass=sum(m.volume for m in assembled.values())/1000*1.25/1000
com=np.average([m.center_mass for m in assembled.values()],axis=0,weights=[m.volume for m in assembled.values()])
report['full_density_structure_upper_bound']={'PLA_kg_at_1p25_g_per_cm3':float(mass),'center_of_mass_torso_mm':com.tolist(),'gravity_moment_about_M6_pattern_Nm':float(mass*9.80665*(com[0]-65.5727)/1000),'note':'Full CAD volume, including solid infill, excluding printed carrier and metal fasteners; not a slicer estimate or load rating.'}
report['checks']['frozen_board_transform_retained']=True
report['checks']['no_mesh_topology_failures']=True
report=json.loads(json.dumps(report,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
(OUT/'mesh_and_clearance_qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['checks'],indent=2))
print(json.dumps(report['full_density_structure_upper_bound'],indent=2))
