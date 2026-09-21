"""Validate compact coupons against their original exported interfaces."""
from pathlib import Path
import json,sys,hashlib
BASE=Path(__file__).resolve().parents[1];OLD=BASE/'artifacts/g1_mount_v7';OUT=BASE/'artifacts/g1_coupon_r2'
sys.path.insert(0,str(OLD/'python_packages'))
import numpy as np
import trimesh
D=json.loads((OLD/'design_inputs.json').read_text(encoding='utf-8'))
report={}; main_qa=json.loads((OLD/'mesh_and_clearance_qa.json').read_text(encoding='utf-8'))
def first_z(mesh,origins,direction):
    hits={}
    for start in range(0,len(origins),256):
        rays=origins[start:start+256]
        p,idx,_=mesh.ray.intersects_location(rays,np.tile(direction,(len(rays),1)),multiple_hits=False)
        hits.update({int(i)+start:float(q[2]) for i,q in zip(idx,p)})
    return hits
for key,r in D['roots'].items():
    name='FIT_ONLY_'+key
    path=OLD/'cad'/(name+'_PRINT.stl')
    old=trimesh.load_mesh(OUT/'original_reference'/path.name)
    reference=trimesh.load_mesh(OUT/'precision_reference'/('V7_crossbar_'+key.split('_')[0]+'.stl'))
    new=trimesh.load_mesh(path)
    native=json.loads((OLD/(name+'_fusion.json')).read_text(encoding='utf-8'))
    u=r['center'][1];cut=r['washer_depth']
    assert new.is_watertight and new.is_winding_consistent and len(new.split())==1,key
    assert np.allclose(new.extents[:2],[28,28],atol=.001),key
    assert abs(new.volume/1000-native['volume_cm3'])/native['volume_cm3']<.001,key
    assert abs(new.bounds[0,2]-cut)<.001,key
    # Same entire seating patch, including locations near its four edges.
    grid=np.linspace(-13.95,13.95,41)
    samples=np.array([[u+x,y,90] for x in grid for y in grid if x*x+y*y>3.6**2])
    a=first_z(reference,samples,[0,0,-1]);b=first_z(new,samples,[0,0,-1])
    assert len(a)==len(samples) and len(b)==len(samples),key
    contact_error=max(abs(a[i]-b[i]) for i in range(len(samples)))
    # Independent STL tessellations can differ on the trimmed loft surface.
    # Limit the sampled agreement to 0.05 mm, below the 0.20 mm print layer.
    assert contact_error<.05,(key,contact_error)
    washer=np.array([[u+radius*np.cos(t),radius*np.sin(t),0] for radius in [4,5.5,6.5] for t in np.linspace(0,2*np.pi,32,endpoint=False)])
    old_seat=first_z(reference,washer,[0,0,1]);new_seat=first_z(new,washer,[0,0,1])
    assert len(old_seat)==len(washer) and len(new_seat)==len(washer),key
    washer_error=max(abs(old_seat[i]-new_seat[i]) for i in range(len(washer)))
    assert washer_error<.002,(key,washer_error)
    assert not first_z(new,np.array([[u,0,0]]),[0,0,1]),key
    # Keep the old export coordinates for CAD; provide bed-centered printable copies.
    centered=new.copy();centered.apply_translation([-u,0,-new.bounds[0,2]])
    assert abs(centered.bounds[0,2])<.001
    centered.export(OUT/'print_parts'/('FIT_R2_'+key+'_PRINT.stl'))
    centered.export(OLD/'fit_coupons'/path.name)
    report[key]={'original_volume_cm3':old.volume/1000,'revised_volume_cm3':new.volume/1000,
                 'volume_reduction_percent':100*(1-new.volume/old.volume),'print_extents_mm':new.extents.tolist(),
                 'contact_samples':len(samples),'max_contact_mesh_difference_mm':contact_error,
                 'contact_mesh_comparison_tolerance_mm':0.05,
                 'reference':'Original native crossbar exported at 0.002 mm surface deviation and 2.5 mm maximum edge length',
                 'washer_samples':len(washer),'max_washer_difference_mm':washer_error,'screw_axis_clear':True,
                 'watertight':True,'single_solid':True,'original_nominal_center_stack_mm':12}
    main_qa['parts'][name]={'watertight':True,'single_body':True,'extents_mm':new.extents.tolist(),'volume_cm3':new.volume/1000,
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'revision':'R2 compact'}
for name,part in main_qa['parts'].items():
    if name.startswith('V7_'):
        assert hashlib.sha256((OLD/'cad'/(name+'_PRINT.stl')).read_bytes()).hexdigest()==part['sha256'],name
main_qa['checks']['compact_coupon_contact_and_washer_faces_preserved']=True
(OLD/'mesh_and_clearance_qa.json').write_text(json.dumps(main_qa,indent=2))
(OUT/'geometry_qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
