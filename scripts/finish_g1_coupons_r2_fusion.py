"""Export compact coupon CAD and inspect both sides in the native viewport."""
from pathlib import Path
import json
import adsk.core, adsk.fusion
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'artifacts/g1_coupon_r2';MAIN=BASE/'artifacts/g1_mount_v7'

def run(_context: str):
    app=adsk.core.Application.get();d=adsk.fusion.Design.cast(app.activeProduct)
    if app.activeDocument.name!='G1 V7 - broad crossbars and flat side plates':raise RuntimeError('Wrong document')
    D=json.loads((MAIN/'design_inputs.json').read_text(encoding='utf-8'))
    (OUT/'cad').mkdir(exist_ok=True);(OUT/'renders').mkdir(exist_ok=True)
    coupons=[o for o in d.rootComponent.occurrences if o.component.name.startswith('FIT_ONLY_')]
    assert len(coupons)==4
    assert d.exportManager.execute(d.exportManager.createFusionArchiveExportOptions(str(MAIN/'cad/G1_calibration_mount_native.f3d')))
    health={'timeline_entries':d.timeline.count,'feature_health_issues':[],
            'components':[o.component.name for o in d.rootComponent.occurrences]}
    for i in range(d.timeline.count):
        entity=d.timeline.item(i).entity
        if hasattr(entity,'healthState') and entity.healthState!=adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
            health['feature_health_issues'].append({'name':entity.name,'message':entity.errorOrWarningMessage})
    assert not health['feature_health_issues']
    (MAIN/'fusion_assembly_qa.json').write_text(json.dumps(health,indent=2))
    visibility=[(o,o.isLightBulbOn) for o in d.rootComponent.occurrences]
    transforms=[(o,o.transform2) for o in coupons]
    original_camera=app.activeViewport.camera
    positions={'upper_left':[-22,-22],'upper_right':[22,-22],'lower_left':[-22,22],'lower_right':[22,22]}
    for o,_ in visibility:o.isLightBulbOn=o.component.name.startswith('FIT_ONLY_')
    for occ in coupons:
        key=occ.component.name.removeprefix('FIT_ONLY_');r=D['roots'][key];xy=positions[key]
        m=adsk.core.Matrix3D.create();m.translation=adsk.core.Vector3D.create((xy[0]-r['center'][1])/10,xy[1]/10,-r['washer_depth']/10)
        occ.transform2=m
    d.activateRootComponent()
    cam=app.activeViewport.camera;cam.isSmoothTransition=False
    cam.eye=adsk.core.Point3D.create(12,-16,15);cam.target=adsk.core.Point3D.create(0,0,0.7)
    cam.upVector=adsk.core.Vector3D.create(0,0,1);cam.cameraType=adsk.core.CameraTypes.OrthographicCameraType;cam.isFitView=True
    app.activeViewport.camera=cam;app.activeViewport.refresh()
    app.activeViewport.saveAsImageFile(str(OUT/'renders/compact_coupons.png'),1400,1000)
    cam=app.activeViewport.camera;cam.isSmoothTransition=False
    cam.eye=adsk.core.Point3D.create(0,0,-20);cam.target=adsk.core.Point3D.create(0,0,0)
    cam.upVector=adsk.core.Vector3D.create(0,-1,0);cam.isFitView=True
    app.activeViewport.camera=cam;app.activeViewport.refresh()
    app.activeViewport.saveAsImageFile(str(OUT/'renders/identification_marks.png'),1100,1100)
    for occ,m in transforms:occ.transform2=m
    for occ,visible in visibility:occ.isLightBulbOn=visible
    app.activeViewport.camera=original_camera;app.activeViewport.refresh()
    print(json.dumps({'exported_coupons':4,'native_timeline_entries':d.timeline.count,'feature_health_issues':[]}))
