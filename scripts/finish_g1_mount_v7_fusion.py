"""Add frozen reference geometry, restore assembly placement, export local CAD."""
from pathlib import Path
import json
import adsk.core
import adsk.fusion

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'artifacts/g1_mount_v7'
BUNDLE=BASE/'artifacts/g1_mount_v7/reference_inputs'

def run(_context: str):
    ns={'__file__':str(BASE/'scripts/build_g1_mount_v7_fusion.py'),'__name__':'g1_v7_helpers'}
    exec(compile((BASE/'scripts/build_g1_mount_v7_fusion.py').read_text(encoding='utf-8'),ns['__file__'],'exec'),ns)
    app=adsk.core.Application.get();design=adsk.fusion.Design.cast(app.activeProduct)
    if app.activeDocument.name!='G1 V7 - broad crossbars and flat side plates':raise RuntimeError('Wrong active document')
    root=design.rootComponent
    design.activateRootComponent()
    # Imported mesh references retain the supplied geometry and its native units.
    existing=[root.occurrences.item(i).component.name for i in range(root.occurrences.count)]
    for name,files,units in [
        ('REFERENCE - existing printed carrier', ['current_design/cad/carrier_white_body.stl','current_design/cad/carrier_black_inlay.stl'],adsk.fusion.MeshUnits.MillimeterMeshUnit),
        ('REFERENCE - nominal torso', ['robot_model/torso_link_rev_1_0.STL'],adsk.fusion.MeshUnits.MeterMeshUnit),
        ('REFERENCE - nominal head', ['robot_model/head_link.STL'],adsk.fusion.MeshUnits.MeterMeshUnit)]:
        if name in existing:continue
        occ=root.occurrences.addNewComponent(adsk.core.Matrix3D.create());comp=occ.component;comp.name=name
        base=comp.features.baseFeatures.add();base.name='Supplied reference meshes';base.startEdit()
        for file in files:
            bodies=comp.meshBodies.add(str(BUNDLE/file),units,base)
            for i in range(bodies.count):bodies.item(i).name=Path(file).stem
        base.finishEdit()
        comp.isOriginFolderLightBulbOn=False
    data=json.loads((OUT/'design_inputs.json').read_text(encoding='utf-8'))
    for i in range(root.occurrences.count):
        occ=root.occurrences.item(i);name=occ.component.name
        if name.startswith('V7_'):
            occ.isGroundToParent=False
            record=json.loads((OUT/(name+'_fusion.json')).read_text(encoding='utf-8'))
            occ.transform2=ns['matrix'](record['local_to_torso_mm'])
        elif name=='REFERENCE - existing printed carrier':
            # Supplied STLs were translated +15 mm in both in-plane axes
            # for slicing. Undo that before applying the optical-board frame.
            board=data['torso_T_board_mm']
            rows=[r[:3]+[r[3]-15*r[0]-15*r[1]] for r in board[:3]]
            occ.transform2=ns['matrix'](rows)
        elif name=='REFERENCE - nominal head':
            # The G1 head mesh uses the fixed head_joint origin from the included URDF.
            import xml.etree.ElementTree as ET
            urdf=ET.parse(BUNDLE/'robot_model/g1_29dof_rev_1_0.urdf').getroot()
            joint=next(j for j in urdf.findall('joint') if j.find('child').get('link')=='head_link')
            xyz=[float(v)*1000 for v in joint.find('origin').get('xyz','0 0 0').split()]
            rpy=[float(v) for v in joint.find('origin').get('rpy','0 0 0').split()]
            if any(abs(v)>1e-9 for v in rpy):raise RuntimeError('Handle head reference rotation explicitly')
            occ.transform2=ns['matrix']([[1,0,0,xyz[0]],[0,1,0,xyz[1]],[0,0,1,xyz[2]]])
        if name in ['REFERENCE - nominal torso','REFERENCE - nominal head']:occ.isLightBulbOn=True
        occ.component.isOriginFolderLightBulbOn=False
    if design.snapshots.hasPendingSnapshot:design.snapshots.add()
    design.activateRootComponent()
    root.isOriginFolderLightBulbOn=False
    cam=app.activeViewport.camera
    cam.isSmoothTransition=False;cam.eye=adsk.core.Point3D.create(68,-78,52)
    cam.target=adsk.core.Point3D.create(17,0,23)
    cam.upVector=adsk.core.Vector3D.create(0,0,1)
    cam.cameraType=adsk.core.CameraTypes.OrthographicCameraType;cam.isFitView=True
    app.activeViewport.camera=cam
    app.activeViewport.visualStyle=adsk.core.VisualStyles.ShadedWithVisibleEdgesOnlyVisualStyle
    app.activeViewport.refresh()
    (OUT/'renders').mkdir(exist_ok=True)
    app.activeViewport.saveAsImageFile(str(OUT/'renders/fusion_assembly.png'),1700,1200)
    errors=[]
    for i in range(design.timeline.count):
        item=design.timeline.item(i)
        entity=item.entity
        if hasattr(entity,'healthState') and entity.healthState!=adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
            errors.append({'name':entity.name,'state':int(entity.healthState),'message':entity.errorOrWarningMessage})
    report={'timeline_entries':design.timeline.count,'feature_health_issues':errors,
            'components':[root.occurrences.item(i).component.name for i in range(root.occurrences.count)]}
    (OUT/'fusion_assembly_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
