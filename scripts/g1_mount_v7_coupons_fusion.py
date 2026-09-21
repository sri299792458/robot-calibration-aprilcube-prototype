"""Extract exact interface fit coupons from the new native Fusion crossbars."""
from pathlib import Path
import json
import adsk.core
import adsk.fusion

BASE=Path(__file__).resolve().parents[1]

def run(_context: str):
    path=BASE/'scripts/build_g1_mount_v7_fusion.py'
    ns={'__file__':str(path),'__name__':'g1_v7_helpers'}
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),ns)
    app=adsk.core.Application.get();design=adsk.fusion.Design.cast(app.activeProduct);root=design.rootComponent
    if app.activeDocument.name!='G1 V7 - broad crossbars and flat side plates':raise RuntimeError('Wrong document')
    for row in ['upper','lower']:
        source=next(root.occurrences.item(i).component for i in range(root.occurrences.count) if root.occurrences.item(i).component.name=='V7_crossbar_'+row)
        for side in ['left','right']:
            name='FIT_ONLY_'+row+'_'+side
            rows=json.loads((ns['OUT']/('V7_crossbar_'+row+'_fusion.json')).read_text(encoding='utf-8'))['local_to_torso_mm']
            occ=root.occurrences.addNewComponent(ns['matrix'](rows));comp=occ.component;comp.name=name
            occ.isGroundToParent=False
            body=source.bRepBodies.item(0).copyToComponent(occ)
            if comp.bRepBodies.count!=1:raise RuntimeError('Body copy did not enter coupon component')
            body=comp.bRepBodies.item(0)
            print(name,'copied local bounds',body.boundingBox.minPoint.asArray(),body.boundingBox.maxPoint.asArray())
            # A native-object source must keep the crossbar print coordinates.
            if abs(body.boundingBox.minPoint.z)>1e-6:raise RuntimeError('Unexpected copied body coordinate frame')
            u=ns['D']['roots'][row+'_'+side]['center'][1]
            cut=ns['D']['roots'][row+'_'+side]['washer_depth']
            plane=ns['offset'](comp,'z',cut,'R2 unchanged M6 washer bearing plane')
            sk=ns['sketch'](comp,plane,'R2 full 28 x 28 mm shell-contact footprint')
            ns['polygon'](sk,[[u-14,-14,cut],[u+14,-14,cut],[u+14,14,cut],[u-14,14,cut]])
            ns['extrude'](comp,sk,65,adsk.fusion.FeatureOperations.IntersectFeatureOperation,'R2 compact coupon: full seat to washer plane')
            sk=ns['sketch'](comp,plane,'UP arrow and identification dots')
            ns['polygon'](sk,[[u,-12,cut],[u-2,-8,cut],[u+2,-8,cut]])
            dots={'upper_left':1,'upper_right':2,'lower_left':3,'lower_right':4}[row+'_'+side]
            for i in range(dots):ns['circle'](sk,[u+(i-(dots-1)/2)*3,11,cut],1.5)
            ns['extrude'](comp,sk,0.6,adsk.fusion.FeatureOperations.CutFeatureOperation,'R2 UP arrow and identification dots',all_profiles=True)
            body=comp.bRepBodies.item(0);body.name=name
            ns['export_part'](design,comp,name,rows)
            occ.isLightBulbOn=False
    design.activateRootComponent()
