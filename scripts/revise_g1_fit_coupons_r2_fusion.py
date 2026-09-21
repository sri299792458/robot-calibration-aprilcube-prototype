"""Trim existing native coupons to the full 28 mm seat and washer plane."""
from pathlib import Path
import json
import adsk.core, adsk.fusion
BASE=Path(__file__).resolve().parents[1]

def run(_context: str):
    path=BASE/'scripts/build_g1_mount_v7_fusion.py'
    ns={'__file__':str(path),'__name__':'g1_helpers'}
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),ns)
    app=adsk.core.Application.get();design=adsk.fusion.Design.cast(app.activeProduct)
    if app.activeDocument.name!='G1 V7 - broad crossbars and flat side plates':raise RuntimeError('Wrong document')
    out=BASE/'artifacts/g1_coupon_r2'
    (out/'precision_reference').mkdir(exist_ok=True)
    def precise_export(body,path):
        options=design.exportManager.createSTLExportOptions(body,str(path))
        options.isBinaryFormat=True
        options.surfaceDeviation=0.0002
        options.maximumEdgeLength=0.25
        assert design.exportManager.execute(options)
    for occ in design.rootComponent.occurrences:
        if occ.component.name.startswith('V7_crossbar_'):
            precise_export(occ.component.bRepBodies.item(0),out/'precision_reference'/(occ.component.name+'.stl'))
    reports={}
    for occ in design.rootComponent.occurrences:
        comp=occ.component;name=comp.name
        if not name.startswith('FIT_ONLY_'):continue
        key=name.removeprefix('FIT_ONLY_');r=ns['D']['roots'][key]
        cut=r['washer_depth'];u=r['center'][1]
        minimum_shell=min(q[1] for section in r['sections'] for q in section['shell_v_depth'])
        assert minimum_shell-cut>4.5,(key,minimum_shell-cut)
        if not any(comp.features.item(i).name=='R2 compact coupon: full seat to washer plane' for i in range(comp.features.count)):
            plane=ns['offset'](comp,'z',cut,'R2 unchanged M6 washer bearing plane')
            sk=ns['sketch'](comp,plane,'R2 full 28 x 28 mm shell-contact footprint')
            ns['polygon'](sk,[[u-14,-14,cut],[u+14,-14,cut],[u+14,14,cut],[u-14,14,cut]])
            ns['extrude'](comp,sk,65,adsk.fusion.FeatureOperations.IntersectFeatureOperation,'R2 compact coupon: full seat to washer plane')
        if not any(comp.features.item(i).name=='R2 UP arrow and identification dots' for i in range(comp.features.count)):
            plane=ns['offset'](comp,'z',cut,'R2 front markings plane')
            sk=ns['sketch'](comp,plane,'UP arrow and 1 UL / 2 UR / 3 LL / 4 LR dots')
            ns['polygon'](sk,[[u,-12,cut],[u-2,-8,cut],[u+2,-8,cut]])
            dot_count={'upper_left':1,'upper_right':2,'lower_left':3,'lower_right':4}[key]
            for i in range(dot_count):ns['circle'](sk,[u+(i-(dot_count-1)/2)*3,11,cut],1.5)
            ns['extrude'](comp,sk,0.6,adsk.fusion.FeatureOperations.CutFeatureOperation,'R2 UP arrow and identification dots',all_profiles=True)
        rows=json.loads((ns['OUT']/('V7_crossbar_'+key.split('_')[0]+'_fusion.json')).read_text(encoding='utf-8'))['local_to_torso_mm']
        ns['export_part'](design,comp,name,rows)
        precise_export(comp.bRepBodies.item(0),ns['OUT']/'cad'/(name+'_PRINT.stl'))
        reports[key]={'front_cut_mm':cut,'minimum_sampled_depth_mm':minimum_shell-cut,'full_contact_footprint_mm':[28,28],
                      'nominal_center_screw_stack_mm':12,'volume_cm3':comp.bRepBodies.item(0).volume,
                      'identity_dots':{'upper_left':1,'upper_right':2,'lower_left':3,'lower_right':4}[key]}
        occ.isLightBulbOn=False
    assert len(reports)==4
    design.activateRootComponent()
    errors=[]
    for i in range(design.timeline.count):
        entity=design.timeline.item(i).entity
        if hasattr(entity,'healthState') and entity.healthState!=adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
            errors.append({'name':entity.name,'message':entity.errorOrWarningMessage})
    assert not errors,errors
    (out/'native_revision.json').write_text(json.dumps({'coupons':reports,'feature_health_issues':errors},indent=2))
    print(json.dumps(reports))
