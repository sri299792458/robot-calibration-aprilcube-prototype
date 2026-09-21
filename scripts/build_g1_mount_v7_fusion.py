"""Native Fusion V7 fixture, executed through Fusion MCP. Dimensions are mm.

Run build('side_left'), build('side_right'), build('crossbar_upper'),
build('crossbar_lower') independently. No cloud save or existing-document edits.
"""
import json
import math
from pathlib import Path
import adsk.core
import adsk.fusion

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'artifacts/g1_mount_v7'
D = json.loads((OUT/'design_inputs.json').read_text(encoding='utf-8'))
F = adsk.fusion.FeatureOperations

def p(a):
    return adsk.core.Point3D.create(*[v/10 for v in a])

def val(v):
    return adsk.core.ValueInput.createByString(v if isinstance(v,str) else f'{v:.10f} mm')

def coll(items):
    c=adsk.core.ObjectCollection.create()
    for item in items: c.add(item)
    return c

def offset(comp,axis,distance,name):
    planes={'x':comp.yZConstructionPlane,'y':comp.xZConstructionPlane,'z':comp.xYConstructionPlane}
    plane=planes[axis]
    normal=plane.geometry.normal
    direction={'x':normal.x,'y':normal.y,'z':normal.z}[axis]
    inp=comp.constructionPlanes.createInput()
    inp.setByOffset(plane,val(distance*direction))
    result=comp.constructionPlanes.add(inp)
    result.name=name
    result.isLightBulbOn=False
    return result

def polygon(sk,points,radius=0):
    pts=[sk.modelToSketchSpace(p(a)) for a in points]
    xy=[(a.x,a.y) for a in pts]
    if radius==0:
        for i in range(len(pts)):
            sk.sketchCurves.sketchLines.addByTwoPoints(pts[i],pts[(i+1)%len(pts)])
        return
    ends=[]
    for i,b in enumerate(xy):
        a=xy[i-1]; c=xy[(i+1)%len(xy)]
        la=math.dist(a,b); lc=math.dist(c,b)
        ua=((a[0]-b[0])/la,(a[1]-b[1])/la)
        uc=((c[0]-b[0])/lc,(c[1]-b[1])/lc)
        angle=math.acos(max(-1,min(1,ua[0]*uc[0]+ua[1]*uc[1])))
        trim=min(radius/10/math.tan(angle/2),la*0.35,lc*0.35)
        r=trim*math.tan(angle/2)
        start=(b[0]+ua[0]*trim,b[1]+ua[1]*trim)
        end=(b[0]+uc[0]*trim,b[1]+uc[1]*trim)
        bis=(ua[0]+uc[0],ua[1]+uc[1]); bl=math.hypot(*bis)
        unit=(bis[0]/bl,bis[1]/bl)
        center=(b[0]+unit[0]*r/math.sin(angle/2),b[1]+unit[1]*r/math.sin(angle/2))
        middle=(center[0]-unit[0]*r,center[1]-unit[1]*r)
        ends.append((start,middle,end))
    def sp(q): return adsk.core.Point3D.create(q[0],q[1],0)
    for i,(start,middle,end) in enumerate(ends):
        sk.sketchCurves.sketchArcs.addByThreePoints(sp(start),sp(middle),sp(end))
        sk.sketchCurves.sketchLines.addByTwoPoints(sp(end),sp(ends[(i+1)%len(ends)][0]))

def largest(sk):
    return max([sk.profiles.item(i) for i in range(sk.profiles.count)],key=lambda q:q.areaProperties().area)

def extrude(comp,sk,depth,operation,name,symmetric=False,all_profiles=False):
    profiles=coll([sk.profiles.item(i) for i in range(sk.profiles.count)]) if all_profiles else largest(sk)
    inp=comp.features.extrudeFeatures.createInput(profiles,operation)
    if operation in (F.CutFeatureOperation,F.IntersectFeatureOperation):
        inp.participantBodies=[comp.bRepBodies.item(i) for i in range(comp.bRepBodies.count)]
    if symmetric: inp.setSymmetricExtent(val(depth),True)
    else: inp.setDistanceExtent(False,val(depth))
    feat=comp.features.extrudeFeatures.add(inp)
    feat.name=name
    sk.isVisible=False
    return feat

def sketch(comp,plane,name):
    sk=comp.sketches.add(plane)
    sk.name=name
    return sk

def circle(sk,center,diameter):
    sk.sketchCurves.sketchCircles.addByCenterRadius(sk.modelToSketchSpace(p(center)),diameter/20)

def matrix(rows):
    m=adsk.core.Matrix3D.create()
    for i in range(3):
        for j in range(4):m.setCell(i,j,rows[i][j]/10 if j==3 else rows[i][j])
    return m

def component(name):
    app=adsk.core.Application.get()
    if app.activeDocument.name != 'G1 V7 - broad crossbars and flat side plates':
        raise RuntimeError('Activate the G1 V7 document first; existing documents are not modified.')
    design=adsk.fusion.Design.cast(app.activeProduct)
    root=design.rootComponent
    if any(root.occurrences.item(i).component.name==name for i in range(root.occurrences.count)):
        raise RuntimeError(f'{name} already exists; do not silently duplicate it.')
    occ=root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.isGroundToParent=False
    comp=occ.component
    comp.name=name
    return app,design,occ,comp

def custom_plane(comp,center,normal,name):
    sk=sketch(comp,comp.xYConstructionPlane,name+' defining points')
    a=center
    b=[center[0],center[1],center[2]+10]
    c=[center[0]-normal[1]*10,center[1]+normal[0]*10,center[2]]
    points=[sk.sketchPoints.add(p(q)) for q in [a,b,c]]
    inp=comp.constructionPlanes.createInput()
    inp.setByThreePoints(*points)
    plane=comp.constructionPlanes.add(inp)
    plane.name=name
    sk.isVisible=False;plane.isLightBulbOn=False
    return plane

def color(app,design,body,color_name):
    library=app.materialLibraries.itemByName('Fusion Appearance Library')
    source=library.appearances.itemByName('Plastic - Matte ('+color_name+')')
    target=design.appearances.itemByName('G1 V7 '+color_name)
    if target is None: target=design.appearances.addByCopy(source,'G1 V7 '+color_name)
    body.appearance=target

def export_part(design,comp,name,transform):
    if comp.bRepBodies.count!=1:raise RuntimeError(f'{name} has {comp.bRepBodies.count} bodies')
    body=comp.bRepBodies.item(0)
    if not body.isSolid:raise RuntimeError(f'{name} is not a solid')
    box=body.boundingBox
    dims=[(b-a)*10 for a,b in zip(box.minPoint.asArray(),box.maxPoint.asArray())]
    folder=OUT/'cad';folder.mkdir(exist_ok=True)
    options=design.exportManager.createSTLExportOptions(body,str(folder/(name+'_PRINT.stl')))
    options.meshRefinement=adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
    if name.startswith('FIT_ONLY_'):
        options.surfaceDeviation=0.0002
        options.maximumEdgeLength=0.25
    options.isBinaryFormat=True
    assert design.exportManager.execute(options)
    report={'name':name,'body_count':comp.bRepBodies.count,'is_solid':body.isSolid,'volume_cm3':body.volume,
            'extents_mm':dims,'print_bounds_mm':[[x*10 for x in box.minPoint.asArray()],[x*10 for x in box.maxPoint.asArray()]],
            'local_to_torso_mm':transform,'feature_count':comp.features.count,'stl':name+'_PRINT.stl'}
    (OUT/(name+'_fusion.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

def build_side(side):
    name='V7_side_'+side
    app,design,occ,comp=component(name)
    sign=1 if side=='left' else -1
    def points(poly,z=0):return [[sign*q[0],q[1],z] for q in poly]
    if not design.userParameters.itemByName('v7PlateThickness'):
        design.userParameters.add('v7PlateThickness',val('10 mm'),'mm','Fixed mating-interface thickness. A change requires updating component placement, carrier tabs and screw stacks, then repeating clearance checks.')
    sk=sketch(comp,comp.xYConstructionPlane,'Broad perimeter and continuous diagonal')
    polygon(sk,points(D['side_outer_xz']),3)
    feat=extrude(comp,sk,'v7PlateThickness',F.NewBodyFeatureOperation,'10 mm flat side plate')
    body=feat.bodies.item(0);body.name=name
    sk=sketch(comp,comp.xYConstructionPlane,'Two rounded lightening windows; 18 mm rails')
    for window in D['side_windows_xz']:polygon(sk,points(window),6)
    extrude(comp,sk,12,F.CutFeatureOperation,'Rounded windows and 18 mm diagonal',all_profiles=True)
    sk=sketch(comp,comp.xYConstructionPlane,'Full backing beneath both carrier tabs')
    for poly in D['board_tabs_xz']:polygon(sk,points(poly),1.5)
    extrude(comp,sk,'v7PlateThickness',F.JoinFeatureOperation,'Continuous tab backing to print bed',all_profiles=True)
    for i,poly in enumerate(D['board_tabs_xz']):
        plane=offset(comp,'z',8,f'Carrier tab {i+1} overlap plane')
        sk=sketch(comp,plane,f'Carrier tab {i+1} broad footprint')
        polygon(sk,points(poly,8),1.5)
        extrude(comp,sk,26.5,F.JoinFeatureOperation,f'Carrier tab {i+1} continuous 28 x 12 mm root')
        key=('upper_' if i==0 else 'lower_')+side
        q=D['board_connector_centers_torso_mm'][key]
        center=[sign*q[0],q[2],26.5]
        normal=[sign*math.cos(math.radians(21)),-math.sin(math.radians(21)),0]
        plane=custom_plane(comp,center,normal,f'M4 board-normal axis {i+1}')
        sk=sketch(comp,plane,f'M4 carrier hole {i+1}, existing board coordinate')
        circle(sk,center,4.5)
        extrude(comp,sk,50,F.CutFeatureOperation,f'M4 through hole {i+1}',symmetric=True)
    sk=sketch(comp,comp.xYConstructionPlane,'Paired M5 bolts at both crossbars')
    for z in [D['upper_z'],D['lower_z']]:
        for dz in [-10,10]:circle(sk,[sign*94,z+dz,0],5.4)
    extrude(comp,sk,12,F.CutFeatureOperation,'Four M5 through holes, 20 mm pair spacing',all_profiles=True)
    color(app,design,body,'Green')
    rows=[[sign,0,0,0],[0,0,-sign,sign*125],[0,1,0,0]]
    export_part(design,comp,name,rows)
    occ.transform2=matrix(rows)
    if design.snapshots.hasPendingSnapshot:design.snapshots.add()
    app.activeViewport.fit();app.activeViewport.refresh()

def build_crossbar(row):
    name='V7_crossbar_'+row
    app,design,occ,comp=component(name)
    zrow=D[row+'_z']
    sk=sketch(comp,comp.xYConstructionPlane,'230 x 40 mm full-width torso crossbar')
    polygon(sk,[[-115,-20,0],[115,-20,0],[115,20,0],[-115,20,0]],4)
    feat=extrude(comp,sk,20,F.NewBodyFeatureOperation,'20 mm continuous crossbar')
    body=feat.bodies.item(0);body.name=name
    for side in ['left','right']:
        root=D['roots'][row+'_'+side]
        loft=comp.features.loftFeatures.createInput(F.JoinFeatureOperation)
        for j,section in enumerate(root['sections']):
            u=section['u']
            plane=offset(comp,'x',u,f'{side} contact section {j+1}')
            sk=sketch(comp,plane,f'{side} nominal shell section {j+1}')
            shell=section['shell_v_depth']
            coords=[[u,-14,18],[u,14,18]]+[[u,v,d] for v,d in reversed(shell)]
            polygon(sk,coords)
            loft.loftSections.add(largest(sk))
            sk.isVisible=False
        loft.isSolid=True
        feature=comp.features.loftFeatures.add(loft)
        feature.name=f'{side} 28 x 28 mm shell seat integrated into crossbar'
        sk=sketch(comp,comp.xYConstructionPlane,f'{side} straight M6 washer and Allen access')
        circle(sk,[root['center'][1],0,0],16)
        extrude(comp,sk,root['washer_depth'],F.CutFeatureOperation,f'{side} M6 well; 12 mm center clamp stack')
        sk=sketch(comp,comp.xYConstructionPlane,f'{side} M6 through bore')
        circle(sk,[root['center'][1],0,0],6.8)
        extrude(comp,sk,65,F.CutFeatureOperation,f'{side} M6 6.8 mm clearance')
    for sign in [-1,1]:
        plane=offset(comp,'x',sign*115,'Side plate bolt entry plane')
        sk=sketch(comp,plane,'Two transverse M5 bolt holes')
        for v in [-10,10]:circle(sk,[sign*115,v,10],5.4)
        extrude(comp,sk,40,F.CutFeatureOperation,'M5 side connection bores',symmetric=True,all_profiles=True)
        sk=sketch(comp,comp.xYConstructionPlane,'Front-loading M5 hex-nut slots')
        for v in [-10,10]:
            center=sign*102
            polygon(sk,[[center-2.2,v-4.15,0],[center+2.2,v-4.15,0],[center+2.2,v+4.15,0],[center-2.2,v+4.15,0]])
        extrude(comp,sk,15.2,F.CutFeatureOperation,'M5 nut pockets, 4.4 mm thick / 8.3 mm flats',all_profiles=True)
    color(app,design,body,'Gray')
    rows=[[0,0,-1,104],[1,0,0,0],[0,-1,0,zrow]]
    export_part(design,comp,name,rows)
    occ.transform2=matrix(rows)
    if design.snapshots.hasPendingSnapshot:design.snapshots.add()
    app.activeViewport.fit();app.activeViewport.refresh()

def build(part):
    if part.startswith('side_'):build_side(part[5:])
    elif part.startswith('crossbar_'):build_crossbar(part[9:])
    else:raise ValueError(part)

def reinforce_existing_left():
    app=adsk.core.Application.get()
    design=adsk.fusion.Design.cast(app.activeProduct)
    root=design.rootComponent
    comp=next(root.occurrences.item(i).component for i in range(root.occurrences.count) if root.occurrences.item(i).component.name=='V7_side_left')
    sk=sketch(comp,comp.xYConstructionPlane,'Full backing beneath both carrier tabs')
    for poly in D['board_tabs_xz']:polygon(sk,[[q[0],q[1],0] for q in poly],1.5)
    extrude(comp,sk,'v7PlateThickness',F.JoinFeatureOperation,'Continuous tab backing to print bed',all_profiles=True)
    export_part(design,comp,'V7_side_left',[[1,0,0,0],[0,0,-1,125],[0,1,0,0]])

def run(_context: str):
    build('side_left')
