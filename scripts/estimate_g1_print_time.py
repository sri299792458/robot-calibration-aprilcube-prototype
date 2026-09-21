"""Local Bambu Studio estimates; never connects to or starts a printer."""
from pathlib import Path
import json, subprocess, sys, time
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'artifacts/g1_print_time/controlled_comparison'
PROFILES=Path('C:/Program Files/Bambu Studio/resources/profiles/BBL')
OUT.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(BASE/'artifacts/g1_mount_v7/python_packages'))
import trimesh
index={}
for p in PROFILES.rglob('*.json'):
    d=json.loads(p.read_text(encoding='utf-8-sig'))
    if 'name' in d:index[d['name']]=(p,d)
def flatten(name,trail=()):
    assert name not in trail,(name,trail)
    p,d=index[name]
    out=flatten(d['inherits'],trail+(name,)) if d.get('inherits') else {}
    for included in d.get('include',[]):
        template=flatten(included,trail+(name,))
        out.update({k:v for k,v in template.items() if k not in ['name','type','from','instantiation','setting_id']})
    out.update(d)
    out.pop('inherits',None)
    out.pop('include',None)
    return out

machine=flatten('Bambu Lab H2D 0.4 nozzle')
process=flatten('0.20mm Standard @BBL H2D')
filament=flatten('Bambu PLA Basic @BBL H2D')
process.update({'enable_support':'0','enable_prime_tower':'0','brim_type':'outer_only','brim_width':'5'})
standard_coarse=process.copy();standard_coarse.update({'layer_height':'0.28','top_shell_layers':'4','bottom_shell_layers':'3'})
reinforced=process.copy();reinforced.update({'wall_loops':'6','top_shell_layers':'6','bottom_shell_layers':'6',
                'sparse_infill_density':'35%','sparse_infill_pattern':'gyroid'})
reinforced_coarse=reinforced.copy();reinforced_coarse.update({'layer_height':'0.28','top_shell_layers':'5','bottom_shell_layers':'5'})
candidate=reinforced_coarse.copy();candidate.update({'wall_loops':'4','sparse_infill_density':'25%'})
profiles={'standard_020':process,'standard_028':standard_coarse,'reinforced_020':reinforced,'reinforced_028':reinforced_coarse,'candidate_028':candidate}
for name,data in [('machine',machine),('filament',filament)]+[('process_'+k,v) for k,v in profiles.items()]:
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
print(json.dumps({'machine_keys':len(machine),'process_keys':len(process),'filament_keys':len(filament),
                 'max_volumetric_speed':filament.get('filament_max_volumetric_speed'),
                 'nozzle':machine.get('nozzle_diameter')},indent=2),flush=True)

inputs=OUT/'positioned_inputs';inputs.mkdir(exist_ok=True)
for stl in (BASE/'artifacts/g1_mount_v7/print_parts').glob('*.stl'):
    mesh=trimesh.load_mesh(stl)
    mesh.apply_translation([175,160,0])
    assert mesh.bounds[0,0]>30 and mesh.bounds[1,0]<320 and mesh.bounds[0,1]>5 and mesh.bounds[1,1]<315
    mesh.export(inputs/stl.name)
if '--prepare-only' in sys.argv:sys.exit(0)
summary={}
selected_profiles=['standard_020','standard_028'] if '--quick-check' in sys.argv else list(profiles)
selected_parts=['V7_side_left'] if '--quick-check' in sys.argv else ['V7_side_left','V7_side_right','V7_crossbar_upper','V7_crossbar_lower']
for profile in selected_profiles:
    summary[profile]=[]
    for part in selected_parts:
        job=OUT/(profile+'_'+part);job.mkdir(exist_ok=True)
        cmd=['C:/Program Files/Bambu Studio/bambu-studio.exe',
             '--load-settings',str(OUT/'machine.json')+';'+str(OUT/('process_'+profile+'.json')),
             '--load-filaments',str(OUT/'filament.json'),'--curr-bed-type','Textured PEI Plate',
             '--arrange','0','--slice','0','--debug','2','--export-3mf','estimate.3mf','--outputdir',str(job),
             str(inputs/(part+'_PRINT.stl'))]
        (job/'command.json').write_text(json.dumps(cmd,indent=2))
        if not (job/'result.json').exists() or json.loads((job/'result.json').read_text(encoding='utf-8')).get('return_code')!=0:
            with (job/'stdout.txt').open('w') as stdout,(job/'stderr.txt').open('w') as stderr:
                result=subprocess.run(cmd,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW,timeout=240)
            assert result.returncode==0,(job,result.returncode)
        record=json.loads((job/'result.json').read_text(encoding='utf-8'))
        assert record['return_code']==0 and len(record['sliced_plates'])==1,record
        plate=record['sliced_plates'][0]
        item={'part':part,'total_seconds':plate['total_predication'],'model_seconds':plate['main_predication'],
              'grams':sum(f['total_used_g'] for f in plate['filaments']),'bbox':plate.get('objects',[{}])[0].get('bbox'),
              'warnings':plate.get('warning_message',''),'feature_seconds':plate['feature_type_times']}
        summary[profile].append(item)
        (OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'profile':profile,'part':part,'hours':round(item['total_seconds']/3600,2),'grams':round(item['grams'],1),'warnings':item['warnings']}),flush=True)
    print(json.dumps({'profile':profile,'total_hours':round(sum(i['total_seconds'] for i in summary[profile])/3600,2)}),flush=True)
