"""Compare old and compact coupons using identical H2D coupon settings."""
from pathlib import Path
import json,sys,subprocess
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'artifacts/g1_coupon_r2'
sys.path.insert(0,str(BASE/'artifacts/g1_mount_v7/python_packages'))
import trimesh
profiles=BASE/'artifacts/g1_print_time/controlled_comparison'
process=json.loads((profiles/'process_standard_020.json').read_text(encoding='utf-8'))
process.update({'layer_height':'0.20','wall_loops':'3','sparse_infill_density':'10%','sparse_infill_pattern':'gyroid',
                'top_shell_layers':'4','bottom_shell_layers':'4','top_shell_thickness':'0.8','bottom_shell_thickness':'0.8',
                'brim_type':'no_brim','enable_support':'0','enable_prime_tower':'0'})
(OUT/'coupon_process.json').write_text(json.dumps(process,indent=2))
summary={}
for revision in ['original','R2']:
    job=OUT/('slice_'+revision);job.mkdir(exist_ok=True)
    inputs=job/'inputs';inputs.mkdir(exist_ok=True)
    paths=[]
    for key,xy in [('upper_left',[145,135]),('upper_right',[195,135]),('lower_left',[145,185]),('lower_right',[195,185])]:
        source=OUT/('original_reference/FIT_ONLY_'+key+'_PRINT.stl' if revision=='original' else 'print_parts/FIT_R2_'+key+'_PRINT.stl')
        mesh=trimesh.load_mesh(source)
        mesh.apply_translation([xy[0]-mesh.bounds[:,0].mean(),xy[1]-mesh.bounds[:,1].mean(),-mesh.bounds[0,2]])
        p=inputs/('FIT_'+revision+'_'+key+'.stl');mesh.export(p);paths.append(str(p))
    cmd=['C:/Program Files/Bambu Studio/bambu-studio.exe','--load-settings',str(profiles/'machine.json')+';'+str(OUT/'coupon_process.json'),
         '--load-filaments',str(profiles/'filament.json'),'--curr-bed-type','Textured PEI Plate','--arrange','0','--slice','0','--debug','2',
         '--outputdir',str(job),'--export-3mf','G1_'+revision+'_fit_coupons.3mf']+paths
    (job/'command.json').write_text(json.dumps(cmd,indent=2))
    with (job/'stdout.txt').open('w') as stdout,(job/'stderr.txt').open('w') as stderr:
        result=subprocess.run(cmd,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW,timeout=240)
    assert result.returncode==0,(revision,result.returncode)
    record=json.loads((job/'result.json').read_text(encoding='utf-8'))
    assert record['return_code']==0 and len(record['sliced_plates'])==1,record
    p=record['sliced_plates'][0]
    summary[revision]={'total_seconds':p['total_predication'],'model_seconds':p['main_predication'],
                       'filament_g':sum(f['total_used_g'] for f in p['filaments']),'warning':p.get('warning_message','')}
    print(json.dumps({revision:summary[revision]}),flush=True)
summary['filament_reduction_percent']=100*(1-summary['R2']['filament_g']/summary['original']['filament_g'])
summary['time_reduction_percent']=100*(1-summary['R2']['total_seconds']/summary['original']['total_seconds'])
summary['settings']={'printer':'Bambu Lab H2D, 0.4 mm standard nozzle','filament':'Bambu PLA Basic',
                     'layer_height_mm':0.20,'walls':3,'infill':'10% gyroid','top_bottom_layers':4,'brim':'none','supports':'none',
                     'scope':'Four coupons on one plate; both revisions use the same settings.'}
(OUT/'slicer_comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
