"""Package the final CAD and print artifacts without local Python dependencies."""
from pathlib import Path
import hashlib, json, zipfile

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'artifacts/g1_mount_v7'
data_path = OUT / 'design_inputs.json'
data = json.loads(data_path.read_text(encoding='utf-8'))
data['carrier_print_to_board_translation_mm'] = [-15,-15,0]
data['notes'] = [
    'M6 x 20 projects 6.4 mm past the nominal outer shell with a 1.6 mm washer; usable thread engagement depends on the unmeasured insert recess.'
    if n.startswith('M6 x 20') else n for n in data['notes']]
data_path.write_text(json.dumps(data, indent=2))

files = [(p, p.relative_to(OUT).as_posix()) for p in sorted(OUT.glob('*.json'))]
files.extend((p,p.relative_to(OUT).as_posix()) for p in sorted(OUT.glob('*.md')))
files.extend((p,p.relative_to(OUT).as_posix()) for p in sorted((OUT/'reference_inputs').rglob('*')) if p.is_file())
for folder in ['cad','print_parts','fit_coupons']:
    files.extend((p,p.relative_to(OUT).as_posix()) for p in sorted((OUT/folder).iterdir()) if p.is_file())
for name in ['fusion_assembly.png','board_connections.png','m6_root_section.png','print_orientations.png']:
    files.append((OUT/'renders'/name, 'renders/'+name))
for name in ['prepare_g1_mount_v7.py','build_g1_mount_v7_fusion.py','finish_g1_mount_v7_fusion.py',
             'g1_mount_v7_coupons_fusion.py','validate_g1_mount_v7.py','check_g1_mount_v7_board_contact.py',
             'render_g1_mount_v7_review.py','package_g1_mount_v7.py',
             'revise_g1_fit_coupons_r2_fusion.py','validate_g1_coupons_r2.py','finish_g1_coupons_r2_fusion.py',
             'slice_g1_coupons_r2.py','package_g1_coupons_r2.py','estimate_g1_print_time.py','G1_MOUNT_README.md']:
    files.append((BASE/'scripts'/name, 'source/'+name))

assert all(p.is_file() for p,_ in files)
assert len({n for _,n in files}) == len(files)
with zipfile.ZipFile(OUT/'cad/G1_calibration_mount_native.f3d') as native:
    # Fusion uses a ZIP compression method unsupported by Python 3.12.
    # Inspect its directory; the containing delivery ZIP verifies F3D bytes.
    assert native.infolist(), 'Native archive has no entries'
payloads = {name:p.read_bytes() for p,name in files}
payloads['README.md'] = payloads['README.md'].decode('utf-8').replace(
    '../../scripts/G1_MOUNT_README.md', 'source/G1_MOUNT_README.md').encode('utf-8')
payloads['source/G1_MOUNT_README.md'] = payloads['source/G1_MOUNT_README.md'].decode('utf-8').replace(
    '../artifacts/g1_mount_v7/', '../').encode('utf-8')
manifest = {name:{'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()}
target = OUT/'G1_calibration_mount_review.zip'
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
    for name,data in payloads.items(): package.writestr(name,data)
    package.writestr('package_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as package:
    assert package.testzip() is None, 'Package failed CRC check'
    assert not any('python_packages' in name or 'v8' in name.lower() for name in package.namelist())
print(json.dumps({'zip':str(target), 'files':len(files)+1, 'bytes':target.stat().st_size,
                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'CRC':'pass'},indent=2))
