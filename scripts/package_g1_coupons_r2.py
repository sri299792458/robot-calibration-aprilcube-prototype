"""Package the compact coupon deliverables and update the main mount handoff."""
from pathlib import Path
import json,zipfile,hashlib,shutil,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'artifacts/g1_coupon_r2';MAIN=BASE/'artifacts/g1_mount_v7'
project=OUT/'slice_R2/G1_R2_fit_coupons.3mf'
if not project.exists():project=OUT/'G1_R2_fit_coupons.3mf'
with zipfile.ZipFile(project) as z:
    assert z.testzip() is None
    tree=ET.fromstring(z.read('3D/3dmodel.model'))
    ns={'m':'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    assert len(tree.findall('m:build/m:item',ns))==4, 'Expected four objects on the coupon plate'
if project.resolve() != (OUT/'G1_R2_fit_coupons.3mf').resolve():
    shutil.copy2(project,OUT/'G1_R2_fit_coupons.3mf')
main_readme=(OUT/'README.md').read_text(encoding='utf-8').replace('the individual STLs in `print_parts/`','the four `FIT_ONLY_*_PRINT.stl` files in this folder')
main_readme=main_readme.replace('`renders/identification_marks.png` shows the markings.','The markings are engraved on the flat washer side.')
main_readme=main_readme.replace('`geometry_qa.json` and `slicer_comparison.json`','`../compact_coupon_geometry_qa.json` and `../compact_coupon_slicer_comparison.json`')
(MAIN/'fit_coupons/README.md').write_text(main_readme,encoding='utf-8')
shutil.copy2(OUT/'FIT_STATUS.md',MAIN/'fit_coupons/FIT_STATUS.md')
shutil.copy2(project,MAIN/'fit_coupons/G1_R2_fit_coupons.3mf')
shutil.copy2(OUT/'geometry_qa.json',MAIN/'compact_coupon_geometry_qa.json')
shutil.copy2(OUT/'slicer_comparison.json',MAIN/'compact_coupon_slicer_comparison.json')
files=[OUT/'FIT_STATUS.md',OUT/'README.md',OUT/'G1_R2_fit_coupons.3mf',OUT/'geometry_qa.json',OUT/'slicer_comparison.json']
files+=sorted((OUT/'print_parts').glob('*.stl'))+sorted((OUT/'renders').glob('*.png'))
manifest={p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
target=OUT/'G1_compact_fit_coupons_R2.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(OUT).as_posix())
    z.writestr('package_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:assert z.testzip() is None
print(json.dumps({'package':str(target),'bytes':target.stat().st_size,'verified_files':len(files)+1,'plate_objects':4},indent=2))
