# G1 V7 mount scripts

Start with [the CAD and printing guide](../artifacts/g1_mount_v7/README.md)
and [physical fit status](../artifacts/g1_mount_v7/FIT_STATUS.md). The F3D is
self-contained. STLs and the coupon 3MF are ready to open without running
Python. Geometry and M6 lengths remain subject to physical validation.

These scripts preserve the native Fusion build, checks and Bambu comparisons.
They locate the checkout from `__file__`; moving the checkout is supported.
Frozen carrier, shell, head, URDF and camera inputs are included under
`artifacts/g1_mount_v7/reference_inputs/`. The obsolete V6 frame generator has been removed; these are the current mount build scripts.

## Dependencies and execution

Local mesh/plot scripts require Python with numpy, trimesh, scipy, shapely,
rtree and matplotlib. Install these in your own environment; the original
vendored `python_packages` directory is deliberately omitted. The scripts
also work with these packages installed normally on Python's import path.
Fusion scripts additionally require Autodesk Fusion's `adsk` API and must
run inside Fusion. They expect the document named
`G1 V7 - broad crossbars and flat side plates`.

When invoking a script through Fusion MCP, load it with its real path so
`__file__` resolves correctly:

```python
from pathlib import Path

def run(_context):
    path = Path('/absolute/path/to/checkout/scripts/revise_g1_fit_coupons_r2_fusion.py')
    namespace = {'__file__': str(path), '__name__': 'g1_script'}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    namespace['run'](_context)
```

`build_g1_mount_v7_fusion.py` builds one requested part at a time through
`build('side_left')`, `build('side_right')`, `build('crossbar_upper')` and
`build('crossbar_lower')`. Start from a new empty document with the expected
name; existing component names are deliberately rejected. The coupon
builder also expects components to be absent. The R2 revision script is
idempotent for its named trim/marking features. Native scripts can alter the
active design; use the supplied archive or a separate document for review.

`validate_g1_mount_v7.py` writes centered print STLs and checks complete
geometry/access/target rays. `check_g1_mount_v7_board_contact.py` checks the
four carrier contacts. `validate_g1_coupons_r2.py` compares revised coupons
with the included pre-R2 and precision meshes; the pre-R2 STLs in
`original_reference/` are comparison inputs, not current print files.

`slice_g1_coupons_r2.py` requires Bambu Studio at its standard Windows path.
The comparison profiles are included under
`artifacts/g1_print_time/controlled_comparison/`. It slices both revisions;
the included R2 3MF is already sliced. `estimate_g1_print_time.py --quick-check` regenerates the limited side-plate comparison using locally
installed Bambu profiles; it is not a complete structural print study.

Run `package_g1_coupons_r2.py` then `package_g1_mount_v7.py` to refresh the
ZIPs. They include file hashes and verify ZIP integrity. The ZIP `source/`
files are provenance copies, not a separate runnable directory layout.
