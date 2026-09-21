# Geometry inputs for the current mount

Carrier geometry and mounting coordinates derive from source commit
`b555c3adea33966607c26837264230dbc1208c6c`; see `PROVENANCE.md`.
The carrier meshes match the retained accepted meshes under
`cad/torso_charuco_fixture/`. The manifest has been reduced to the carrier
definition and frozen input coordinates; previous frame and hardware
instructions are removed. `reference_code/config/torso_fixture_layout_camera.json`
matches the repository's measured camera configuration semantically.

The robot mesh and URDF reference files carry the upstream license included
as `robot_model/UNITREE_ROS_LICENSE`. This partial snapshot is sufficient
for the mount scripts, not a complete robot simulation model.

Use `../README.md` and `../FIT_STATUS.md` for current design and hardware status.
