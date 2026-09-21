# Input provenance

- Source repository: https://github.com/sri299792458/robot-calibration-aprilcube-prototype
- Source branch at handoff: `feature/dex3-aruco-calibration`
- Source fixture commit: `b555c3adea33966607c26837264230dbc1208c6c`
- Carrier mesh bytes are retained from that commit. The manifest is reduced
  to the carrier definition and frozen mount inputs, without the old frame.
- Camera layout source: https://github.com/sri299792458/g1-dex3-tabletop
- Camera source commit: `a26cf3752a54e1aa7f42398e48a0b5e230bbda41`
- Measured camera bundle: `dex3_shared_20260812_selected_free`
- Bundle SHA-256: `dae03341c35f7844cfb43cfee0a92386f2c508d26fa081f6683fb19ece99a2f3`
- Camera serial: `348522074178`

Robot model references are from the supplied handoff's Unitree ROS model;
the upstream license is preserved in `robot_model/UNITREE_ROS_LICENSE`.
The current frame geometry is the V7 native Fusion redesign.
