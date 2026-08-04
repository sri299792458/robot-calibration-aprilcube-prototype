# G1 camera calibration report

- Dataset SHA-256: `8369ba65246418f0f0844a6adfde173b19fb9ff542ca95497dc5253cf518cf11`
- Result SHA-256: `3520a1e965722362f1b7a63e722bb4f1495fbe5eaff4ef1a7471f63027ba384f`
- Training radial RMS: 9.3401 px
- Holdout radial RMS: 5.8587 px
- Jacobian rank: 12/12
- Jacobian condition number: 133.114

## Residual-first decision

Inspect `corner_residuals.csv`, per-pose RMS, tag grouping, holdout error, and the calibration-joint correlations in `result.json` before freeing any joint offset. A joint-correlated pattern is evidence to investigate, not automatic permission to add parameters.
