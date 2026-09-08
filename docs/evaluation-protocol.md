# Badminton evidence and evaluation protocol

The product must not launch a capability merely because an interface can draw
it. A capability launches only after evaluation on held-out, human-labelled
badminton video representing different phones, rear/side cameras, lighting,
singles and doubles, skill levels, occlusions, court colours, and shuttle
speeds.

| Capability | Launch measure | Gate before customer coaching |
| --- | --- | --- |
| Player identity | IDF1 and identity-switch rate | Stable selected-player track on rally clips |
| Pose | keypoint OKS / PCK and temporal jitter | Joint-level confidence calibrated against labels |
| Shuttle | recall, precision, trajectory gap rate | Never interpolate an unobserved shuttle as evidence |
| Court registration | labelled-point / line reprojection error | Reject metre claims when line support or error fails |
| Contact | median timing error vs labelled hits | Do not derive split-step/recovery from pose alone |
| Stroke labels | macro F1 by shot class and camera | No heuristic smash/clear/drop labels |
| Coaching | blinded coach agreement and harmful-advice review | Every claim must cite visible evidence and uncertainty |

Store anonymised, consented expert corrections as training data. Version every
model and test set. Compare every release against the previous model by camera
angle and demographic/skill slice; block a release that improves an aggregate
while materially regressing a supported slice. A multi-camera 3D product is a
separate, explicitly calibrated capability—not an extrapolation of monocular
pose.
