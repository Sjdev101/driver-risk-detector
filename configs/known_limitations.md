# Known Limitations

## 1. EAR Head Pose Sensitivity
**Problem:** Drowsiness detection triggers incorrectly when driver tilts 
head backwards, even with eyes open. EAR formula cannot distinguish 
between eye closure and head rotation.

**Impact:** False positive drowsiness alerts on head tilt.

**Proposed Fix:** Implement head pose estimation using facial landmarks 
to normalize EAR calculation relative to head rotation angle before 
thresholding.

**Reference:** Euler angle estimation from facial landmarks using 
solvePnP (OpenCV).

## 2. Landmark Accuracy on Non-Generic Eye Shapes
**Problem:** MediaPipe's 6-point eye landmarks don't perfectly fit all 
eye shapes, reducing EAR sensitivity.

**Impact:** Smaller EAR range between open and closed eyes, requiring 
per-user calibration.

**Proposed Fix:** Use a larger subset of landmarks to build a convex 
hull around the eye for more accurate height estimation.
```

