import os
import sys
import numpy as np
osp = os.path
np.random.seed(0)
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "build"))

import libndt_sample

# create reference point cloud
point = np.array([0, 0 ,0], dtype=np.float32)
transform = np.array([0, 0, 0, 30, 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz

# create map
ndt = libndt_sample.NDT()
score, jacobian, hessian = ndt.calcScore(point, transform)

jacobian = list(map(float, jacobian))

jacobian = np.array(list(map(float, jacobian)), dtype=np.float32).reshape((6, 3)).T
hessian = np.array(list(map(float, hessian)), dtype=np.float32).reshape((6, 18)).T

print(score)
print(jacobian)
print(hessian)
