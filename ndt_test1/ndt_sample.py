import os
import sys
import numpy as np
import math
from matplotlib import pyplot

osp = os.path
np.random.seed(0)
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "build"))

import libndt_sample

pi = math.pi
t = np.linspace(0, 2*pi, 100)  #0から2πまでの範囲を100分割したnumpy配列
theta = np.sin(t)
j = np.sin(t)
h = np.sin(t)
jx = np.sin(t)
jy = np.sin(t)
jz = np.sin(t)
for num in range(100):
    
    # create reference point cloud
    point = np.array([0.99985,0.01745,0], dtype=np.float32)
    # transform = np.array([0, 0, 0, theta[num], 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    transform = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    # create map
    ndt = libndt_sample.NDT()
    score, jacobian, hessian = ndt.calcScore(point, transform)

    jacobian = list(map(float, jacobian))

    jacobian = np.array(list(map(float, jacobian)), dtype=np.float32).reshape((6, 1)).T
    hessian = np.array(list(map(float, hessian)), dtype=np.float32).reshape((6, 6)).T
    j[num] = jacobian[0,3]
    h[num] = hessian[3,3]
    jx[num] = jacobian[0,0]
    jy[num] = jacobian[0,1]
    jz[num] = jacobian[0,2]
    print(score)
    print(jacobian)
    print(hessian)

# pyplot.plot(t, theta, label='theta')
# pyplot.plot(t, j, label='j')
# pyplot.plot(t, h, label='h')
pyplot.plot(t, jx, label='jx')
pyplot.plot(t, jy, label='jy')
pyplot.plot(t, jz, label='jz')
pyplot.legend()
pyplot.show()