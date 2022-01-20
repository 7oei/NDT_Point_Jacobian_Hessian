import os
import sys
import numpy as np
import math
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R

osp = os.path
np.random.seed(0)
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "build"))

import libndt_sample

pi = math.pi
t = np.linspace(0, 2*pi, 100)  #0から2πまでの範囲を100分割したnumpy配列
#t = np.linspace(-1, 1, 100)  #0から2πまでの範囲を100分割したnumpy配列
theta = np.sin(t)
j = np.sin(t)
h = np.sin(t)
jx = np.sin(t)
jy = np.sin(t)
jz = np.sin(t)
hxx = np.zeros(100)
jrz = np.zeros(100)
hrzrz = np.zeros(100)

for num in range(100):
    
    # create reference point cloud
    
    # transform = np.array([0, 0, 0, theta[num], 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    #transform = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    #transform = np.array([t[num], 0, 0, 0, 0, 0], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    #point = np.array([t[num],0,0], dtype=np.float32)
    #p1 = np.array([-2/3+t[num],-2/3,0], dtype=np.float32)
    #p2 = np.array([1/3+t[num],-2/3,0], dtype=np.float32)
    #p3 = np.array([1/3+t[num],4/3,0], dtype=np.float32)

    transform = np.array([0, 0, 0, 0, 0, t[num]], dtype=np.float32) # tx, ty, tz, rx, ry, rz
    p1 = np.array([-2/3,-2/3,0], dtype=np.float32)
    p2 = np.array([1/3,-2/3,0], dtype=np.float32)
    p3 = np.array([1/3,4/3,0], dtype=np.float32)

    r = R.from_euler('z', t[num], degrees=False)
    r.apply(p1)
    r.apply(p2)
    r.apply(p3)

    point_list = [p1, p2, p3]
    score_list = []
    jacobian_list = []
    hessian_list = []
    # create map
    ndt = libndt_sample.NDT()

    for point in point_list:
        score, jacobian, hessian = ndt.calcScore(point, transform)

        jacobian = list(map(float, jacobian))

        jacobian = np.array(list(map(float, jacobian)), dtype=np.float32).reshape((6, 1)).T
        hessian = np.array(list(map(float, hessian)), dtype=np.float32).reshape((6, 6)).T

        score_list.append(score)
        jacobian_list.append(jacobian)
        hessian_list.append(hessian)


    jacobian = sum(jacobian_list)
    hessian = sum(hessian_list)
    score = sum(score_list)
    j[num] = jacobian[0,3]
    h[num] = hessian[3,3]
    jx[num] = jacobian[0,0]
    jy[num] = jacobian[0,1]
    jz[num] = jacobian[0,2]
    hxx[num] = hessian[0,0]
    
    jrz[num] = jacobian[0,5]
    hrzrz[num] = hessian[5,5]
    
    print(score)
    print(jacobian)
    print(hessian)
    print(transform)

# pyplot.plot(t, theta, label='theta')
# pyplot.plot(t, j, label='j')
# pyplot.plot(t, h, label='h')
#pyplot.plot(t, jx, label='jx')
#pyplot.plot(t, hxx, label='hxx')
#pyplot.plot(t, jy, label='jy')
#pyplot.plot(t, jz, label='jz')

pyplot.plot(t, jrz, label='jrz')
pyplot.plot(t, hrzrz, label='hrzrz')
pyplot.legend()
pyplot.show()
