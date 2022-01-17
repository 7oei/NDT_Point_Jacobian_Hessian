import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
osp = os.path
np.random.seed(0)
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "build"))
import libndt_sample

def visualize_pc(ax, pc, label):
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], label=label)

mu = 0
sigma = 1.0

# create reference point cloud
sample_count = 100
reference_pc = np.random.normal(mu, sigma, (sample_count, 3))

# create map
ndt = libndt_sample.NDT()
ndt.create_map(reference_pc)

# create scan point cloud
euler = np.array([ 5.0, 0, 0.0])
rot = Rotation.from_euler('zyx', euler, degrees=True)
trans = np.array([ 0.1, 0.0, 0.0])
scan_pc = np.apply_along_axis(lambda x: rot.apply(x) + trans, 1, reference_pc)

# registration and get result transform.
# TODO: use boost python

transform = np.eye(4)
ans_pc = np.apply_along_axis(lambda x: transform[:3, :3].dot(x) + transform[0:3, 3], 1, scan_pc)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

visualize_pc(ax, reference_pc, "reference")
visualize_pc(ax, scan_pc, "scan")
visualize_pc(ax, ans_pc, "registered")

ax.legend()

plt.show()
