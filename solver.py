from scipy import integrate as i
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
#plt.style.use('science')

def rtt(t, data, m1, m2, g):
    g_  = g
    r1 = np.array(data[:3])
    r2 = np.array(data[3:6])
    R = r1 - r2
    r1r2_mod = np.linalg.norm(r1 - r2)
    v1 = np.array(data[6:9])
    v2 = np.array(data[9:12])
    a1 = -(m2 * g_ * R) / r1r2_mod**3
    a2 = (m1 * g_ * R) / r1r2_mod**3

    return np.concatenate([v1, v2, a1, a2])
