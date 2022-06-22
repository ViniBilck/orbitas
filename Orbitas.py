import os
import argparse
import configparser
import solver
import numpy as np
from astropy import units, constants
from scipy import integrate as i
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="", dest='config_file', default='config.ini', type=str)

    args = parser.parse_args()
    here = os.path.realpath('.ipynb_checkpoints')
    config = configparser.ConfigParser(defaults={'here': here})
    config.read(args.config_file)

    ## Gal_1
    pos_gal1 = [float(_) for _ in list(config.get("Gal1", "pos_array").split(","))]
    vel_gal1 = [float(_) for _ in list(config.get("Gal1", "vel_array").split(","))]
    m_gal1 = float(config.get("Gal1", "mass"))
    ## Gal_2
    pos_gal2 = [float(_) for _ in list(config.get("Gal2", "pos_array").split(","))]
    vel_gal2 = [float(_) for _ in list(config.get("Gal2", "vel_array").split(","))]
    m_gal2 = float(config.get("Gal2", "mass"))

    data = np.concatenate([pos_gal1, pos_gal2, vel_gal1, vel_gal2])
    ## Dados
    g_pc = 1e10 * constants.G.to((units.kpc ** 3) / (units.M_sun * units.Gyr ** 2)).value  ## kpc**3 / (Msol * Gyr**2)

    ## EDO Solver
    sol = i.solve_ivp(solver.rtt, (0, 1),
                      data,
                      args=(m_gal1, m_gal2, g_pc),
                      t_eval=np.arange(0, 1, 1e-4), method='Radau')

    ## Vetores Posição em Relação a (0,0,0)
    r_1 = np.column_stack((sol.y[0], sol.y[1], sol.y[2]))  # X, Y, Z
    v_1 = np.column_stack((sol.y[6], sol.y[7], sol.y[8]))
    r_2 = np.column_stack((sol.y[3], sol.y[4], sol.y[5]))  # X, Y, Z
    v_2 = np.column_stack((sol.y[9], sol.y[10], sol.y[11]))
    rcm = ((m_gal1 * r_1) + (m_gal2 * r_2)) / (m_gal1 + m_gal2)  # R do centro de massa
    vcm = ((m_gal1 * v_1) + (m_gal2 * v_2)) / (m_gal1 + m_gal2)  # V do centro de massa

    ## Em relação a rcm
    r1rc = rcm - r_1
    r2rc = rcm - r_2
    v1rc = vcm - v_1
    v2rc = vcm - v_2

    print(f"Ultimo ponto em relação ao centro de massa")
    print(f"PR1: {r1rc[-1]} | VR1: {-v1rc[-1]}")
    print(f"PR2: {r2rc[-1]} | VR2: {-v2rc[-1]}")

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(r1rc[:, 0], r1rc[:, 1], r1rc[:, 2], label="Galaxy 1")
    ax.plot3D(r2rc[:, 0], r2rc[:, 1], r2rc[:, 2], label="Galaxy 2")
    ax.scatter(0, 0, 0, c='r')

    ax.set_xlabel('x [Kpc]')
    ax.set_ylabel('y [Kpc]')
    ax.set_zlabel('z [Kpc]')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()