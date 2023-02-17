__author__ = 'BTWang'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as inter


def divider(epsilon_intr, plot=False):

    '''
    A theoretical track on lambda-epsilon diagram based on TVT to divide galaxies into fast and slow, assuming velocity anisotropy delta = 0.7*epsilon_intr
    plot=True to check the dividing line.
    e.g.,
    func_div = divider(0.525, plot=True)
    # epsilon_intr = 0.525 -> lambda_intr = 0.4
    cri_fr = data['lambda']>func_div(data['epsilon'])
    cri_sr = data['lambda']<=func_div(data['epsilon'])
    '''

    alpha = 0.15
    epsilon_intr1 = epsilon_intr
    inc = np.arange(0.09,90,0.09)
    e = (1-(1-epsilon_intr1)**2)**0.5
    Omega = (0.5*(np.arcsin(e)/(1-e**2)**0.5-e))/(e-np.arcsin(e)*(1-e**2)**0.5)
    delta_certain = 0.7*epsilon_intr1
    ratio_vs_certain = (((1-delta_certain)*Omega-1)/(alpha*(1-delta_certain)*Omega+1))**0.5

    ratio_vs_obs2 = ratio_vs_certain*(np.sin(np.radians(inc)))/(1-delta_certain*np.cos(np.radians(inc))**2)**0.5
    lambda_e_obs2 = (1.1*ratio_vs_obs2)/(1+1.1**2*ratio_vs_obs2**2)**0.5
    epsilon_obs2 = 1-(1+epsilon_intr1*(epsilon_intr1-2)*np.sin(np.radians(inc))**2)**0.5

    func_divider = inter.InterpolatedUnivariateSpline(epsilon_obs2,lambda_e_obs2)

    if plot:
        plt.figure()
        plt.plot(epsilon_obs2,lambda_e_obs2, 'k.')
        plt.plot(np.arange(0,0.99,0.01), func_divider(np.arange(0,0.99,0.01)), lw=1, alpha=0.7)
        plt.xlim(0,1)
        plt.ylim(0,0.9)

    return func_divider
