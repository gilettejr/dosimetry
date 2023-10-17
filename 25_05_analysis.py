#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:03:56 2023

@author: robertsoncl
"""
from flattenedFilmDosimetryTools import FlatRadioChromicFilm
from flattenedFilmProfileTools import flatYagProfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a_r = 9.53385895e-03
b_r = 4.40553270e-04
c_r = 3.71012295e+00

a_g = 1.36037090e-02
b_g = 1.48062223e-04
c_g = 5.72374622e+00

a_b = 2.22791213e-02
b_b = 3.04096810e-04
c_b = 1.43716055e+01

# dd1 F011-F033
# dd2 F034-F035, G001-G021


def film_main():
    # test
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/14-07-23/'
    name = 'Film_B026.tif'
    a = 18.5
    b = 26.5
    film = FlatRadioChromicFilm(path_to_folder+name)
    film.get_dose_strips(a_g, b_g, c_g, 1.1, a, b, 10)
    film.plot_xy_hists(film.xpos_mm, film.ypos_mm,
                       film.radius_mm, lim=40, savepath=path_to_folder+name+'.png')
    film.getDensityStats(a, b, 15)


def depthDose():
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    xmean = [9.89, 9.93, 9.94, 10.0, 10.17, 9.99, 9.92, 9.61, 9.45, 9.38, 9.5,
             9.35, 9.12, 9.02, 8.72, 8.78, 8.71, 8.37, 8.27, 7.95, 7.57, 7.16, 6.78]
    xstd = [0.14, 0.18, 0.14, 0.15, 0.16, 0.17, 0.15, 0.12, 0.17, 0.16, 0.14,
            0.10, 0.17, 0.12, 0.11, 0.14, 0.10, 0.12, 0.10, 0.16, 0.11, 0.07, 0.07]

    xmeanconv = [11.17, 11.01, 11.05, 11.14, 11.22, 11.37, 11.61, 11.21, 11.10, 11.12, 10.86,
                 10.65, 10.72, 10.65, 10.53, 10.45, 10.07, 9.66, 9.60, 9.30, 9.07, 8.58, 8.40]
    xstdconv = [0.20, 0.15, 0.14, 0.17, 0.18, 0.18, 0.17, 0.20, 0.13, 0.15, 0.16, 0.20,
                0.14, 0.13, 0.18, 0.18, 0.17, 0.14, 0.14, 0.11, 0.12, 0.11, 0.13]

    xsigmaconv = [4.16, 4.24, 4.24, 4.24, 4.24, 4.24, 4.24, 4.24, 4.33, 4.33,
                  4.41, 4.33, 4.41, 4.50, 4.59, 4.59, 4.75, 4.84, 4.92, 5.09, 5.18, 5.26, 5.52]
    xkurtconv = [-0.99, -0.98, -0.96, -0.95, -0.95, -0.93, -0.91, -0.91, -0.87, -0.84, -
                 0.81, -0.82, -0.78, -0.75, -0.71, -0.71, -0.68, -0.66, -0.64, -0.62, -0.61, -0.63, -0.59]

    xsigma = [4.24, 4.24, 4.24, 4.24, 4.24, 4.24, 4.24, 4.33, 4.33, 4.33, 4.41,
              4.41, 4.41, 4.50, 4.58, 4.67, 4.75, 4.84, 5.01, 5.09, 5.18, 5.34, 5.52,]
    xkurt = [-0.90, -0.90, -0.89, -0.87, -0.87, -0.87, -0.87, -0.84, -0.82, -0.79, -0.76, -
             0.76, -0.75, -0.71, -0.69, -0.66, -0.63, -0.63, -0.60, -0.60, -0.60, -0.59, -0.59,]

    ysigmaconv = [3.99, 4.07, 4.07, 4.07, 4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                  4.16, 4.16, 4.24, 4.24, 4.33, 4.41, 4.50, 4.58, 4.75, 4.84, 5.01, 5.09, 5.43]
    ykurtconv = [-0.84, -0.82, -0.82, -0.80, -0.80, -0.80, -0.79, -0.79, -0.78, -0.77, -
                 0.73, -0.73, -0.70, -0.71, -0.67, -0.64, -0.65, -0.63, -0.61, -0.60, -0.60, -0.59, -0.57]

    ysigma = [4.41, 4.33, 4.24, 4.24, 4.33, 4.24, 4.33, 4.33, 4.41, 4.41, 4.41,
              4.50, 4.41, 4.50, 4.67, 4.67, 4.67, 4.84, 4.92, 5.09, 5.17, 5.35, 5.52,]
    ykurt = [-0.62, -0.67, -0.74, -0.72, -0.67, -0.68, -0.68, -0.68, -0.66, -0.65, -0.65, -
             0.62, -0.65, -0.62, -0.57, -0.58, -0.60, -0.57, -0.56, -0.55, -0.55, -0.55, -0.55]

    depth = np.linspace(20, 130, 23)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.errorbar(depth, xmeanconv, yerr=xstdconv, capsize=2, color='blue',
                label='4 shots, 5.81nC transmitted charge')
    # ax.errorbar(depth, xmean, yerr=xstd, capsize=2, color='k',
    #            label='Single shot FLASH, 4.85nC transmitted charge')

    ax.set_xlabel('Water Depth [mm]')
    ax.set_ylabel('Dose [Gy]')
    ax.legend()
    ax.grid(True)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
    ax[0].errorbar(depth, xsigmaconv, yerr=0.1, capsize=2, color='blue',
                   label='4 shots (x)')
    ax[0].errorbar(depth, ysigmaconv, yerr=0.1, capsize=2, color='cyan',
                   label='4 shots (y)')

    # ax[0].errorbar(depth, xsigma, yerr=0.1, capsize=2, color='black',
    #               label='Single shot (x)')
    # ax[0].errorbar(depth, ysigma, yerr=0.1, capsize=2, color='red',
    #               label='Single shot (y)')

    # ax[1].set_xlabel('Water Depth [mm]')
    ax[0].set_ylabel('Beamsize [mm]')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].errorbar(depth, xkurtconv, yerr=0.01, capsize=2, color='blue',
                   label='4 shots (x)')
    ax[1].errorbar(depth, ykurtconv, yerr=0.01, capsize=2, color='cyan',
                   label='4 shots (y)')

    # ax[1].errorbar(depth, xkurt, yerr=0.01, capsize=2, color='black',
    #               label='Single shot (x)')
    # ax[1].errorbar(depth, ykurt, yerr=0.01, capsize=2, color='red',
    #               label='Single shot (y)')

    ax[1].set_xlabel('Water Depth [mm]')
    ax[1].set_ylabel('Kurtosis')
    ax[1].legend()
    ax[1].grid(True)


# depthDose()


# strip_deviation_plot()
# profile_main()
film_main()
