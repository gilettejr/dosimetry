#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:49:50 2023

@author: robertsoncl
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from yagProfile import yagProfile

import sys
# 200: sigma_x=0.86,
#    sigma_y=1.14,
#    sigma_px=4.17,
#    sigma_py=3.26,

# 150:sigma_x=1.83657125,
# sigma_y=0.99433641,
# sigma_px=7.80910106,
# sigma_py=5.78318716,


def runSim(thickness, s1Position, yagPosition, NParticles):
    sys.path.append('/home/robertsoncl/dphil/rf-track-2.1.6/')
    from scorerScript import scorerScript
    from beamScript import beamScript
    from scatScript import scatScript
    from guiScript import guiScript
    gs = guiScript()
    ss = scatScript()
    bs = beamScript()
    srs = scorerScript()

   # bs.addGaussianPhspBeam(0.86, 1.14, 4.17,
   #                        3.26, 200, 0.05, NParticles)
    bs.addGaussianPhspBeam(1.83657125,
                           0.99433641,
                           7.80910106,
                           5.78318716, 150, 0.0, NParticles)
    ss.addFlatScatterer(thickness, 'PLA', s1Position)
    srs.addPhspScorer(yagPosition)
    gs.runTopas()


def show30mm():
    matPath = "matlab_images/30_250_200.mat"
    matFile = loadmat(matPath)
    yagFile = np.transpose(matFile["Images_beam"][0])
    yip = yagProfile(
        '/home/robertsoncl/topas/matlab_images/30_250_200.mat')


def plot250Stats():
    s30 = np.zeros((2, 10))
    r30 = np.zeros((2, 10))
    s20 = np.zeros((2, 10))
    r20 = np.zeros((2, 10))
    s10 = np.zeros((2, 10))
    r10 = np.zeros((2, 10))

    matPath = "matlab_images/30_250_150.mat"
    matFile = loadmat(matPath)

    for i in range(10):
        runSim(30, 220, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r30[0][i], r30[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s30[0][i], s30[1][i] = sfit.x_stddev.value, sfit.y_stddev.value
    matPath = "matlab_images/20_250_150.mat"
    matFile = loadmat(matPath)
    for i in range(10):
        runSim(20, 225, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r20[0][i], r20[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s20[0][i], s20[1][i] = sfit.x_stddev.value, sfit.y_stddev.value
    matPath = "matlab_images/10_250_150.mat"
    matFile = loadmat(matPath)

    for i in range(10):
        runSim(10, 230, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r10[0][i], r10[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s10[0][i], s10[1][i] = sfit.x_stddev.value, sfit.y_stddev.value

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    rx = [np.median(r10[0]), np.median(r20[0]), np.median(r30[0])]
    rxErr = [np.std(r10[0]), np.std(r20[0]), np.std(r30[0])]
    ry = [np.median(r10[1]), np.median(r20[1]), np.median(r30[1])]
    ryErr = [np.std(r10[1]), np.std(r20[1]), np.std(r30[1])]
    sx = [np.median(s10[0]), np.median(s20[0]), np.median(s30[0])]
    sxErr = [np.std(s10[0]), np.std(s20[0]), np.std(s30[0])]
    sy = [np.median(s10[1]), np.median(s20[1]), np.median(s30[1])]
    syErr = [np.std(s10[1]), np.std(s20[1]), np.std(s30[1])]
    positions = [10, 20, 30]
    ax[0].errorbar(positions, rx, yerr=rxErr, capsize=2,
                   color='black', label='Experiment')
    ax[0].errorbar(positions, sx, yerr=sxErr, capsize=2,
                   color='red', label='Simulation')
    ax[0].set_ylabel('$\sigma_x$')
    ax[0].legend()
    ax[0].grid()

    ax[1].errorbar(positions, ry, yerr=ryErr, capsize=2,
                   color='black', label='Experiment')
    ax[1].errorbar(positions, sy, yerr=syErr, capsize=2,
                   color='red', label='Simulation')
    ax[1].set_ylabel('$\sigma_y$')
    ax[1].legend()
    ax[0].grid()


def plot500Stats():
    s30 = np.zeros((2, 10))
    r30 = np.zeros((2, 10))
    s20 = np.zeros((2, 10))
    r20 = np.zeros((2, 10))
    s10 = np.zeros((2, 10))
    r10 = np.zeros((2, 10))

    matPath = "matlab_images/30_500_150.mat"
    matFile = loadmat(matPath)

    for i in range(10):
        runSim(30, 0, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r30[0][i], r30[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s30[0][i], s30[1][i] = sfit.x_stddev.value, sfit.y_stddev.value
    matPath = "matlab_images/20_500_150.mat"
    matFile = loadmat(matPath)
    for i in range(10):
        runSim(20, 0, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r20[0][i], r20[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s20[0][i], s20[1][i] = sfit.x_stddev.value, sfit.y_stddev.value
    matPath = "matlab_images/10_500_150.mat"
    matFile = loadmat(matPath)

    for i in range(10):
        runSim(10, 0, 500, 10000)
        yagFile = np.transpose(matFile['Images_beam'])[i]
        yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
        rfit, sfit = yip.fitBoth()
        r10[0][i], r10[1][i] = rfit.x_stddev.value, rfit.y_stddev.value
        s10[0][i], s10[1][i] = sfit.x_stddev.value, sfit.y_stddev.value

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    rx = [np.median(r10[0]), np.median(r20[0]), np.median(r30[0])]
    rxErr = [np.std(r10[0]), np.std(r20[0]), np.std(r30[0])]
    ry = [np.median(r10[1]), np.median(r20[1]), np.median(r30[1])]
    ryErr = [np.std(r10[1]), np.std(r20[1]), np.std(r30[1])]
    sx = [np.median(s10[0]), np.median(s20[0]), np.median(s30[0])]
    sxErr = [np.std(s10[0]), np.std(s20[0]), np.std(s30[0])]
    sy = [np.median(s10[1]), np.median(s20[1]), np.median(s30[1])]
    syErr = [np.std(s10[1]), np.std(s20[1]), np.std(s30[1])]
    positions = [10, 20, 30]
    ax[0].errorbar(positions, rx, yerr=rxErr, capsize=2,
                   color='black', label='Experiment')
    ax[0].errorbar(positions, sx, yerr=sxErr, capsize=2,
                   color='red', label='Simulation')
    ax[0].grid(True)
    ax[0].set_ylabel('$\sigma_x$')
    ax[0].legend()

    ax[1].errorbar(positions, ry, yerr=ryErr, capsize=2,
                   color='black', label='Experiment')
    ax[1].errorbar(positions, sy, yerr=syErr, capsize=2,
                   color='red', label='Simulation')
    ax[1].set_ylabel('$\sigma_y$')
    ax[1].legend()
    ax[1].grid(True)


plot250Stats()
#matPath = "matlab_images/30_250_200.mat"
#matFile = loadmat(matPath)
#runSim(30, 220, 500, 10000)
#yagFile = np.transpose(matFile['Images_beam'])[0]
#yip = yagIntensityPlotter(yagFile, 'phspScorer1.phsp')
#rfit, sfit = yip.fitBoth()
#print(rfit.x_stddev.value, rfit.y_stddev.value)
#print(sfit.x_stddev.value, sfit.y_stddev.value)
