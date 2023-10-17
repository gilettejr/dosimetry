#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:54:38 2023

@author: robertsoncl
"""
from beamProfile import beamProfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class simIntensityProfile(beamProfile):
    def __init__(self, phspFile, xfov, yfov):

        def phspToHist(phsp, xfov, yfov):
            data_sim, x_edges, y_edges = np.histogram2d(
                phsp['X'], phsp['Y'], bins=100, range=[[-xfov, xfov], [-yfov, yfov]]
            )
            x_centres = (x_edges[:-1] + x_edges[1:]) / 2
            y_centres = (y_edges[:-1] + y_edges[1:]) / 2
            x, y = np.meshgrid(x_centres, y_centres)
            return data_sim, x, y

        def getSlices(phsp, slice_width=1):
            phsp_xslice = phsp[(phsp["Y"] < slice_width)]
            phsp_xslice = phsp_xslice[(phsp_xslice["Y"] > -slice_width)]
            phsp_yslice = phsp[(phsp["X"] < slice_width)]
            phsp_yslice = phsp_yslice[(phsp_yslice["X"] > -slice_width)]
            return phsp_xslice, phsp_yslice
        self.phspToHist = phspToHist
        self.getSlices = getSlices
        self.mmWidth = xfov*2
        self.mmHeight = yfov*2
        super(simIntensityProfile, self).__init__()
        self.sim_intensity = True

        particles = 'e'

        # read Topas ASCII output file
        phase_space = pd.read_csv(
            phspFile,
            names=["X", "Y", "Z", "PX", "PY",
                   "E", "Weight", "PDG", "9", "10"],
            delim_whitespace=True,
        )
        phase_space["X"] = phase_space["X"] * 10
        phase_space["Y"] = phase_space["Y"] * 10
        phase_space['PX'] = phase_space['PX'] * 1000
        phase_space['PY'] = phase_space['PY'] * 1000
        # add "R" column for radial distance from origin in mm
        phase_space["R"] = np.sqrt(
            np.square(phase_space["X"]) + np.square(phase_space["Y"])
        )

        gamma_phase_space = phase_space.copy()
        positron_phase_space = phase_space.copy()
        # create DataFrame containing only electron data at patient
        electron_phase_space = phase_space.drop(
            phase_space[phase_space["PDG"] != 11].index
        )
        # create DataFrame containing only gamma data at patient
        gamma_phase_space = gamma_phase_space.drop(
            phase_space[phase_space["PDG"] != 22].index
        )

        # create DataFrame containing only gamma data at patient
        positron_phase_space = positron_phase_space.drop(
            phase_space[phase_space["PDG"] != -11].index
        )
        phsp_dict = {
            "all": phase_space,
            "e": electron_phase_space,
            "y": gamma_phase_space,
            "p": positron_phase_space
        }
        try:
            phsp = phsp_dict[particles]
        except KeyError:
            print('Particle type not found, should be one of all, e, or y')
        self.phsp = phsp
        self.dosemap, self.smeshX, self.smeshY = phspToHist(
            phsp, xfov, yfov)

        self.pxheight = np.shape(self.dosemap)[0]
        self.pxwidth = np.shape(self.dosemap)[1]
        self.scale = self.mmWidth/self.pxwidth

    def get_sim_strips(self, slicewidth_mm, xpos_mm, ypos_mm):
        xpos, ypos, width = (xpos_mm+self.mmWidth/2)/self.scale, ypos_mm+(self.mmHeight/2) / \
            self.scale, slicewidth_mm/self.scale
        x_slice = self.dosemap[int(ypos-width/2):int(ypos+width/2)]
        y_slice = (np.rot90(self.dosemap)[
            int(xpos-width/2):int(xpos+width/2)])
        x_slice = np.mean(x_slice, axis=0)
        y_slice = np.mean(y_slice, axis=0)
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.xpos_mm = xpos_mm
        self.ypos_mm = ypos_mm
        self.slicewidth_mm = slicewidth_mm

    def show_transverse_beam(self, fov=50, col=50, slice_width=1, eRange=False):
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        get_slices = self.getSlices(slice_width=slice_width)
        phsp = self.phsp
        phsp_xslice, phsp_yslice = get_slices(phsp)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].hist(phsp_xslice["X"], bins=50, range=[-fov, fov], color="b")
        ax[0, 0].set_xlabel("X [mm]")
        ax[0, 0].set_ylabel("N")
        ax[0, 0].set_title("X Distribution")
        ax[0, 1].hist(phsp_yslice["Y"], bins=50, range=[-fov, fov], color="b")
        ax[0, 1].set_xlabel("Y [mm]")
        ax[0, 1].set_ylabel("N")
        ax[0, 1].set_title("Y Distribution")
        ax[1, 0].hist2d(
            phsp["X"], phsp["Y"], bins=100, range=[[-fov, fov], [-fov, fov]], cmap="jet"
        )
        ax[1, 0].set_xlabel("X [mm]")
        ax[1, 0].set_ylabel("Y [mm]")
        ax[1, 0].set_title("XY Distribution")
        if eRange is False:
            eRange = [0, max(phsp['E'])]
        ax[1, 1].hist(phsp["E"], bins=100, color="k", range=eRange)
        ax[1, 1].set_xlabel("E [MeV]")
        ax[1, 1].set_ylabel("N")
        ax[1, 1].set_yscale('log')

        ax[1, 1].set_title("Energy Spectrum")

        col_phsp = phsp[(phsp.R < col)]
        col_phsp = col_phsp.dropna()
        print(
            "Electron Transmission within Virtual Collimator = " +
            str(len(col_phsp["X"]) / len(phsp["X"]) * 100)
        )
        print("Mean Energy at Dump:" + str(np.mean(col_phsp["E"])))


filename = '/home/robertsoncl/dphil/s1Data/Dists/TOPAS/flat10.0200.001000000.8Copper300'
sip = simIntensityProfile(filename, 20, 20)
sip.get_sim_strips(1, 0, 0)
print(sip.get1DSuperGaussian(power=1))
sip.plot_xy_fits()
