#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:11:42 2023

@author: robertsoncl
"""
from beamProfile import beamProfile
import numpy as np
from topas2numpy import BinnedResult

# import phase space from defined file


class simDoseProfile(beamProfile):
    def __init__(self, doseFilePath, simParticles, acChargenC, xyBinTomm, zBinTomm=1):

        dosemap = np.squeeze(BinnedResult(doseFilePath).data['Sum'])

        self.scale = xyBinTomm
        super(simDoseProfile, self).__init__()
        self.pxheight = np.shape(dosemap)[0]
        self.pxwidth = np.shape(dosemap)[1]
        self.mmWidth = self.px_to_mm(self.pxwidth)
        self.mmHeight = self.px_to_mm(self.pxheight)
        # charge to simulate for# nC
        # electron charge
        eCharge = 1.60217663e-19  # C
        # numer of particles to estimate total dose
        target_nPart = acChargenC*1e-9 / eCharge
        scaling_factor = target_nPart/simParticles
        self.dosemap = dosemap*scaling_factor
        self.acChargenC = acChargenC
        self.nParticles = simParticles
        self.sim_dose = True

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

    def get_depth_slice(self, depthmm):

        # plt.contourf(xi*0.35, yi*0.405, di*scaling_factor, 100, cmap='jet' )

        bindepth = np.shape(self.dosemap[2]) - int(depthmm/self.scale)
        self.dosemap = self.dosemap[:, :, bindepth]


filepath = '/home/robertsoncl/topas/DoseAtFilm1.csv'

sdf = simDoseProfile(filepath, 1000000, 2, 0.35)
sdf.get_sim_strips(1, 0, 0)
sdf.get1DSuperGaussian(power=1)
sdf.plot_xy_fits()
