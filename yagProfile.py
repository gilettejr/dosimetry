#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:14:53 2023

@author: robertsoncl
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import matplotlib.colors as clr
from astropy.modeling import models, fitting
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from beamProfile import beamProfile


class yagProfile(beamProfile):
    def __init__(self, yagFile, scale=0.036):
        realData = loadmat(yagFile)


# need to double check that this is correct
        self.scale = scale

        self.x_extent = np.shape(realData['image_new'])[1] * scale / 2
        self.y_extent = np.shape(realData['image_new'])[0] * scale / 2
        self.realData = realData['image_new']
        self.pxheight = np.shape(realData['image_new'])[0]
        self.pxwidth = np.shape(realData['image_new'])[1]
        self.mmHeight = self.pxheight*scale
        self.mmWidth = self.pxwidth*scale
        self.scale = scale
        self.yagfile = yagFile
        self.dosemap = self.realData
        super(yagProfile, self).__init__()
        self.yag = True

    def get_yag_strips(self, slicewidth_mm, xpos_mm, ypos_mm):
        # self.trimEdges(5)
        scale = self.scale

        xpos,  ypos, width = (xpos_mm+self.x_extent) / \
            scale, (ypos_mm+self.y_extent)/scale, slicewidth_mm/scale
        ypos = self.pxheight-ypos
        xpos = self.pxwidth-xpos
        x_slice = self.dosemap[int(ypos-width/2):int(ypos+width/2)]
        y_slice = np.flip(np.rot90(self.dosemap)[
            int(xpos-width/2):int(xpos+width/2)])
        x_slice = np.mean(x_slice, axis=0)
        y_slice = np.mean(y_slice, axis=0)
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.xpos_mm = xpos_mm
        self.ypos_mm = ypos_mm
        self.slicewidth_mm = slicewidth_mm

    def get_yag_region(self, xpos_mm, ypos_mm, regionx, regiony):
        scale = self.scale
        xpos,  ypos, regionxpx, regionypx = (xpos_mm+self.x_extent) / \
            scale, (ypos_mm+self.y_extent)/scale, regionx/scale, regiony/scale
        ypos = self.pxheight-ypos
        xpos = self.pxwidth-xpos
        self.region_image = self.realData[int(
            ypos-regionypx/2):int(ypos+regionypx/2)]
        self.region_image = np.flip(np.rot90(self.region_image)[
                                    int(xpos-regionxpx/2):int(xpos+regiony/2)])

        print('Mean Dose Across Selected region:')
        print(np.mean(self.region_image))
        print('Standard Deviation Across Selected region:')
        print(np.std(self.region_image))
        return np.mean(self.region_image), np.std(self.region_image)
        # print(skew(region_dosemap))
