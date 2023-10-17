from filmDosimetryTools import RadioChromicFilm
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from astropy.modeling import models, fitting
# import calibration_curve as cc
import os
from scipy.optimize import curve_fit
from collections import OrderedDict
from operator import getitem
from scipy import ndimage, misc
from matplotlib.widgets import SpanSelector
import math
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import matplotlib.colors as clr
import cv2
from scipy.stats import kurtosis, skew
from beamProfile import beamProfile

# mm per inch


class filmProfile(beamProfile):
    def __init__(self, filepath, a, b, c, RGB_max=65535, dpi=300, mmpi=25.4):
        def get_rgb_d(image):
            input_image = image.convert("RGB")
            input_imarray = np.array(input_image)

            X_d = np.array(input_imarray / 65535)

            X_d_transpose = np.transpose(X_d)

            rX_d, gX_d, bX_d = X_d_transpose
            return rX_d, gX_d, bX_d

        def getDoseMap(a, b, c, X_d, channel='green'):
            dose = np.fliplr(np.rot90((a - c * X_d) / (X_d - b), 3))
            return dose
        input_image = Image.open(filepath).convert("RGB")
        r, g, bl = get_rgb_d(input_image)

        self.image = input_image
        self.scale = mmpi/dpi

        self.dosemap = getDoseMap(a, b, c, g)
        self.a, self.b, self.c = a, b, c
        self.pxwidth, self.pxheight = self.image.size  # film width and height in dots
        self.mmWidth = self.pxwidth*mmpi/dpi  # film width and height in mm
        self.mmHeight = self.pxheight*mmpi/dpi

        self.getDoseMap = getDoseMap
        self.get_rgb_d = get_rgb_d
        super(filmProfile, self).__init__()
        self.film = True

    def get_dose_strips(self, slicewidth_mm, xpos_mm, ypos_mm):
        a, b, c = self.a, self.b, self.c
        # self.trimEdges(5)
        xpos_mm = xpos_mm+self.mmWidth/2
        ypos_mm = -ypos_mm
        ypos_mm = ypos_mm+self.mmHeight/2
        xpos, ypos, slicewidth = self.mm_to_px(xpos_mm), self.mm_to_px(
            ypos_mm), self.mm_to_px(slicewidth_mm)

        xslice_image = self.image.crop(
            (0, ypos-slicewidth/2, self.pxwidth, ypos+slicewidth/2))
        yslice_image = self.image.crop(
            (xpos-slicewidth/2, 0, xpos+slicewidth/2, self.pxheight))
        xr, xg, xb = self.get_rgb_d(xslice_image)
        yr, yg, yb = self.get_rgb_d(yslice_image)

        # print(xg)
        x_dosemap = self.getDoseMap(a, b, c, xg)
        y_dosemap = self.getDoseMap(a, b, c, yg)

        self.x_slice = np.mean(x_dosemap, axis=0)
        self.y_slice = np.mean(y_dosemap, axis=1)
        self.xpos_mm = xpos_mm-self.mmWidth/2
        ypos_mm = ypos_mm-self.mmHeight/2
        self.ypos_mm = -ypos_mm

        self.film = True

        self.slicewidth_mm = slicewidth_mm

    def get_dose_region(self, xpos_mm, ypos_mm, regionx_mm, regiony_mm):
        a, b, c = self.a, self.b, self.c
        xpos_mm = xpos_mm+self.mmWidth/2
        ypos_mm = -ypos_mm
        xpos, ypos, regionxpx, regionypx = self.mm_to_px(xpos_mm), self.mm_to_px(
            ypos_mm), self.mm_to_px(regionx_mm), self.mm_to_px(regiony_mm)
        self.region_image = self.image.crop((
            xpos-regionxpx/2, ypos-regionypx/2, xpos+regionxpx/2, ypos+regionypx/2))
        rr, rg, rb = self.get_rgb_d(self.region_image)
        rt, gt, bt = self.get_rgb_d(self.image)
        region_dosemap = self.getDoseMap(a, b, c, rg)
        # plt.imshow(region_dosemap)
        dosemap = self.getDoseMap(a, b, c, gt)
        print('Mean Dose Across Selected region:')
        print(np.mean(region_dosemap))
        print('Standard Deviation Across Selected region:')
        print(np.std(region_dosemap))
        return np.mean(region_dosemap), np.std(region_dosemap)
        # print(skew(region_dosemap))

    def getDensityStats(self, a, b, r):

        x_slice = np.mean(self.x_dosemap, axis=0)
        y_slice = np.mean(self.y_dosemap, axis=1)
        x_range = np.linspace(0, self.mmWidth, len(x_slice))
        y_range = np.linspace(0, self.mmHeight, len(y_slice))
        bins = np.arange(1, len(x_range), 1)
        x_slice = x_slice*1000
        x_slice = np.rint(x_slice)
        x_slice = x_slice.astype(int)

        y_slice = y_slice*1000
        y_slice = np.rint(y_slice)
        y_slice = y_slice.astype(int)
        for i in range(len(x_slice)):

            if x_slice[i] < 0:
                x_slice[i] = 0
        for i in range(len(y_slice)):
            if y_slice[i] < 0:
                y_slice[i] = 0
        xdata = np.repeat(x_range, x_slice)
        ydata = np.repeat(y_range, y_slice)
        for i in range(len(xdata)):
            if xdata[i] < a-r or xdata[i] > a+r:
                xdata[i] = np.nan
        meanx = np.nanquantile(xdata, 0.50, method='interpolated_inverted_cdf')
        sigmax = meanx-np.nanquantile(xdata, 0.159,
                                      method='interpolated_inverted_cdf')
        for i in range(len(xdata)):
            if xdata[i] < meanx-sigmax*2 or xdata[i] > meanx+sigmax*2:
                xdata[i] = np.nan
        print(sigmax)
        print(kurtosis(xdata, nan_policy='omit'))

        for i in range(len(ydata)):
            if ydata[i] < b-r or ydata[i] > b+r:
                ydata[i] = np.nan
        meany = np.nanquantile(ydata, 0.50, method='interpolated_inverted_cdf')
        sigmay = meany-np.nanquantile(ydata, 0.159,
                                      method='interpolated_inverted_cdf')
        for i in range(len(ydata)):
            if ydata[i] < meany-sigmay*2 or ydata[i] > meany+sigmay*2:
                ydata[i] = np.nan
        print(sigmay)
        print(kurtosis(ydata, nan_policy='omit'))
        # plt.figure()
        # plt.hist(xdata, bins=100)

        # print(self.y_dosemap)
