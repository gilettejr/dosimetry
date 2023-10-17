from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
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


class flatYagProfile:
    def __init__(self, filename, scale=0.072):

        image_data = loadmat(filename)
        self.dosemap = image_data['image']
        self.scale = scale

        def px_to_mm(px):
            return px * scale

        def mm_to_px(mm):
            return mm/scale

        def set_geometry():
            self.height, self.width = np.shape(self.dosemap)[
                0], np.shape(self.dosemap)[1]
            self.mmWidth, self.mmHeight = px_to_mm(
                self.width), px_to_mm(self.height)

        self.set_geometry = set_geometry
        self.px_to_mm = px_to_mm
        self.mm_to_px = mm_to_px
        set_geometry()

    def show_image(self):
        plt.imshow(
            self.dosemap, cmap="jet", interpolation="nearest", vmin=0, vmax=100
        )
        plt.xlabel("X [pixels]")
        plt.ylabel("Y [pixels]")

    def get2DSuperGaussian(self, binSize=2, plot=False):
        # super gaussian function
        # gaussan function with exponent to produce flat top
        def super_g(xy, amplitude, x_mean, y_mean, x_stddev, y_stddev, power):
            # print(amplitude, x_mean, y_mean, x_stddev, y_stddev, power)
            """Two dimensional Gaussian function"""
            x, y = xy
            xstd2 = x_stddev**2
            ystd2 = y_stddev**2
            xdiff = x - x_mean
            ydiff = y - y_mean
            xdiff2 = xdiff**2
            ydiff2 = ydiff**2
            exponent = np.divide(xdiff2, 2*xstd2)+np.divide(ydiff2, 2*ystd2)
            g = amplitude*np.exp(-(exponent**power))
            return g.ravel()

        def get_normalised_sig(muX, muY, sigX, sigY, peak_dose):
            binSize = (np.mean([sigX, sigY])/10)
            max_dose_std = np.round(np.std(self.dosemap[int(muX-binSize/2):int(
                muX+binSize/2), int(muY-binSize/2):int(muY+binSize/2)]), 2)
            max_dose_std = max_dose_std/peak_dose
            return max_dose_std

        # set up coordinates from original image size
        y, x = np.mgrid[:self.height, :self.width]
        # set reasonable bounds to prevent fitting divergence
        # these parameters should be okay for films
        bounds = np.transpose([[-np.inf, np.inf], [100, 400], [
            100, 400], [0, 500], [0, 500], [0, 10]])
        # carry out fit to supergaussian, retrieving parameters and
        # covariance matrix
        popt, pcov = curve_fit(
            super_g, (x, y), self.dosemap.ravel(), bounds=bounds, p0=(np.mean(self.dosemap), int(np.mean(x)), int(np.mean(y)), 100, 100, 2))
        # retrieve fit data
        fit_data = super_g((x, y), *popt)
        # uncollapse fit data
        super_gaussian_dosemap = fit_data.reshape(self.height, self.width)
        # retrieve errors from covariance matrix (these are likely small)
        perr = np.sqrt(np.diag(pcov))
        # I'll have to edit these
        peak_loc = np.unravel_index(
            super_gaussian_dosemap.argmax(), super_gaussian_dosemap.shape)
        # max_dose=np.round(gaussian_dosemap[peak_loc],1)
        max_dose = np.round(np.mean(self.dosemap[int(popt[1]-binSize/2):int(
            popt[1]+binSize/2), int(popt[2]-binSize/2):int(popt[2]+binSize/2)]), 2)
        max_dose_std = np.round(np.std(self.dosemap[int(popt[1]-binSize/2):int(
            popt[1]+binSize/2), int(popt[2]-binSize/2):int(popt[2]+binSize/2)]), 2)
        max_dose_fit = np.round(np.mean(super_gaussian_dosemap[int(peak_loc[0]-binSize/2):int(
            peak_loc[0]+binSize/2), int(peak_loc[1]-binSize/2):int(peak_loc[1]+binSize/2)]), 2)
        extent = [0, self.mmWidth, 0, self.mmHeight]
        # plot data, fit and residuals as required
        if plot:

            plt.rc("axes", labelsize=20)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            fig, ax = plt.subplots(1, 3, figsize=(20, 8))
            plt.rc('font', size=20)
            ax[0].imshow(
                self.dosemap,
                cmap="jet",
                vmin=0,
                vmax=max_dose_fit,
                interpolation="nearest", extent=extent
            )
            ax[0].set_ylabel("y [mm]")
            ax[0].set_xlabel("x [mm]")
            ax[0].set_title(self.stripped_filename+" Data")
            # ax[1].colorbar()
            ax[1].imshow(
                super_gaussian_dosemap,
                cmap="jet",
                interpolation="nearest", extent=extent
            )
            ax[1].set_xlabel("x [mm]")
            ax[1].set_title("Super Gaussian Fit")

            im = ax[2].imshow(
                self.dosemap-super_gaussian_dosemap,
                cmap="jet",
                interpolation="nearest",
                extent=extent,
                vmin=0,
                vmax=max_dose_fit,

            )
            ax[2].set_xlabel("x [mm]")
            ax[2].set_title("Residuals")

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Dose [Gy]', )
        '''Noise reduction and smoothing by fitting the data to a 2D gaussian model.
        It is based on the assumption that the beam has a gaussian distrimution in the transverse plane, and uses the Levenberg-Marquardt fitting algorithm incorporating least squares statistics.
        Based on the resulting gaussian dosemap, the location of the peak dose, and the peak dose averaged over binSize x binSize are found.
        The standard deviations corresponding to the beam size in the x- and y-planes are also computed.
        '''
        muX_px = popt[1]
        muY_px = popt[2]
        sigX_px = popt[3]
        sigY_px = popt[4]
        self.super_gaussian_dosemap = fit_data
        self.muX = np.round(popt[1], 2)
        self.muY = np.round(popt[2], 2)
        self.sigX = np.round(popt[3], 2)
        self.sigY = np.round(popt[4], 2)
        self.power = np.round(popt[5], 2)
        self.peak_dose = max_dose
        self.peak_dose_loc = peak_loc
        self.normalised_sig = get_normalised_sig(
            muX_px, muY_px, sigX_px, sigY_px, self.peak_dose)
        return pd.DataFrame({'muX': self.muX, 'muY': self.muY, 'sigX': self.sigX, 'sigY': self.sigY, 'power': popt[5], 'peak dose': max_dose, 'norm sig dose': self.normalised_sig}, index=[self.stripped_filename])

    def get1DSuperGaussian(self, binsize=2, plot=False):
        def super_g(x, amplitude, mean, stddev, power):
            # print(amplitude, x_mean, y_mean, x_stddev, y_stddev, power)
            """Two dimensional Gaussian function"""
            std2 = stddev**2
            xdiff = x - mean
            xdiff2 = xdiff**2
            exponent = np.divide(xdiff2, 2*std2)
            g = amplitude*np.exp(-(exponent**power))
            return g
        # set up coordinates from original image size

        x = np.arange(0, self.width, step=1)

        # set reasonable bounds to prevent fitting divergence

        # carry out fit to supergaussian, retrieving parameters and
        # covariance matrix
        popt, pcov = curve_fit(
            super_g, x, self.x_slice, bounds=[0, self.width], p0=(np.mean(self.x_slice), int(np.mean(x)), 100, 3))
        # retrieve fit data

        super_gaussian_dosemap = super_g(x, *popt)
        super_gaussian_dosemap = super_g(x, 40, 250, 120, 10)
        # uncollapse fit data
        # retrieve errors from covariance matrix (these are likely small)
        perr = np.sqrt(np.diag(pcov))
        # I'll have to edit these
        # plot data, fit and residuals as required
        extent = [0, self.mmWidth]
        if plot:

            plt.rc("axes", labelsize=20)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            fig, ax = plt.subplots(1, 3, figsize=(20, 8))
            plt.rc('font', size=20)
            ax[0].plot(
                x*self.mmWidth/self.width,
                self.x_slice
            )
            ax[0].set_ylabel("Intensity")
            ax[0].set_xlabel("x [mm]")
            ax[0].set_title("Data")
            # ax[1].colorbar()
            ax[1].plot(x*self.mmWidth/self.width,
                       super_gaussian_dosemap,
                       )
            ax[1].set_xlabel("x [mm]")
            ax[1].set_title("Super Gaussian Fit")

            ax[2].plot(x*self.mmWidth/self.width,
                       self.x_slice-super_gaussian_dosemap,
                       )
            ax[2].set_xlabel("x [mm]")
            ax[2].set_title("Residuals")

    def trim(self, lrup_cropPercent=(0, 0, 0, 0), keepAspectRatio=True):
        '''Crops the film by cropPercent on each side and returns a new object. By default it maintains the original aspect ratio, but this can be changed by setting the keepAspectRatio to False. In this case it is cropped square by maximum cropPercent'''
        if keepAspectRatio:

            left_lim = int(self.width*lrup_cropPercent[0]/100)
            right_lim = int(self.width*(1-lrup_cropPercent[1]/100))
            upper_lim = int(self.height*lrup_cropPercent[2]/100)
            lower_lim = int(self.height*(1-lrup_cropPercent[3]/100))

        self.dosemap = self.dosemap[upper_lim:lower_lim,
                                    :][:, left_lim:right_lim]

    def get_strips(self, width_mm, xpos_mm=False, ypos_mm=False):
        if xpos_mm is False:
            centre = self.find_centre(self.cv_img)
            centre_mm = self.px_to_mm(centre)
            xpos_mm, ypos_mm = centre_mm[0], centre_mm[1]

        xpos, ypos, width = self.mm_to_px(xpos_mm), self.mm_to_px(
            ypos_mm), self.mm_to_px(width_mm)

        self.x_dosemap = self.dosemap[int(ypos -
                                      width/2):int(ypos+width/2), :][:, 0:self.width]
        self.y_dosemap = self.dosemap[0:self.height,
                                      :][:, int(xpos-width/2):int(xpos+width/2)]
        self.x_slice = np.mean(self.x_dosemap, axis=0)
        self.y_slice = np.mean(self.y_dosemap, axis=1)
        # print(xg)
        self.set_geometry()
        self.xpos_mm = xpos_mm
        self.ypos_mm = ypos_mm
        self.width_mm = width_mm

    def get_stats(self):
        x_dosemap = self.x_dosemap
        std_dev = np.std(x_dosemap)
        mean = np.mean(x_dosemap)
        print(mean)
        print(std_dev)
        return mean, std_dev

    def plot_xy_hists(self, lim=4):
        # print(self.x_dosemap[0])

        x_slice = self.x_slice
        y_slice = self.y_slice

        x_range = np.linspace(0, self.mmWidth, len(x_slice))
        y_range = np.linspace(0, self.mmHeight, len(y_slice))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        plt.rc("axes", labelsize=20)
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        x_patch = Rectangle((0, self.ypos_mm-self.width_mm/2),
                            width=self.mmWidth, height=self.width_mm, fc='None', ec='k', ls='-')

        y_patch = Rectangle((self.xpos_mm-self.width_mm/2, 0),
                            width=self.width_mm, height=self.mmHeight, fc='None', ec='k', ls='-')
        ax[0].imshow(self.dosemap, extent=(
            0, self.mmWidth, self.mmHeight, 0),)
        ax[0].add_patch(x_patch)
        ax[0].add_patch(y_patch)
        ax[0].set_xlabel('x [mm]')
        ax[0].set_ylabel('y [mm]')
        ax[1].plot(x_range, x_slice, label='Dose along X')
        ax[1].set_xlabel('x [mm]')
        ax[1].set_ylabel('Intensity [Arb. units]')
        #ax[1].set_xlim((3, 30))
        ax[1].set_ylim(bottom=0, top=lim)
        #ax[0].set_ylim((0, 4))
        ax[2].plot(y_range, y_slice, label='Dose along Y')
        ax[2].set_xlabel('y [mm]')
        #ax[2].set_ylabel('Dose [Gy]')
        #ax[2].set_xlim((10, 37))
        ax[2].set_ylim(bottom=0, top=lim)
        # print(self.y_dosemap)
