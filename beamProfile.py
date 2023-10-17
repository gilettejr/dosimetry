#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:43:05 2023

@author: robertsoncl
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class beamProfile:
    def __init__(self):
        def mm_to_px(mm):
            return mm/self.scale

        def px_to_mm(px):
            return px*self.scale

        self.mm_to_px = mm_to_px
        self.px_to_mm = px_to_mm
        self.film = False
        self.yag = False
        self.sim_intensity = False
        self.sim_dose = False

    def get1DSuperGaussian(self, power=1):
        # super gaussian function
        # gaussan function with exponent to produce flat top
        def super_g(x, amplitude, mean, stddev, power):
            # print(amplitude, x_mean, y_mean, x_stddev, y_stddev, power)
            """Two dimensional Gaussian function"""

            std2 = stddev**2

            diff = x - mean

            diff2 = diff**2

            exponent = np.divide(diff2, 2*std2)
            g = amplitude*np.exp(-(exponent**power))
            return g

        def gaussian(x, amplitude, mean, stddev, power):
            # print(amplitude, x_mean, y_mean, x_stddev, y_stddev, power)
            """Two dimensional Gaussian function"""

            std2 = stddev**2

            diff = x - mean

            diff2 = diff**2

            exponent = np.divide(diff2, 2*std2)
            g = amplitude*np.exp(-(exponent))
            return g

       # print(self.x_dosemap)
        x_slice = self.x_slice
        y_slice = self.y_slice
        # set up coordinates from original image size
        y, x = np.arange(self.pxheight), np.arange(self.pxwidth)
        # set reasonable bounds to prevent fitting divergence
        # these parameters should be okay for films
        if self.film:
            bounds = np.transpose([[-np.inf, np.inf], [100, 400],
                                   [0, 500], [0, 10]])
        else:
            bounds = np.transpose([[-np.inf, np.inf], [-np.inf, np.inf],
                                   [-np.inf, np.inf], [0, 10]])

        # carry out fit to supergaussian, retrieving parameters and
        # covariance matrix

        if power == 1:
            super_g = gaussian
            powerX = 1
            powerY = 1
       # print((np.mean(x_slice), int(np.mean(x)), 100, 1))
        xpopt, xpcov = curve_fit(
            super_g, x, x_slice, bounds=bounds, p0=(np.mean(x_slice), int(np.mean(x)), 100, 1))
        # retrieve fit data
        xfit = super_g(x, *xpopt)

        ypopt, ypcov = curve_fit(
            super_g, y, y_slice, bounds=bounds, p0=(np.mean(y_slice), int(np.mean(y)), 100, 1))
        # retrieve fit data
        yfit = super_g(y, *ypopt)
        if power != 1:
            powerX = xpopt[3]
            powerY = ypopt[3]

        muX_px = xpopt[1]
        muY_px = ypopt[1]
        sigX_px = xpopt[2]
        sigY_px = ypopt[2]

        self.muX = np.round(self.px_to_mm(muX_px), 2)-self.mmWidth/2
        self.muY = -(np.round(self.px_to_mm(muY_px), 2)-self.mmHeight/2)
        if self.yag or self.sim_intensity:
            self.muY = -self.muY
        self.sigX = np.abs(np.round(self.px_to_mm(sigX_px), 2))
        self.sigY = np.abs(np.round(self.px_to_mm(sigY_px), 2))
        self.powerX = np.round(powerX, 2)
        self.powerY = np.round(powerY, 2)
        self.xfit = xfit
        self.yfit = yfit

        return pd.DataFrame({'muX': self.muX, 'muY': self.muY, 'sigX': self.sigX, 'sigY': self.sigY, 'powerX': self.powerX, 'powerY': self.powerY}, index=[0])

    def get2DSuperGaussian(self, power=1):
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

        def gaussian(xy, amplitude, x_mean, y_mean, x_stddev, y_stddev, power):
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
            g = amplitude*np.exp(-exponent)
            return g.ravel()
        dosemap = self.dosemap
        # set up coordinates from original image size
        y, x = np.mgrid[:self.pxheight, :self.pxwidth]
        # set reasonable bounds to prevent fitting divergence
        # these parameters should be okay for films
        if self.film:
            bounds = np.transpose([[-np.inf, np.inf], [100, 400], [
                100, 400], [0, 500], [0, 500], [0, 10]])
        elif self.yag:
            bounds = np.transpose([[-np.inf, np.inf], [-np.inf, np.inf], [
                -np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [0, 10]])

        # carry out fit to supergaussian, retrieving parameters and
        # covariance matrix
        if power == 1:
            super_g = gaussian
        popt, pcov = curve_fit(
            super_g, (x, y), dosemap.ravel(), bounds=bounds, p0=(np.mean(dosemap), int(np.mean(x)), int(np.mean(y)), 100, 100, 1))
        if power != 1:
            power = np.round(popt[5], 2)
        # retrieve fit data
        fit_data = super_g((x, y), *popt)
        # uncollapse fit data
        super_gaussian_dosemap = fit_data.reshape(self.pxheight, self.pxwidth)
        # retrieve errors from covariance matrix (these are likely small)
        perr = np.sqrt(np.diag(pcov))

        # plot data, fit and residuals as required

        muX = round(self.px_to_mm(popt[1]), 2)-self.mmWidth/2
        muY = -(round(self.px_to_mm(popt[2]), 2)-self.mmHeight/2)
        sigX = round(self.px_to_mm(popt[3]), 2)
        sigY = round(self.px_to_mm(popt[4]), 2)
        self.super_gaussian_dosemap = fit_data
        self.xyfit = fit_data
        return pd.DataFrame({'muX': muX, 'muY': muY, 'sigX': sigX, 'sigY': sigY, 'power': popt[5]}, index=[0])

    def plot_xy_fits(self, showFit=True):
        # print(self.x_dosemap[0])
        if self.film or self.sim_dose:
            quantity = 'Dose'
            unit = '[Gy]'

        else:
            quantity = 'Intensity'
            unit = '[arb.]'

        x_slice = self.x_slice
        y_slice = self.y_slice

        x_range = np.linspace(-self.mmWidth/2, self.mmWidth/2, len(x_slice))
        y_range = (np.linspace(-self.mmHeight/2,
                               self.mmHeight/2, len(y_slice)))
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        plt.rc("axes", labelsize=12)
        plt.rc("xtick", labelsize=12)
        plt.rc("ytick", labelsize=12)
        x_patch = Rectangle((-self.mmWidth/2, self.ypos_mm-self.slicewidth_mm/2),
                            width=self.mmWidth, height=self.slicewidth_mm, fc='None', ec='k', ls='-')

        y_patch = Rectangle((self.xpos_mm-self.slicewidth_mm/2, -self.mmHeight/2),
                            width=self.slicewidth_mm, height=self.mmHeight, fc='None', ec='k', ls='-')
        ax[0].imshow(self.dosemap, extent=(
            -self.mmWidth/2, self.mmWidth/2, -self.mmHeight/2, self.mmHeight/2), cmap='jet')
        ax[0].add_patch(x_patch)
        ax[0].add_patch(y_patch)
        ax[0].set_xlabel('x [mm]')
        ax[0].set_ylabel('y [mm]')
        ax[1].plot(x_range, x_slice, color='r',
                   label=quantity+' along x slice')
        ax[1].set_xlabel('x [mm]')
        ax[1].set_ylabel(quantity + ' '+unit)

        ax[1].grid(True)
        ax[1].legend()
        # ax[0].set_ylim((0, 4))
        ax[2].plot(y_range, y_slice, label=quantity +
                   ' along x slice', color='r')
        ax[2].set_xlabel('y [mm]')

        # ax[2].set_ylabel('Dose [Gy]')

        ax[2].grid(True)
        ax[2].legend()
        if showFit is True:
            ax[1].plot(x_range, self.xfit, label='Gaussian Fit',
                       linestyle='--', color='k')

            ax[1].text(x=self.muX, y=max(self.xfit)*1.2,
                       s='$\sigma_x$='+str(self.sigX)+'mm')
            ax[1].set_xlim(self.muX-3*self.sigX, self.muX+3*self.sigX)
            ax[1].set_ylim(bottom=0, top=max(self.xfit)*2)
            ax[2].plot(y_range, self.yfit, label='Gaussian Fit',
                       linestyle='--', color='k')
            ax[2].text(x=self.muY, y=max(self.xfit)*1.2,
                       s='$\sigma_y$='+str(self.sigY)+'mm')
            ax[2].set_xlim(self.muY-3*self.sigY, self.muY+3*self.sigY)
            ax[2].set_ylim(bottom=0, top=max(self.xfit)*2)
        else:
            ax[1].set_xlim([5, 32])
            ax[2].set_xlim([5, 32])
            ax[1].set_ylim([0, 35])
            ax[2].set_ylim([0, 35])
        ax[1].legend()
        ax[2].legend()
