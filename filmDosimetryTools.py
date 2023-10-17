#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:45:31 2023

@author: vilde

Description
-------------
Collection of help functions for film analysis, to be imported for specific analysis
"""
import sys
import numpy as np
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
import matplotlib.colors as clr
import cv2
mmpi = 25.4  # mm per inch

a_r = 9.53385895e-03
b_r = 4.40553270e-04
c_r = 3.71012295e+00

a_g = 1.36037090e-02
b_g = 1.48062223e-04
c_g = 5.72374622e+00

a_b = 2.22791213e-02
b_b = 3.04096810e-04
c_b = 1.43716055e+01


class RadioChromicFilm:
    # class associating all properties of a radiochromic film
    def __init__(self, filepath, charge=0, RGB_max=65535, dpi=300, back_vals=(16.699885656571432, 16.739994028249626, 17.192428103235425)):
        def get_rgb_d_from_file(filename):
            print(filename)
            input_image = Image.open(filename).convert("RGB")
            input_imarray = np.array(input_image)

            X_d = np.array(input_imarray / 65535)

            X_d_transpose = np.transpose(X_d)

            rX_d, gX_d, bX_d = X_d_transpose
            return rX_d, gX_d, bX_d

        def get_rgb_d_from_arg(image):
            input_image = image.convert("RGB")
            input_imarray = np.array(input_image)

            X_d = np.array(input_imarray / 65535)

            X_d_transpose = np.transpose(X_d)

            rX_d, gX_d, bX_d = X_d_transpose
            return rX_d, gX_d, bX_d

        def set_geometry():
            self.width, self.height = self.image.size  # film width and height in dots
            self.mmWidth = self.width*mmpi/dpi  # film width and height in mm
            self.mmHeight = self.height*mmpi/dpi
            self.aspectRatio = self.height/self.width

        def mm_to_px(mm):
            return mm*dpi/mmpi

        def px_to_mm(px):
            return px*mmpi/dpi

        def getDoseMap(a, b, c, X_d, channel='green'):
            dose = np.fliplr(np.rot90((a - c * X_d) / (X_d - b), 3))
            return dose

        def trimEdges(cropPercent=0, keepAspectRatio=True):
            '''Crops the film by cropPercent on each side and returns a new object. By default it maintains the original aspect ratio, but this can be changed by setting the keepAspectRatio to False. In this case it is cropped square by maximum cropPercent'''
            if keepAspectRatio:

                left_lim = int(self.width*cropPercent/100)
                right_lim = int(self.width*(1-cropPercent/100))
                upper_lim = int(self.height*cropPercent/100)
                lower_lim = int(self.height*(1-cropPercent/100))

            else:
                upper_lim = int(self.width*(cropPercent*self.aspectRatio/100))
                left_lim = int(self.width*cropPercent/100)
                lower_lim = int(
                    self.width*(1-cropPercent*self.aspectRatio/100))
                right_lim = int(self.width*(1-cropPercent/100))

            self.image = self.image.crop(
                (left_lim, upper_lim, right_lim, lower_lim))
            self.cv_img = self.cv_img[upper_lim:lower_lim, left_lim:right_lim]
            # self.__init__(cropped_image)
            self.set_geometry()
        rX_d, gX_d, bX_d = get_rgb_d_from_file(filepath)
        self.get_rgb_d_from_arg = get_rgb_d_from_arg
        self.dosemap = getDoseMap(a_g, b_g, c_g, gX_d)
        self.mm_to_px = mm_to_px
        self.px_to_mm = px_to_mm
        self.getDoseMap = getDoseMap
        self.image = Image.open(filepath)
        self.cv_img = cv2.imread(filepath, 0)
        self.RGB_max = RGB_max
        # retrieve_RGB()
        set_geometry()
        self.trimEdges = trimEdges
        # seperate film into rgb channels and return data values
        self.name = f'{charge}nC'
        self.dpi = dpi
        self.charge = charge

        self.AOI = None
        self.dosemap = None
        self.peak_dose_loc = None
        self.peak_dose = None
        self.gaussian_dosemap = None
        self.muX = None
        self.muY = None
        self.sigX = None
        self.sigY = None
        self.gaussian_profiles = None
        self.set_geometry = set_geometry
        self.stripped_filename = filepath[-12:-4]

    def selectAOI(self, textsize=20, saveFig=False, saveDir=os.getcwd()):
        def AOI_select_callback(eclick, erelease):
            """
            Callback loop for rectangle selection. *eclick* and *erelease* are the press and release events.
            """
            global x1, y1, x2, y2
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
            # newl='\n'
            plt.gca().set_title("Click and/or drag to select new AOI.\n Press 'c' to confirm selection, close and save figure.", fontsize=textsize)
            # plt.gca().texts.set_visible(False)

            text.set_text(
                rf"Selected AOI: {round(abs(x2-x1)*mmpi/self.dpi,2)} x {round(abs(y2-y1)*mmpi/self.dpi,2)} mm$^2$")
            text.set_x(min(x1, x2)+abs(x2-x1)/2)
            text.set_y(max(y1, y2)+30)

            print(f'Selected AOI coordinates: ({x1},{y1})-->({x2},{y2})')

        def toggle_selector(event):

            print('Selection confirmed.')
            if event.key == 'c':
                name = type(selector).__name__
                if selector.active:
                    print(f'{name} deactivated.')
                    selector.set_active(False)

                    self.AOI = [(x1, y1), (x2, y2)]
                    if saveFig:
                        plt.savefig(saveDir+f'/{self.name}_AOI.png', dpi=100)
                    plt.close()
                    print(
                        f'AOI={self.AOI} stored. Image saved as {self.name}_AOI.png in the current directory')

        fig = plt.figure()
        ax = fig.subplots()
        # ax=fig.add_subplot(111)
        plt.imshow(self.image)  # , extent=(0,self.width, 0, self.height))
        text = ax.text(0, 0, "", fontsize=textsize,
                       horizontalalignment='center', verticalalignment='top',)
        # title=plt.title("Select AOI", fontsize=textsize)
        x_tickpositions = np.arange(0, self.width, 50)
        x_ticklabels = np.round(x_tickpositions*mmpi/self.dpi)
        plt.xticks(x_tickpositions, x_ticklabels, fontsize=textsize)
        plt.xlabel('mm', fontsize=textsize)

        y_tickpositions = np.arange(0, self.height, 50)
        y_ticklabels = np.round(np.flip(y_tickpositions)*mmpi/self.dpi)
        plt.yticks(y_tickpositions, y_ticklabels, fontsize=textsize)
        plt.ylabel('mm', fontsize=textsize)

        props = dict(facecolor=None, edgecolor='red',
                     alpha=1, fill=False, linewidth=3)

        ax.set_title("Click and drag to select AOI.", fontsize=textsize)
        selector = RectangleSelector(ax, AOI_select_callback, rectprops=props, useblit=False, button=[
                                     1, 3], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        fig.canvas.mpl_connect('key_press_event', toggle_selector)

        plt.show()

        return [(x1, y1), (x2, y2)]

    def cropToAOI(self):
        if self.AOI == None:
            print('AOI not defined. Run selectAOI first.')
        else:
            cropped_image = self.image.crop(
                (self.AOI[0][0], self.AOI[0][1], self.AOI[1][0], self.AOI[1][1]))
            return RadioChromicFilm(cropped_image)

    def filterImage(self):
        filtered_image = self.image

        '''???'''
        return RadioChromicFilm(filtered_image)

    def get2DGaussian(self, binSize=2, plot=False):
        '''Noise reduction and smoothing by fitting the data to a 2D gaussian model.
        It is based on the assumption that the beam has a gaussian distrimution in the transverse plane, and uses the Levenberg-Marquardt fitting algorithm incorporating least squares statistics.
        Based on the resulting gaussian dosemap, the location of the peak dose, and the peak dose averaged over binSize x binSize are found.
        The standard deviations corresponding to the beam size in the x- and y-planes are also computed.
        '''
        # peak_loc=np.argmax(self.dosemap)
        y, x = np.mgrid[:self.height, :self.width]
        gaussian2D = models.Gaussian2D(theta=0)
        gaussian2D.theta.fixed = True  # generate a 2D gaussian fitting model
        # Levenberg-Marquardt fitting algorithm using least squares statistics
        fitting_algorithm = fitting.LevMarLSQFitter()
        # fit the data to a 2D gaussian using the fitting algorithm
        fitted_data = fitting_algorithm(gaussian2D, x, y, self.dosemap)
        print(self.dosemap)
        gaussian_dosemap = fitted_data(x, y)
        dimY, dimX = np.shape(gaussian_dosemap)
        peak_loc = np.unravel_index(
            gaussian_dosemap.argmax(), gaussian_dosemap.shape)
        # max_dose=np.round(gaussian_dosemap[peak_loc],1)
        max_dose = np.round(np.mean(gaussian_dosemap[int(peak_loc[0]-binSize/2):int(
            peak_loc[0]+binSize/2), int(peak_loc[1]-binSize/2):int(peak_loc[1]+binSize/2)]), 2)

        if plot:
            plt.figure(figsize=(8, 2.5))
            plt.subplot(1, 3, 1)
            plt.imshow(self.dosemap, cmap="jet",
                       interpolation="nearest")
            plt.title("Data")
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(gaussian_dosemap, cmap="jet", interpolation="nearest")
            plt.plot(peak_loc[1], peak_loc[0], 'o', markeredgecolor='blue',
                     markerfacecolor='none')  # mark the center/peak
            # lines defining the edges for max dose evaluations
            plt.axvline(x=int(peak_loc[1]-binSize/2),
                        ymin=0, ymax=dimY, color='blue')
            plt.axvline(x=int(peak_loc[1]+binSize/2),
                        ymin=0, ymax=dimY, color='blue')
            plt.axhline(y=int(peak_loc[0]-binSize/2),
                        xmin=0, xmax=dimX, color='blue')
            plt.axhline(y=int(peak_loc[0]+binSize/2),
                        xmin=0, xmax=dimX, color='blue')
            plt.title("Gaussian Fit - Max Dose = "+f'{max_dose}' + ' Gy')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.imshow(self.dosemap - gaussian_dosemap, cmap="jet",
                       interpolation="nearest", vmin=-1, vmax=1)
            plt.title("Residual")
            plt.colorbar()

        self.gaussian_dosemap = gaussian_dosemap
        self.muX = np.round(fitted_data.x_mean.value*mmpi/self.dpi, 2)
        self.muY = np.round(fitted_data.y_mean.value*mmpi/self.dpi, 2)
        self.sigX = np.round(fitted_data.x_stddev.value*mmpi/self.dpi, 2)
        self.sigY = np.round(fitted_data.y_stddev.value*mmpi/self.dpi, 2)
        self.peak_dose = max_dose
        self.peak_dose_loc = peak_loc
    # fit data to generalised Gaussian to order n
    # n > 1 represents a flat topped Gaussian
    # n=1 for regular Gaussian
    # I wouldn't use this unless flattened beam is expected
    # keeps badly fitting single scattered beam to n<1  for some reason

    def show_dose(self):
        extent = [0, self.mmWidth, 0, self.mmHeight]
        plt.rc("axes", labelsize=20)
        plt.rc("xtick", labelsize=20)
        plt.rc("ytick", labelsize=20)
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        plt.rc('font', size=20)
        im = ax.imshow(
            self.dosemap,
            cmap="jet",
            interpolation="nearest", extent=extent,

        )
        ax.set_ylabel("y [mm]")
        ax.set_xlabel("x [mm]")
        #ax.set_title(self.stripped_filename+" Data")
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Dose [Gy]', )

    def get1DDoseProfile(self, plot=True, textsize=20):
        # dosemap = np.rot90(np.squeeze(dosedata.data['Sum']))# indexing row,col. want x horizontal and y vertical

        Dmax_locY = self.peak_dose_loc[0]
        Dmax_locX = self.peak_dose_loc[1]
        x_profile = self.dosemap[Dmax_locY, :]
        x_gauss_profile = self.gaussian_dosemap[Dmax_locY, :]
        y_profile = self.dosemap[:, Dmax_locX]
        y_gauss_profile = self.gaussian_dosemap[:, Dmax_locX]
        x = np.arange(-self.peak_dose_loc[1], len(x_profile) -
                      self.peak_dose_loc[1])*mmpi/self.dpi
        y = np.arange(-self.peak_dose_loc[0], len(y_profile) -
                      self.peak_dose_loc[0])*mmpi/self.dpi

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Transverse dose profiles')

            ax1.plot(x, x_profile, label='Measured profile', linewidth=3)
            ax1.plot(x, x_gauss_profile, label='Gaussian fit', linewidth=3)
            ax1.axvline(-self.sigX, color='red', linestyle='--', linewidth=3)
            ax1.axvline(self.sigX, color='red', linestyle='--', linewidth=3)
            ax1.text(-self.sigX*1.5, np.max(x_profile) /
                     2, r"-$\sigma_x$", fontsize=textsize)
            ax1.text(self.sigX*1.2, np.max(x_profile) /
                     2, r'$\sigma_x$', fontsize=textsize)

            # ax.annotate(r'-$\sigma$', xy=(-sig, np.max(Dy0*scaling_factor)/2), xycoords='data',            xytext=(0.01, .99), textcoords='axes fraction',            va='top', ha='left',            arrowprops=dict(facecolor='black', shrink=0.05))
            # plt.xticks(list(plt.xticks()[0]) + [-sig,sig], list(map(str, plt.xticks()[0]))+[r'-$\sigma$',r'$\sigma$'] )
            ax1.legend(fontsize=textsize)
            ax1.set_title(
                rf"{self.name} x-profile - $2\sigma_x$={2*self.sigX} mm", fontsize=textsize)
            ax1.set_xlabel("X (mm)", fontsize=textsize)
            ax1.set_ylabel("Dose (Gy)", fontsize=textsize)

            ax2.plot(y, y_profile, label='Measured profile', linewidth=3)
            ax2.plot(y, y_gauss_profile, label='Gaussian fit', linewidth=3)
            ax2.axvline(-self.sigY, color='red', linestyle='--', linewidth=3)
            ax2.axvline(self.sigY, color='red', linestyle='--', linewidth=3)
            ax2.text(-self.sigY*1.5, np.max(y_profile) /
                     2, r"-$\sigma_y$", fontsize=textsize)
            ax2.text(self.sigY*1.2, np.max(y_profile) /
                     2, r'$\sigma_y$', fontsize=textsize)

            # ax.annotate(r'-$\sigma$', xy=(-sig, np.max(Dy0*scaling_factor)/2), xycoords='data',            xytext=(0.01, .99), textcoords='axes fraction',            va='top', ha='left',            arrowprops=dict(facecolor='black', shrink=0.05))
            # plt.xticks(list(plt.xticks()[0]) + [-sig,sig], list(map(str, plt.xticks()[0]))+[r'-$\sigma$',r'$\sigma$'] )
            ax2.legend(fontsize=textsize)
            ax2.set_title(
                rf"{self.name} y-profile - $2\sigma_y$={2*self.sigY} mm", fontsize=textsize)
            ax2.set_xlabel("Y (mm)", fontsize=textsize)
            ax2.set_ylabel("Dose (Gy)", fontsize=textsize)

            ax1.tick_params(axis='both', labelsize=textsize)
            ax2.tick_params(axis='both', labelsize=textsize)
            plt.savefig(f'{self.name}_profile.png', dpi=1000)

        self.gaussian_profiles = [x_profile, y_profile]


# def calc_correlation(film_frame):
#    def normalise_dose()


#frame = get_superG_params_folder(path_to_folder, [6, 17])
# print(frame['power'])
#print(frame['peak dose'])
#print(frame['norm sig dose'])
#correlation = np.corrcoef(frame['power'], frame['norm sig dose'])
# print(correlation)


# =============================================================================
#
#         def getBeamSize:
#
#             return self.sigmaX, self.sigmaY
#
#         def get2DGaussian(dose_profile):
# =============================================================================
# =============================================================================
#
#             y, x = np.mgrid[: np.shape(dose_profile)[0], : np.shape(dose_profile)[1]]
#             g = models.Gaussian2D(amplitude=8, x_mean=110, y_mean=90)
#             fitter = fitting.LevMarLSQFitter()
#             p = fitter(g, x, y, dose_profile)
#             plt.figure(figsize=(8, 2.5))
#             plt.subplot(1, 3, 1)
#             plt.imshow(dose_profile, cmap="jet", interpolation="nearest", vmin=0, vmax=16)
#             plt.title("Data")
#             plt.colorbar()
#             plt.subplot(1, 3, 2)
#             plt.imshow(p(x, y), cmap="jet", interpolation="nearest")
#             plt.title("Gaussian Fit")
#             plt.colorbar()
#             plt.subplot(1, 3, 3)
#             plt.imshow(
#                 dose_profile - p(x, y), cmap="jet", interpolation="nearest", vmin=-1, vmax=1)
#             plt.title("Residual")
#             plt.colorbar()
#
#             return p
#
#
#
#
#
#
#
#     # method to convert image matrix to dose with calibration constants
#     def get_dose(a, b, c, X_d):
#         # conversion formula and matrix manipulation to retrieve dose
#         dose = np.fliplr(np.rot90((a - c * X_d) / (X_d - b), 3))
#         return dose
#
#
#     calibration_curves={}
#     calibration_curves['EBT3']={}
#     calibration_curves['MDV3']={}
#     calibration_curves['HDV2']={}
#     calibration_curves['EBT3']['LOT']={}
#     calibration_curves['MDV3']['LOT']={}
#     calibration_curves['HDV2']['LOT']={}
#     calibration_curves['EBT3']['LOT']['DATE']={}
#     calibration_curves['MDV3']['LOT']['DATE']={}
#     calibration_curves['HDV2']['LOT']['DATE']={}
#
#
#     film_dict[film_name]['crop']=im.crop((int(width/2-20), int(height/2-20),int(width/2+21),int(height/2+21)))
#     film.dict[film_name]['dose']=int(filename.split('Gy')[0])
#
#         #ebt3 full calibration params
#
#        # a_r = 1.08054260e-02
#        # b_r = 3.96464552e-04
#        # c_r = 4.03747782e+00
#
#        # a_g = 1.53583642e-02
#        # b_g = 9.04451292e-05
#        # c_g = 6.24521113e+00
#
#        # a_b = 2.25123199e-02
#        # b_b = 2.75485087e-04
#        # c_b = 1.36316506e+01
#
#        #mdv3 calibration params
#
#         a_r = 7.21071436e-02
#         b_r = 4.67275120e-04
#         c_r = 2.38511751e+01
#
#         a_g = 1.11306451e-01
#         b_g = 3.18670259e-04
#         c_g = 3.79674212e+01
#
#         a_b = 2.32990669e-01
#         b_b = 7.33015188e-04
#         c_b = 9.54995622e+01
#
#
#         # retrieve red, green and blue image data from file
#         rX_d, gX_d, bX_d = get_rgb_d_from_file(filename)
#         # retrieve dose in each colour channel
#         self.dose_red = get_dose(a_r, b_r, c_r, rX_d)
#         self.dose_green = get_dose(a_g, b_g, c_g, gX_d)
#         self.dose_blue = get_dose(a_b, b_b, c_b, bX_d)
# =============================================================================


class CalibrationFilm(RadioChromicFilm):
    def __init__(self, image, dose, cropPercent=0, RGB_max=65535, dpi=300):
        RadioChromicFilm.__init__(self, image, cropPercent, RGB_max, dpi)
        self.dose = dose
        self.OD_r = -np.log(self.r)  # optical density
        self.OD_g = -np.log(self.g)
        self.OD_b = -np.log(self.b)


def loadAndCrop(filmDir, cutoff):
    '''Loads all .tif filmscans in a given directory, converts them to RGB format, and crops them by cutoff percent on each side'''
    filmDir = os.fsencode(filmDir)
    film_dict = {}  # dictionary to store films
    for file in os.listdir(filmDir):
        filename = os.fsdecode(file)
        if filename.endswith('.tif'):
            film_name = filename.split('.')[0]
            im = Image.open(os.path.join(filmDir, filename)).convert('RGB')
            width, height = im.size  # film width and height
            film_dict[film_name] = {}
            film_dict[film_name]['crop'] = im.crop(
                (int(width/2-20), int(height/2-20), int(width/2+21), int(height/2+21)))
            film_dict[film_name]['dose'] = int(filename.split('Gy')[0])
        else:
            print(f'{filename} is not a .tif file. Skipping it...')

        return film_dict


# %%
# =============================================================================
# RPV=[] #array of red values for all films
# GPV=[]
# BPV=[]
# dose=[]
# calib_films=OrderedDict(sorted(calib_films.items(), key=lambda x:getitem(x[1],'dose')))
#
#
#
# OD_R = -np.log(np.array(RPV)/RGB_max) #optical density
# OD_G = -np.log(np.array(GPV)/RGB_max)
# OD_B = -np.log(np.array(BPV)/RGB_max)
#
# def get_rgb_d_from_file(filename):
#     input_film = Image.open(filename).convert("RGB")
#     width, height= input_film.size #film width and height
#     print(width,height)
#     cropped_film=input_film.crop((int(width/2-100), int(height/2-140),int(width/2+120),int(height/2+42)))
#     input_PV = np.array(cropped_film)
#     X = input_PV / RGB_max #Channel value
#     rX, gX, bX = np.transpose(X)
#     #X_transpose = np.transpose(X)
#     #rX_d, gX_d, bX_d = X_d_transpose
#
#     return rX, gX, bX
#
# dose = np.array(dose)
#
# plt.scatter(dose, OD_R, color='red')
# plt.scatter(dose, OD_G, color='green')
# plt.scatter(dose, OD_B, color='blue')
#
# def calib_func(d, a, b, c):
#     return -1 * np.log((a + b*d)/(c + d))
#
# popt, pcov = curve_fit(calib_func, dose, OD_R)
# print(popt)
# plt.plot(dose, calib_func(dose, *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# popt, pcov = curve_fit(calib_func, dose, OD_G)
# print(popt)
# plt.plot(dose, calib_func(dose, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# popt, pcov = curve_fit(calib_func, dose, OD_B)
# print(popt)
# plt.plot(dose, calib_func(dose, *popt), 'b--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# plt.xlabel('Dose (Gy)')
# plt.ylabel('Optical Density')
# plt.legend()
#
