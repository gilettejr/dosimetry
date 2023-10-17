#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file_name.py
# Python 3.8.3
"""
Author: Joseph Cockman
Created: Fri Nov  5 16:01:25 2021
Modified: Fri Nov  5 16:01:25 2021

Description 
Roses are red,
Violets are Blue,
This code should probably have comments,
But I can't be arsed so well, fuck you'

"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from astropy.modeling import models, fitting
from scipy.signal import savgol_filter
from scipy.stats import kurtosis
from scipy.ndimage import rotate
import matplotlib.patches as patches

# class holding fitting and dose measurement methods


class film_analyser:
    # initialise by defining calibration constants
    # and retrieving dose in seperate channels from given file
    def __init__(self, filename):
        # seperate film into rgb channels and return data values
        def get_rgb_d_from_file(filename):
            # open file and convert to Image object in RGB format for analysis
            input_image = Image.open(filename).convert("RGB")
            # convert to numpy array for ease
            input_imarray = np.array(input_image)
            # X_d reweighted because reasons
            X_d = np.array(input_imarray / 65535)
            # transpose carried out to retrieve channels from matrix
            X_d_transpose = np.transpose(X_d)
            # values in r,g,b colour channels retrieved
            rX_d, gX_d, bX_d = X_d_transpose
            return rX_d, gX_d, bX_d

        # method to convert image matrix to dose with calibration constants
        def get_dose(a, b, c, X_d):
            # conversion formula and matrix manipulation to retrieve dose
            dose = np.fliplr(np.rot90((a - c * X_d) / (X_d - b), 3))
            return dose

        # collapse 2d matrix into 1d matrix of median values
        def collapse_along_axis(matrix):
            # initialise list for holding result
            collapsed_matrix = []
            # loop through matrix and add median of each row to 1D array
            # median better than mean here as it smooths out extrema
            for i in matrix:
                collapsed_matrix.append(np.median(i))
            # return 1D array of medians
            return collapsed_matrix

        self.collapse_along_axis = collapse_along_axis

        # red calibration constants
        a_r = 1.08054260e-02
        b_r = 3.96464552e-04
        c_r = 4.03747782e00
        # green calibration constants
        a_g = 1.53583642e-02
        b_g = 9.04451292e-05
        c_g = 6.24521113e00
        # blue calibration constants
        a_b = 2.25123199e-02
        b_b = 2.75485087e-04
        c_b = 1.36316506e01
        # retrieve red, green and blue image data from file
        rX_d, gX_d, bX_d = get_rgb_d_from_file(filename)
        # retrieve dose in each colour channel
        self.dose_red = get_dose(a_r, b_r, c_r, rX_d)
        self.dose_green = get_dose(a_g, b_g, c_g, gX_d)
        self.dose_blue = get_dose(a_b, b_b, c_b, bX_d)

    # vile function to automatically crop images
    # janky as shit, whoever's reading this pls make it better

    def remove_borders(
        self, dose_profile, left_xlim=80, right_xlim=350, up_ylim=90, down_ylim=350
    ):
        # collapse 2d matrix into 1d matrix of median values
        def collapse_along_axis(matrix):
            # initialise list for holding result
            collapsed_matrix = []
            # loop through matrix and add median of each row to 1D array
            # median better than mean here as it smooths out extrema
            for i in matrix:
                collapsed_matrix.append(np.median(i))
            # return 1D array of medians
            return collapsed_matrix

        # collapse matrix to get median x dose distributions along y axis
        collapsed_dose_along_y = collapse_along_axis(dose_profile)
        # collapse matrix to get median y dose distributions along x axis
        collapsed_dose_along_x = collapse_along_axis(
            np.transpose(dose_profile))
        # smooth data even further so that only significant spikes remain
        cdy_hat = savgol_filter(collapsed_dose_along_y, 40, 1)
        cdx_hat = savgol_filter(collapsed_dose_along_x, 40, 1)

        # uncomment lines for relevant graphs showing the smoothed fit

        # y_pixels = np.arange(0, len(dose_profile), 1)
        # x_pixels = np.arange(0, len(dose_profile_T), 1)

        # plt.figure()
        # plt.plot(y_pixels, cdy_hat)
        # plt.plot(y_pixels, collapsed_dose_along_y)
        # plt.yscale("log")
        # plt.figure()

        # plt.plot(x_pixels, cdx_hat)
        # plt.plot(x_pixels, collapsed_dose_along_x)

        # define maximum limits for cropping
        # decrease l variables and increase r variables to reduce max cropping
        lx_sp = left_xlim
        rx_sp = right_xlim
        ly_sp = up_ylim
        ry_sp = down_ylim

        # defines pixel limits in x and y
        # represent limits of what respective variables above can be set tp
        x_ltp = 0
        x_rtp = 440
        y_ltp = 0
        y_rtp = 504

        # search for increase in dose moving outwards from centre
        # and label detected edge coordinates
        # starting at max cropping limit

        # -ve x edge

        for i in range(0, lx_sp):
            if cdx_hat[lx_sp - i] < cdx_hat[lx_sp - i - 1]:
                x_ltp = lx_sp - i
                break
        # +ve x edge
        for i in range(rx_sp, len(cdx_hat) - 1):
            if cdx_hat[i] < cdx_hat[i + 1]:
                x_rtp = i
                break
        # -ve y edge
        for i in range(0, ly_sp):
            if cdy_hat[ly_sp - i] < cdy_hat[ly_sp - i - 1]:
                y_ltp = ly_sp - i
                break
        # +ve y edge
        for i in range(ry_sp, len(cdy_hat) - 1):
            if cdy_hat[i] < cdy_hat[i + 1]:
                y_rtp = i
                break
        # truncate 2d dose profile in y at the detected edges
        dose_profile = dose_profile[y_ltp: y_rtp + 1]
        # truncate 2d dose profile in x at the detected edges
        # save to new variable because numpy doesn't like in-place modification
        new_dose_profile = []
        for i in range(len(dose_profile)):
            new_dose_profile.append(dose_profile[i][x_ltp: x_rtp + 1])
        # convert new list variable to array
        new_dose_profile = np.array(new_dose_profile)
        # return cropped profile
        return new_dose_profile
        # print(len(dose_profile))
        # plt.plot(x_pixels, cdx_hat)
        # plt.plot(x_pixels, collapsed_dose_along_x)
        # plt.yscale("log")
        # plt.figure(figsize=(8, 2.5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(
        #    new_dose_profile, cmap="jet", interpolation="nearest", vmin=0, vmax=10
        # )
        # plt.title("Data")
        # plt.colorbar()

    # fit 2d gaussian to get beam size and return fit
    def fit_2D_gaussian_to_dose(self, dose_profile, filename, show_graphs=False):
        # get x and y coordinates from dose data
        y, x = np.mgrid[: np.shape(dose_profile)[0],
                        : np.shape(dose_profile)[1]]
        # define Astropy Gaussian model
        g = models.Gaussian2D()
        # intialise least squares fitter
        fitter = fitting.LevMarLSQFitter()
        # carry out fit to data
        p = fitter(g, x, y, dose_profile)
        if show_graphs is True:
            # plot dose, fit and residuals

            plt.figure(figsize=(8, 2.5))

            plt.subplot(1, 3, 1)
            plt.imshow(
                dose_profile, cmap="jet", interpolation="nearest", vmin=0, vmax=10
            )
            plt.title(filename + " Data")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(p(x, y), cmap="jet",
                       interpolation="nearest", vmin=0, vmax=10)
            plt.title("Gaussian Fit")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(
                dose_profile - p(x, y),
                cmap="jet",
                interpolation="nearest",
                vmin=-2,
                vmax=2,
            )
            plt.title("Residual")
            plt.colorbar()
        # return fit
        return p

    def get_dose_rms(self, profile, filename):

        # return fit
        y_vals_init = self.collapse_along_axis(profile)
        x_vals_init = self.collapse_along_axis(np.transpose(profile))
        x_bins = np.arange(-len(x_vals_init) / 2, len(x_vals_init) / 2, 1)
        y_bins = np.arange(-len(y_vals_init) / 2, len(y_vals_init) / 2, 1)

        x_vals = np.array(x_vals_init) * 1000
        y_vals = np.array(y_vals_init) * 1000
        x_vals = x_vals.astype(int)
        y_vals = y_vals.astype(int)
        x_val_array = []
        y_val_array = []
        for i in range(len(x_bins)):
            for j in range(x_vals[i]):
                x_val_array.append(x_bins[i])
        for i in range(len(y_bins)):
            for j in range(y_vals[i]):
                y_val_array.append(y_bins[i])
        x_val_array = np.array(x_val_array)
        y_val_array = np.array(y_val_array)
        x_rms = np.sqrt(np.mean(np.square(x_val_array)))
        y_rms = np.sqrt(np.mean(np.square(y_val_array)))
        x_kurt = kurtosis(x_val_array)
        y_kurt = kurtosis(y_val_array)
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)

        plt.imshow(profile, cmap="jet",
                   interpolation="nearest", vmin=0, vmax=20)
        plt.title(filename + " Data")
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.plot(x_bins, x_vals_init)
        plt.xlim([-200, 200])
        plt.title("1D profile in x")
        plt.subplot(1, 3, 3)
        plt.plot(y_bins, y_vals_init)
        plt.xlim([-200, 200])
        plt.title("1D profile in y")
        return x_rms, y_rms, x_kurt, y_kurt
        # x_hist=np.hist()

    def get_dose_strips(self, profile, filename, beam_centre=[-10, 10], angle=0):
        profile = rotate(profile, 0)
        y_strip = profile.copy()
        x_strip = profile.copy()
        x_strip = x_strip[
            int(len(x_strip) / 2 - (beam_centre[1] - beam_centre[0]) / 2): int(
                len(x_strip) / 2 + (beam_centre[1] - beam_centre[0]) / 2
            )
        ]

        y_strip = np.rot90(y_strip)
        y_strip = y_strip[
            int(len(y_strip) / 2 - (beam_centre[1] - beam_centre[0]) / 2): int(
                len(y_strip) / 2 + (beam_centre[1] - beam_centre[0]) / 2
            )
        ]

        # x_strip = x_strip[~np.isnan(x_strip)]
        # y_strip = y_strip[~np.isnan(y_strip)]

        # return fit
        y_vals_init = self.collapse_along_axis(np.transpose(y_strip))
        x_vals_init = self.collapse_along_axis(np.transpose(x_strip))
        x_bins = np.arange(0, len(x_vals_init), 1)
        y_bins = np.arange(0, len(y_vals_init), 1)

        x_vals = np.array(x_vals_init) * 1000
        y_vals = np.array(y_vals_init) * 1000
        x_vals = x_vals.astype(int)
        y_vals = y_vals.astype(int)
        x_val_array = []
        y_val_array = []
        for i in range(len(x_bins)):
            for j in range(x_vals[i]):
                x_val_array.append(x_bins[i])
        for i in range(len(y_bins)):
            for j in range(y_vals[i]):
                y_val_array.append(y_bins[i])
        x_val_array = np.array(x_val_array)
        y_val_array = np.array(y_val_array)

        # x_vals_init = savgol_filter(x_vals_init, 60, 3)
        # y_vals_init = savgol_filter(y_vals_init, 60, 3)

        fig, ax = plt.subplots(1, 3, figsize=[8, 2.5])
        x_rect = patches.Rectangle(
            (200, -200), 21, 800, linewidth=1, edgecolor="k", facecolor="none"
        )
        y_rect = patches.Rectangle(
            (-200, 200), 800, 21, linewidth=1, edgecolor="k", facecolor="none"
        )

        ax[0].imshow(
            profile,
            cmap="jet",
            interpolation="nearest",
            vmin=0,
            vmax=40,
        )

        ax[0].set_title(filename + " Data")
        ax[0].add_patch(x_rect)
        ax[0].add_patch(y_rect)
        # ax[0].add_colorbar()
        ax[1].plot(x_bins, x_vals_init)
        # ax[1].set_xlim([0, 400])
        ax[1].set_title("1D profile in x")
        ax[2].plot(y_bins, y_vals_init)
        # ax[2].set_xlim([0, 400])
        ax[2].set_title("1D profile in y")

    # construct table of fitting parameters
    def make_fit_params_table(self, red_fit, green_fit, blue_fit):
        def elliptical_transform(fake_sigx, fake_sigy, theta):
            def sigma(a, b, m):
                sigma = np.sqrt(
                    np.abs(np.divide((1 + m ** 2), ((1 / a) ** 2 + (m / b) ** 2)))
                )
                return sigma

            m_x = -np.tan(theta)
            m_y = np.tan(np.pi / 2 - theta)
            true_sigx = sigma(fake_sigx, fake_sigy, m_x)
            true_sigy = sigma(fake_sigx, fake_sigy, m_y)
            return true_sigx, true_sigy

        # create lists of data and labels for ease of looping through
        fit_list = [red_fit, green_fit, blue_fit]
        colours = ["Red", "Green", "Blue"]
        # initialise lists for holding fit data
        amplitudes = []
        x_means = []
        y_means = []
        x_stddevs = []
        y_stddevs = []
        x_true_stddevs = []
        y_true_stddevs = []
        # fill with data from astropy fitting object
        for i in fit_list:
            amplitudes.append(i.amplitude)
            x_means.append(i.x_mean)
            y_means.append(i.y_mean)
            true_sig_x, true_sig_y = elliptical_transform(
                i.x_stddev, i.y_stddev, i.theta
            )
            # I presume there's a reason for the weighting factor here?
            x_stddevs.append(i.x_stddev * 25.5 / 300)
            y_stddevs.append(i.y_stddev * 25.4 / 300)
            x_true_stddevs.append(true_sig_x * 25.5 / 300)
            y_true_stddevs.append(true_sig_y * 25.5 / 300)
            print(str(((np.abs(true_sig_x - i.x_stddev)) / i.x_stddev) * 100) + " %")
        # merge data into pandas DataFrame object
        fitting_data = pd.DataFrame(
            {
                "Colour": colours,
                "Amplitude": amplitudes,
                "x_mean": x_means,
                "y_mean": y_means,
                "x_stddev": x_stddevs,
                "y_stddev": y_stddevs,
            }
        )
        # return DataFrame object holding fitting data
        return fitting_data


# path must end with a forward slash
# function to loop through and analyse every film image in a directory
def analyse_all_films(path_to_directory_holding_images):
    # call film analyser object to retrieve dose and get gaussian fitting data
    def get_fitting_data_table_and_dose_green(filename):
        # initialise film analyser object
        film = film_analyser(filename)
        # get doses in colour channels
        dose_red, dose_green, dose_blue = film.dose_red, film.dose_green, film.dose_blue
        # crop images
        # add extra arguments to remove_borders() to define cropping limit
        dose_red = film.remove_borders(dose_red)
        dose_blue = film.remove_borders(dose_blue)
        dose_green = film.remove_borders(dose_green)
        # fit and retrieve fitting parameters for each colour
        for i in range(len(filename)):
            if filename[len(filename) - i - 1] == "/":
                filename = filename[len(filename) - i: len(filename) - 4]
                break
        # uncomment the line below if you want to take strips
        # I need to add functionality to manually set the strip locations tho
        # film.get_dose_strips(dose_green, filename)
        x_rms, y_rms, x_kurt, y_kurt = film.get_dose_rms(dose_green, filename)

        print("Filename: " + filename)
        print("x_rms= " + str(x_rms) + "mm")
        print("y_rms= " + str(y_rms) + "mm")

        print("x_kurt= " + str(x_kurt))
        print("y_kurt= " + str(y_kurt))
        fitted_red = film.fit_2D_gaussian_to_dose(dose_red, filename)
        fitted_green = film.fit_2D_gaussian_to_dose(dose_green, filename)
        fitted_blue = film.fit_2D_gaussian_to_dose(dose_blue, filename)

        # create table holding all fitting data for film
        fitting_data_table = film.make_fit_params_table(
            fitted_red, fitted_green, fitted_blue
        )
        # return table and green dose for further processing
        return fitting_data_table, dose_green

    # return max dose and save to file
    def get_max_dose(fitting_data_table, dose_green):

        # x mean position green channel
        xmean = fitting_data_table.iloc[1, 2] * 25.4 / 300

        # y mean position green channel
        ymean = fitting_data_table.iloc[1, 3] * 25.4 / 300

        dim = 10
        # find max dose
        max_dose = np.round(
            np.mean(
                dose_green[
                    int(fitting_data_table.iloc[1, 3] - dim / 2): int(
                        fitting_data_table.iloc[1, 3] + dim / 2
                    ),
                    int(fitting_data_table.iloc[1, 2] - dim / 2): int(
                        fitting_data_table.iloc[1, 2] + dim / 2
                    ),
                ]
            ),
            2,
        )
        # print max dose
        print(
            max_dose, fitting_data_table.iloc[1, 4], fitting_data_table.iloc[1, 5])
        # save results to .csv file
        with open("zfe_dosimetry.csv", "a", newline="") as csv_file:
            print(
                max_dose,
                fitting_data_table.iloc[1, 4],
                fitting_data_table.iloc[1, 5],
                file=csv_file,
                sep=",",
            )

    # get directory in bytes format
    directory = os.fsencode(path_to_directory_holding_images)
    # begin loop for analysing every file in folder
    for file in os.listdir(directory):

        # decode from bytes into string
        file_str = file.decode("utf-8")
        if file_str[0] != ".":
            filename = file_str
            # add path to directory to filename to point correctly
            file_str = path_to_directory_holding_images + file_str
            # generate and retrieve fitting table and green_dose
            fitting_data_table, dose_green = get_fitting_data_table_and_dose_green(
                file_str
            )
            print("File: " + filename)
            # print fitting tables
            print(fitting_data_table)
            # retrieve and print max_dose
            get_max_dose(fitting_data_table, dose_green)


def single_tester(path_to_file):
    fa = film_analyser(
        "/home/robertsoncl/dphil/dosimetry/28-07-22/Film_006.tif")
    cropped_profile = fa.remove_borders(fa.dose_green)
    fa.get_dose_strips(cropped_profile, "Film_006", [-10, 10])


# main function
def main():
    # carry out full analysis of films in chosen directory

    analyse_all_films("/home/robertsoncl/dphil/dosimetry/28-07-22/")

    # single_tester("/home/robertsoncl/dphil/dosimetry/28-07-22/Film_006.tif")


main()
