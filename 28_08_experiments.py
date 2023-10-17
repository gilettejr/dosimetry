#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:15:20 2023

@author: robertsoncl
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filmProfile import filmProfile
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from yagProfile import yagProfile
sys.path.append('/home/robertsoncl/topas/')

# ebt-xd

# dd1 F011-F033
# dd2 F034-F035, G001-G021

# arrange film names and associate with charge values and quad settings


def organiseFilms():

    # 2x col out runs, 1x col in, 1x full cycle (OUTIN)
    THzValues29 = [[3.80, 3.85], [3.80, 3.79], [3.75, 3.74], [3.83, 3.82], [3.76, 3.75], [3.34, 3.37], [3.35, 3.32], [3.42, 3.39], [3.47, 3.45], [3.42, 3.39], [3.90, 2.46], [3.78, 0.93], [
        3.83, 0.94], [3.85, 0.93], [3.83, 0.92], [9.35, 9.49], [8.78, 8.76], [8.65, 8.63], [9.11, 9.08], [9.12, 9.08], [10.16, 6.12], [9.90, 2.43], [10.10, 2.48], [9.85, 2.44], [9.11, 2.40]]
    #INOUT,INOUT, OUTIN, OUTIN
    THzValues30 = [[8.50, 5.40], [8.69, 2.59], [8.50, 2.46], [8.56, 2.50], [8.68, 2.49], [8.80, 8.92], [8.58, 8.54], [8.57, 8.53], [8.32, 8.29], [8.52, 8.47], [19.34, 11.81], [18.09, 4.41], [17.37, 4.32], [18.66, 4.67], [18.26, 4.61], [18.97, 19.24], [18.47, 18.35], [18.37, 18.27], [18.07, 17.97], [
        16.99, 16.89], [37.4, 37.5], [37.01, 36.7], [37.2, 36.1], [35.3, 35.1], [35.9, 35.6], [37.8, 20.8], [34.9, 9.5], [34.8, 9.5], [33.9, 9.1], [32.6, 8.9], [20.1, 20.4], [20.2, 20.1], [19.4, 19.3], [20.6, 20.5], [20.4, 20.3], [20.8, 13.5], [21.8, 5.3], [19.8, 4.8], [20.5, 5.0], [20.5, 5.1]]
    # INOUT OUTIN
    THzValues31 = [[44.8, 26.2], [39.8, 12.4], [41.2, 13.0,], [39.2, 12.4], [38.1, 11.7], [44.2, 42.1], [42.5, 40.5], [41.9, 40.3], [42.9, 40.9], [
        40.9, 39.6], [20.4, 20.7], [19.6, 19.6], [20.3, 20.3], [19.9, 19.9], [20.4, 20.4], [19.4, 12.3], [20.2, 5.2], [19.2, 4.9], [19.9, 5.1], [19.1, 4.9]]
    # OUTIN, OUTIN (both missing later air s2 out measurement)
    THzValues01 = [[20.1, 20.3], [20.2, 20.1], [20.0, 20.0], [20.3, 20.2], [20.3, 20.2], [20.3, 6.2], [20.2, 6.1], [20.0, 6.2], [
        20.1, 6.2], [2.1, 2.1], [2.0, 1.97], [1.92, 1.92], [2.01, 1.97], [2.00, 1.94], [2.07, 0.45], [2.04, 0.44], [2.12, 0.44], [2.05, 0.44]]

    def sequence(flagList):
        flagList.append('INOUTAIR-4')
        flagList.append('ININWATER-4')
        flagList.append('ININWATER+36')
        flagList.append('ININWATER+76')
        flagList.append('ININAIR-4')

    def colSequence(colList, colBool):
        for i in range(5):
            colList.append(colBool)

    def createFlagList():
        fl = []
        for i in range(18):
            sequence(fl)
        fl.append('ININWATER-4')
        fl.append('ININWATER+36')
        fl.append('ININWATER+76')
        fl.append('ININAIR-4')
        sequence(fl)
        fl.append('ININWATER-4')
        fl.append('ININWATER+36')
        fl.append('ININWATER+76')
        fl.append('ININAIR-4')
        return(fl)

    def createFilmList():
        # includes top and bottom
        def createRun(filmList, letter, start, stop):
            for i in range(start, stop+1):
                if i < 10:
                    filmList.append(letter+'00'+str(i))
                else:
                    filmList.append(letter+'0'+str(i))
        # A and B are EBT-3

        day1 = ['A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010', 'C001', 'C002', 'C003', 'A011', 'A012', 'A013', 'A014', 'A015', 'A016',
                'A017', 'A018', 'A019', 'A020', 'A021', 'A022', 'A023', 'A024', 'A025', 'A026', 'A027', 'A028', 'A029', 'A030', 'A031', 'A032', 'A033', 'A034', 'A035']
        day1back = ['A001', 'A002', 'A005', 'A009', 'A010', 'C001',
                    'C002', 'A015', 'A016', 'A027', 'A028', 'A034', 'A035']
        # C18 and 19 are X-ray films
        day2back = ['C004', 'C005', 'C011', 'C017', 'D011', 'D012', 'C018', 'C019',
                    'F001', 'F002', 'F013', 'F017', 'F018', 'F026', 'F027', 'F028']
        # i think E18 and E19 were x-ray
        day3back = ['F029', 'F030', 'E008',
                    'E009', 'E018', 'E019', 'E020', 'E021']
        day4back = ['D013', 'D014', 'D020', 'D030', 'D035']
        createRun(day1, 'C', 4, 15)
        day1.extend(['C016', 'C017'])
        day1.extend(['D001', 'D002'])
        createRun(day1, 'D', 3, 12)
        day1.extend(['C018', 'C019'])
        createRun(day1, 'F', 1, 28)
        createRun(day1, 'F', 29, 35)
        createRun(day1, 'E', 1, 21)
        createRun(day1, 'D', 13, 35)

        # E pellet, B X-Rays
        #createRun(day1, 'E', 22, 30)
        #createRun(day1, 'B', 1, 20)
        day1back.extend(day2back)
        day1back.extend(day3back)
        day1back.extend(day4back)
        for i in day1back:
            day1.remove(i)
        return(day1)
        # print(day1)

    def createColList():
        cl = []
        colSequence(cl, False)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, True)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, False)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, True)
        colSequence(cl, False)
        colSequence(cl, False)
        colSequence(cl, True)
        colSequence(cl, False)
        for i in range(4):
            cl.append(True)
        colSequence(cl, False)
        for i in range(4):
            cl.append(True)
        return(cl)

    def createQlist():
        ql = []
        for i in range(25):
            ql.append('4447')
        for i in range(10):
            ql.append('3336')
        for i in range(30):
            ql.append('4447')
        for i in range(38):
            ql.append('3839')
        return ql

    filmList = createFilmList()
    fl = createFlagList()
    cl = createColList()
    ql = createQlist()

    # labelling convention for irradiation runs
    # irradiations in sets of 5, always: s1INs2OUTAIR, s1INs2INWATER-4,36,76, s1INs2INAIR
    # Plot each set together with corresponding charge measurement
    # use dataframe with appropriate columns to match Film names etc

    allTHzValues = np.concatenate(
        (THzValues29, THzValues30, THzValues31, THzValues01))
    allTHz1Values = np.transpose(allTHzValues)[0]
    allTHz2Values = np.transpose(allTHzValues)[1]
    allTransport = (allTHz2Values/allTHz1Values) * 100
    mf = pd.DataFrame({'ScatFlag': fl, 'Col': cl,
                      'THz1': allTHz1Values, 'THz2': allTHz2Values, 'Quads': ql}, index=filmList)
    return mf


mf = organiseFilms()

# ebt-3

a_r = 9.53385895e-03
b_r = 4.40553270e-04
c_r = 3.71012295e+00

a_g3 = 1.36037090e-02
b_g3 = 1.48062223e-04
c_g3 = 5.72374622e+00

#a_g3 = 5.69
#b_g3 = 0.0599
#c_g3 = 5.75


a_b = 2.22791213e-02
b_b = 3.04096810e-04
c_b = 1.43716055e+01

# ebt-xd


a_gx = 5.06620041e-02
b_gx = 9.34532386e-05
c_gx = 1.81500016e+01


def getGaussianSigmas():
    names = ['29_08/A003', '29_08/C003', '29_08/A022', '30_08/C012', '30_08/D006',
             '30_08/F003', '30_08/F014', '31_08/E001', '31_08/E006', '1_09/D015',]
    centres = [[20.5, 24.5], [20, 21.5]]
    # test
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    for i in range(len(names)):

        name = names[i]+'.tif'
        if i < 4:
            a_g = a_g3
            b_g = b_g3
            c_g = c_g3
        else:
            a_g = a_gx
            b_g = b_gx
            c_g = c_gx
        film = filmProfile(path_to_folder+name, a_g, b_g, c_g)
        ft2 = film.get2DSuperGaussian(power=1)
        a = ft2['muX'].at[0]
        b = ft2['muY'].at[0]
        film.get_dose_strips(1.1,
                             a, b)
        ft1 = film.get1DSuperGaussian(power=1)
        print('Values from 2D Fits:')
        print(ft2)
        print('Values from 1D Fits: ')
        print(ft1)
        film.plot_xy_fits()
        #film.plot_xy_hists(a, b, 10, lim=0.5)
        #print('Values from direct CDF measurements: ')
        #film.getDensityStats(a, b, 15)


def getAirDoses():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    namedates = ['29_08/A021', '29_08/A033', '30_08/C010', '30_08/D005', '30_08/F012',
                 '30_08/F025', '31_08/F035', '31_08/E017', '1_09/D024']
    doses = []
    errs = []
    for i in range(len(namedates)):
        if i < 3:
            a_g = a_g3
            b_g = b_g3
            c_g = c_g3
        else:
            a_g = a_gx
            b_g = b_gx
            c_g = c_gx

        film = FlatRadioChromicFilm(path_to_folder+namedates[i]+'.tif')

        mf = film.get2DSuperGaussian(a_g, b_g, c_g, power=2)
        cx = mf['muX'].at[0]
        cy = mf['muY'].at[0]

        film.get_dose_strips(a_g, b_g, c_g, 1.1, cx, cy)
        dose, err = film.get_dose_region(a_g, b_g, c_g, cx, cy, 1.1, 1.1)
        doses.append(dose)
        errs.append(err)
        film.get1DSuperGaussian(a_g, b_g, c_g, power=2)
        film.plot_xy_fits(False)
    dose = np.array((doses, errs))
    np.save('/home/robertsoncl/dphil/dosimetry/28-08-23/air5', dose)


def plotAirCharges():
    doserr = np.load('/home/robertsoncl/dphil/dosimetry/28-08-23/air5.npy')
    doses = doserr[0]
    errs = doserr[1]
    names = ['A021', 'A033', 'C010', 'D005', 'F012',
             'F025', 'F035', 'E017', 'D024']
    charges = []
    for i in names:
        charges.append(mf['THz2'].loc[i])

    # The model expects shape (n_samples, n_features).
    x_data = np.array(charges).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_data, doses, sample_weight=1/errs)
    fit = model.predict(x_data)
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(charges, doses, yerr=errs, capsize=2,
                linestyle='none', color='black')
    ax.plot(charges, fit, color='red')
    ax.grid(True)
    ax.set_xlabel('Charge at THz2 [nC]')
    ax.set_ylabel('Dose across 1mm region')
    ax.text(8, 5, 'y='+str(round(model.coef_[0], 2)) +
            '+/-'+str(round(mean_squared_error(doses, fit), 1))+'x'+str(round(fit[2]-charges[2]*model.coef_[0], 2)))


def getWaterColOut():
    namedates = ['29_08/A023', '29_08/A024', '29_08/A025', '29_08/A026']
    namedates = ['31_08/E002', '31_08/E003', '31_08/E004', '31_08/E005']
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    for i in range(len(namedates)):
        a_g = a_gx
        b_g = b_gx
        c_g = c_gx
        film = FlatRadioChromicFilm(path_to_folder+namedates[i]+'.tif')

        #mf = film.get2DSuperGaussian(a_g, b_g, c_g, power=2)
        #cx = mf['muX'].at[0]
        #cy = mf['muY'].at[0]
        film.get_dose_strips(a_g, b_g, c_g, 1.1, 17, 22)
        #film.get1DSuperGaussian(a_g, b_g, c_g, power=1)

        film.plot_xy_fits(False)


def getWaterDoses():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    # problem with 18,19
    namedates = [['29_08/A030', '29_08/A031', '29_08/A032'], ['30_08/C007', '30_08/C008', '30_08/C009'], ['30_08/D002', '30_08/D003', '30_08/D004'], ['30_08/F009', '30_08/F010',
                                                                                                                                                      '30_08/F011'], ['30_08/F022', '30_08/F023', '30_08/F024'], ['31_08/F032', '31_08/F033', '31_08/F034'], ['31_08/E014', '31_08/E015', '31_08/E016'], ['1_09/D021', '1_09/D022', '1_09/D023']]
    dosesnear = []
    errsnear = []
    dosesmid = []
    errsmid = []
    dosesfar = []
    errsfar = []
    for i in range(len(namedates)):
        if i < 2:
            a_g = a_g3
            b_g = b_g3
            c_g = c_g3
        else:
            a_g = a_gx
            b_g = b_gx
            c_g = c_gx
        for j in range(len(namedates[i])):

            film = FlatRadioChromicFilm(path_to_folder+namedates[i][j]+'.tif')

            mf = film.get2DSuperGaussian(a_g, b_g, c_g, power=2)
            cx = mf['muX'].at[0]
            cy = mf['muY'].at[0]

            film.get_dose_strips(a_g, b_g, c_g, 1.1, cx, cy)
            dose, err = film.get_dose_region(a_g, b_g, c_g, cx, cy, 1.1, 1.1)
            if j == 0:
                dosesnear.append(dose)
                errsnear.append(err)
            elif j == 1:
                dosesmid.append(dose)
                errsmid.append(err)
            else:
                dosesfar.append(dose)
                errsfar.append(err)
            film.get1DSuperGaussian(a_g, b_g, c_g, power=2)
            film.plot_xy_fits(showFit=False)
    dosenear = np.array((dosesnear, errsnear))
    dosemid = np.array((dosesmid, errsmid))
    dosefar = np.array((dosesfar, errsfar))
    np.save('/home/robertsoncl/dphil/dosimetry/28-08-23/waternear', dosenear)
    np.save('/home/robertsoncl/dphil/dosimetry/28-08-23/watermid', dosemid)
    np.save('/home/robertsoncl/dphil/dosimetry/28-08-23/waterfar', dosefar)


def plotWaterCurves():
    namedates = [['29_08/A030', '29_08/A031', '29_08/A032'], ['30_08/C007', '30_08/C008', '30_08/C009'], ['30_08/D002', '30_08/D003', '30_08/D004'], ['30_08/F009', '30_08/F010',
                                                                                                                                                      '30_08/F011'], ['30_08/F022', '30_08/F023', '30_08/F024'], ['31_08/F032', '31_08/F033', '31_08/F034'], ['31_08/E014', '31_08/E015', '31_08/E016'], ['1_09/D021', '1_09/D022', '1_09/D023']]

    doserrnear = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/waternear.npy')
    doserrmid = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/watermid.npy')
    doserrfar = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/waterfar.npy')
    dosesnear = doserrnear[0]
    errsnear = doserrnear[1]
    dosesmid = doserrmid[0]
    errsmid = doserrmid[1]
    dosesfar = doserrfar[0]
    errsfar = doserrfar[1]
    chargesnear = []
    depths = [20, 60, 100]
    for i in range(len(dosesnear)):
        fig, ax, = plt.subplots(1, 1, figsize=(8, 8))
        ax.errorbar(depths, [dosesnear[i], dosesmid[i], dosesfar[i]], yerr=[
                    errsnear[i], errsmid[i], errsfar[i]], capsize=20, color='k', linestyle='none')
        ax.set_xlabel('Depth [mm]')
        ax.set_ylabel('Dose [Gy]')
        ax.grid(True)
        ax.set_title(namedates[i][0])


def plotWaterCharges():
    namedates = [['29_08/A030', '29_08/A031', '29_08/A032'], ['30_08/C007', '30_08/C008', '30_08/C009'], ['30_08/D002', '30_08/D003', '30_08/D004'], ['30_08/F009', '30_08/F010',
                                                                                                                                                      '30_08/F011'], ['30_08/F022', '30_08/F023', '30_08/F024'], ['31_08/F032', '31_08/F033', '31_08/F034'], ['31_08/E014', '31_08/E015', '31_08/E016'], ['1_09/D021', '1_09/D022', '1_09/D023']]
    namesnear = []
    namesmid = []
    namesfar = []
    for i in namedates:
        namesnear.append(i[0][-4:])
        namesmid.append(i[1][-4:])
        namesfar.append(i[2][-4:])
    print(namesnear)
    doserrnear = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/waternear.npy')
    doserrmid = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/watermid.npy')
    doserrfar = np.load(
        '/home/robertsoncl/dphil/dosimetry/28-08-23/waterfar.npy')
    dosesnear = doserrnear[0]
    errsnear = doserrnear[1]
    dosesmid = doserrmid[0]
    errsmid = doserrmid[1]
    dosesfar = doserrfar[0]
    errsfar = doserrfar[1]
    chargesnear = []
    chargesmid = []
    chargesfar = []
    for i in range(len(namesnear)):
        chargesnear.append(mf['THz2'].loc[namesnear[i]])
        chargesmid.append(mf['THz2'].loc[namesmid[i]])
        chargesfar.append(mf['THz2'].loc[namesfar[i]])

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))

    # The model expects shape (n_samples, n_features).
    x_data = np.array(chargesnear).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_data, dosesnear, sample_weight=1/errsnear)
    fit = model.predict(x_data)

    ax[0].errorbar(chargesnear, dosesnear, yerr=errsnear, capsize=2,
                   linestyle='none', color='black')
    ax[0].plot(chargesnear, fit, color='red')
    ax[0].grid(True)
    ax[0].set_xlabel('Charge at THz2 [nC]')
    ax[0].set_ylabel('Dose across 1mm region')
    ax[0].text(8, 5, 'y='+str(round(model.coef_[0], 2)) +
               '+/-'+str(round(mean_squared_error(dosesnear, fit), 1))+'x + ' + str(round(fit[2]-chargesnear[2]*model.coef_[0], 2)))
    x_data = np.array(chargesmid).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_data, dosesmid, sample_weight=1/errsmid)
    fit = model.predict(x_data)

    ax[1].errorbar(chargesmid, dosesmid, yerr=errsmid, capsize=2,
                   linestyle='none', color='black')
    ax[1].plot(chargesmid, fit, color='red')
    ax[1].grid(True)
    ax[1].set_xlabel('Charge at THz2 [nC]')
    ax[1].set_ylabel('Dose across 1mm region')
    ax[1].text(8, 5, 'y='+str(round(model.coef_[0], 2)) +
               '+/-'+str(round(mean_squared_error(dosesmid, fit), 1))+'x + ' + str(round(fit[2]-chargesmid[2]*model.coef_[0], 2)))
    x_data = np.array(chargesfar).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_data, dosesfar, sample_weight=1/errsfar)
    fit = model.predict(x_data)

    ax[2].errorbar(chargesfar, dosesfar, yerr=errsfar, capsize=2,
                   linestyle='none', color='black')
    ax[2].plot(chargesfar, fit, color='red')
    ax[2].grid(True)
    ax[2].set_xlabel('Charge at THz2 [nC]')
    ax[2].set_ylabel('Dose across 1mm region')
    ax[2].text(8, 5, 'y='+str(round(model.coef_[0], 2)) +
               '+/-'+str(round(mean_squared_error(dosesfar, fit), 1))+'x + ' + str(round(fit[2]-chargesfar[2]*model.coef_[0], 2)))


def getYAGSigmas():

    preamble = '/home/robertsoncl/dphil/dosimetry/28-08-23/YAG/image_'
    names = ['29-07-23_5-3.mat', '29-07-23_14-3.mat',
             '30-07-23_2-3.mat', '30-07-23_12-3.mat', '30-07-23_22-3.mat']
    # image_29_07_23_4
    for i in names:
        yag = yagProfile(
            preamble+i)
        ft2 = yag.get2DSuperGaussian(power=1)
        cx = ft2['muX'][0]
        cy = ft2['muY'][0]

        yag.get_yag_strips(1.1, cx, cy)
        ft1 = yag.get1DSuperGaussian(power=1)
        print(ft2)
        print(ft1)
        yag.plot_xy_fits()


def showFLASHYAG():
    from yagIntensityPlotter import yagIntensityPlotter
    preamble = '/home/robertsoncl/dphil/dosimetry/28-08-23/YAG/image_'
    names = ['30-07-23_18-1.mat', '30-07-23_18-2.mat', '30-07-23_18-3.mat']
    for i in names:
        yag = yagIntensityPlotter(
            preamble+i)
        ft = yag.get2DSuperGaussian(power=2)
        cx = ft['muX'][0]
        cy = ft['muY'][0]
        yag.get_yag_strips(10, cx, cy)
        yag.get1DSuperGaussian(power=2)
        yag.plotxyFits(False)


def sizeComparison():
    charge = [2, 10, 2, 40, 2]
    x_ratio = [4.09/3.89, 3.86/4.07, 3.92/3.87, 4.29/4.85, 3.87/3.82]
    y_ratio = [4.13/3.98, 3.96/4.17, 3.92/3.95, 4.1/4.65, 3.88/3.86]
    fig, ax = plt.subplots()
    ax.scatter(charge, x_ratio, color='r', label='x')
    ax.scatter(charge, y_ratio, color='k', label='y')
    ax.set_xlabel('Charge/shot [nC]')
    ax.set_ylabel('$\sigma_{film}$/$\sigma_{YAG}$')
    ax.legend()
    ax.grid(True)


def xrayTest():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    namedates = ['1_09/B001', '1_09/B005',
                 '1_09/B010', '1_09/B015', '1_09/B020']
    a_g = a_g3
    b_g = b_g3
    c_g = c_g3
    for i in namedates:
        film = FlatRadioChromicFilm(path_to_folder+i+'.tif')
        film.get_dose_strips(a_g, b_g, c_g, 10, 17, 17)
        film.plot_xy_fits(False)

#


def pelletTest():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/28-08-23/Cam_08_23/'
    namedates = ['1_09/E023', '1_09/E024', '1_09/E025',
                 '1_09/E026', '1_09/E027', '1_09/E028', '1_09/E029']

    a_g = a_gx
    b_g = b_gx
    c_g = c_gx

    #mf = film.get2DSuperGaussian(a_g, b_g, c_g, power=2)
    #cx = mf['muX'].at[0]
    #cy = mf['muY'].at[0]

    cx24 = 18
    cy24 = 20
    h24 = 1.1
    w24 = 2

    cy26 = 21
    cx26 = 17
    w26 = 3
    h26 = 2

    cx28 = 18
    cy28 = 20
    h28 = 1.1
    w28 = 3
    film = FlatRadioChromicFilm(path_to_folder+namedates[0]+'.tif')
    t = film.get2DSuperGaussian(a_g, b_g, c_g)
    cx = t['muX'][0]
    cy = t['muY'][0]
    film.get_dose_strips(a_g, b_g, c_g, h24, cx, cy)
    film.get_dose_region(a_g, b_g, c_g, cx, cy, w24, h24)
    film.plot_xy_fits(False)

    film = FlatRadioChromicFilm(path_to_folder+namedates[2]+'.tif')
    t = film.get2DSuperGaussian(a_g, b_g, c_g)
    cx = t['muX'][0]
    cy = t['muY'][0]
    film.get_dose_strips(a_g, b_g, c_g, h26, cx26, cy26)
    film.get_dose_region(a_g, b_g, c_g, cx, cy, w26, h26)
    film.plot_xy_fits(False)

    film = FlatRadioChromicFilm(path_to_folder+namedates[4]+'.tif')
    t = film.get2DSuperGaussian(a_g, b_g, c_g)
    cx = t['muX'][0]
    cy = t['muY'][0]
    film.get_dose_strips(a_g, b_g, c_g, h28, cx28, cy28)
    film.get_dose_region(a_g, b_g, c_g, cx, cy, w28, h28)
    film.plot_xy_fits(False)
    #dose, err = film.get_dose_region(a_g, b_g, c_g, cx, cy, 1.1, 1.1)

# getGaussianSigmas()


def main():
    #mf = organiseFilms()

    # print(mf.loc['A022'])
    # print(mf.loc['C012'])
   # print(mf.loc['D006'])
    # print(mf.loc['F003'])
   # print(mf.loc['F014'])
    # print(mf.loc['E001'])
    # print(mf.loc['E006'])
    # print(mf.loc['D015'])

    # getAirDoses()
    # plotAirCharges()
    # plotWaterCurves()
    # getWaterDoses()
    # plotWaterCurves()
    getYAGSigmas()
    # getGaussianSigmas()
    # sizeComparison()
    # showFLASHYAG()
    # getWaterDoses()
    # plotWaterCharges()
    # xrayTest()
    # pelletTest()
    # getWaterColOut()


main()
