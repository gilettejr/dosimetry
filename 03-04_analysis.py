from flattenedFilmDosimetryTools import FlatRadioChromicFilm
from flattenedFilmProfileTools import flatYagProfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a_g = 1.53583642e-02
b_g = 9.04451292e-05
c_g = 6.24521113e+00

a_g = 5.531
b_g = 0.051
c_g = 5.614


def get_superG_params_folder(path_to_folder, film_no_range, prefix='Film_'):

    film_list = []
    for i in range(film_no_range[0], film_no_range[1]+1, 1):
        if i < 10:

            film_list.append(prefix+'00'+str(i)+'.tif')
        else:
            film_list.append(prefix+'0'+str(i)+'.tif')

    film_frame = pd.DataFrame({'muX': [], 'muY': [], 'sigX': [], 'sigY': [
    ], 'power': [], 'peak dose': []})
    for i in film_list:

        film = FlatRadioChromicFilm(path_to_folder+i)
        #film.trim((5, 5, 30, 5))
        film.get_strips(200, 200, 10)
        film.computeDoseMap(a_g, b_g, c_g)
        # film_frame = pd.concat(
        # [film_frame, film.get2DSuperGaussian(plot=True)])
    return film_frame


def show_all(path_to_folder, film_no_range, prefix='Film_'):
    a_g = 1.53583642e-02
    b_g = 9.04451292e-05
    c_g = 6.24521113e+00
    film_list = []
    for i in range(film_no_range[0], film_no_range[1]+1, 1):
        if i < 10:

            film_list.append(prefix+'00'+str(i)+'.tif')
        else:
            film_list.append(prefix+'0'+str(i)+'.tif')

    film_frame = pd.DataFrame({'muX': [], 'muY': [], 'sigX': [], 'sigY': [
    ], 'power': [], 'peak dose': []})
    for i in film_list:

        film = FlatRadioChromicFilm(path_to_folder+i)
        #film.trim((5, 5, 5, 5))
        film.get_strips(15, 20, 1)
        film.computeDoseMap(a_g, b_g, c_g)
        film.show_dose()


def film_main():

    # test
    #path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/Day3_eve/'
    film = FlatRadioChromicFilm(
        '/home/robertsoncl/dphil/dosimetry/28-07-22/Film_010.tif')

    #film.trim((5, 20, 20, 15))
    #film.get_dose_strips(a_g, b_g, c_g, 1.5, 13.5, 13)
    # film.show_dose()
    # film.plot_xy_hists()
    #mean, std = film.get_stats()
    #print(mean, std)

    #film.trim((5, 5, 30, 5))
    film.get_dose_strips(1.53583642e-02,
                         9.04451292e-05,
                         6.24521113e+00, 1, 12, 8)
    #film.computeDoseMap(a_g, b_g, c_g)
    print(film.get2DSuperGaussian())
    # long air
    #path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/Day3_long_air/'
    #frame = get_superG_params_folder(path_to_folder, [4, 26], prefix='B')

    # long water
    #path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/Day3_long_water/'
    #frame1 = get_superG_params_folder(path_to_folder, [27, 35], prefix='B')
    #frame2 = get_superG_params_folder(path_to_folder, [1, 14], prefix='C')
    # evening
    #path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/Day3_eve/'
    #frame = get_superG_params_folder(path_to_folder, [2, 11], prefix='M')
    #show_all(path_to_folder, [2, 11], prefix='M')

    #correlation = np.corrcoef(frame['power'], frame['norm sig dose'])
    # print(frame['power'])
    # print(frame1['sigX'])


def profile_main():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/images/'
    profile = 'Cameron_Beam_05_04_2.mat'
    profile = flatYagProfile(path_to_folder+profile)
    # good up to and including 5
    profile.trim([27.5, 21, 10, 32])
    # from 6 onwards
    #profile.trim([37, 30, 5, 32])
    profile.get_strips(5, 20, 11)
    profile.get_stats()
    profile.plot_xy_hists()


def strip_deviation_plot():
    path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/images/'
    water_means = []
    water_stds = []
    water_positions = np.array(
        [-40, -30, -20, -10, 0, 20, 40, 60, 80, 100, 120])
    water_positions = water_positions+60
    air_means = []
    air_stds = []
    for i in range(11, 22):

        profile = 'Cameron_Beam_05_04_'+str(i)+'.mat'
        profile = flatYagProfile(path_to_folder+profile)
        # good up to and including 5
        #profile.trim([27.5, 21, 10, 32])
        # from 6 onwards
        profile.trim([37, 30, 5, 32])
        profile.get_strips(5, 12, 3)
        mean, std = profile.get_stats()
        water_means.append(mean)
        water_stds.append(std)
    water_means = np.array(water_means)
    water_stds = np.array(water_stds)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].errorbar(water_positions, water_means,
                   yerr=water_stds, capsize=5, linestyle='none', marker='o', color='k')
    ax[0].set_xlabel('Depth in Water [mm]')
    ax[0].set_ylabel('Mean Intensity [arb. units]')
    ax[1].plot(water_positions, water_means/water_stds, color='k')
    ax[1].set_xlabel('Depth in Water [mm]')
    ax[1].set_ylabel('Mean Intensity/Standard Deviation')


# strip_deviation_plot()
# profile_main()
film_main()
