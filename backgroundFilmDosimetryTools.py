import numpy as np
from filmDosimetryTools import RadioChromicFilm


class backRadioChromicFilm(RadioChromicFilm):
    def get_background_subtraction(self):
        def PV_to_X(PV):
            return -1*np.log(PV/self.RGB_max)
        r_back, g_back, b_back = self.r, self.g, self.b
        r_back_val = np.mean(r_back.flatten())
        g_back_val = np.mean(g_back.flatten())
        b_back_val = np.mean(b_back.flatten())
        r_back_val = PV_to_X(r_back_val)
        g_back_val = PV_to_X(g_back_val)
        b_back_val = PV_to_X(b_back_val)

        return r_back_val, g_back_val, b_back_val


path_to_folder = '/home/robertsoncl/dphil/dosimetry/03-04-23/Day3_xray_and_back/'
film = backRadioChromicFilm(path_to_folder+'K032.tif')

film.trimEdges(25)
rb, gb, bb = film.get_background_subtraction()

print(rb, gb, bb)
