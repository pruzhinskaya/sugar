"""interpolate sed of snia using gaussian process."""

import numpy as np
import cosmogp
import sugar
import os


class load_data_bin_gp:
    """load data for gp for one wavelength bin."""
    def __init__(self):
        """
        create data used for gp for a given bin.
        """
        self.lds = sugar.load_data_sugar()
        self.lds.load_spectra()

        self.sn_name = self.lds.sn_name

        self.y = []
        self.y_err = []
        self.time = []
        self.wavelength = []

        self.mean_time = []
        self.mean = []
        self.mean_wavelegth = []

        self.diff = []

    def load_data_bin(self, number_bin):
        """
        Load the light-curves for the specific wavelength.
        """
        self.y = []
        self.y_err = []
        self.time = []
        self.wavelength = []

        for i in range(len(self.sn_name)):

            number_points = len(self.lds.spectra[self.sn_name[i]].keys())

            y = np.zeros(number_points)
            y_err = np.zeros(number_points)
            time = np.zeros(number_points)
            wavelength = np.zeros(number_points)

            for j in range(number_points):
                y[j] = self.lds.spectra[self.sn_name[i]]['%i'%j][number_bin]
                y_err[j] = np.sqrt(self.lds.spectra_variance[self.sn_name[i]]['%i'%j])[number_bin]
                time[j] = self.lds.spectra_phases[self.sn_name[i]]['%i'%j]
                wavelength[j] = self.lds.spectra_wavelength[self.sn_name[i]]['%i'%j][number_bin]

            self.y.append(y)
            self.y_err.append(y_err)
            self.time.append(time)
            self.wavelength.append(wavelength)


    def load_mean_bin(self, number_bin, mean_file=None):
        """
        Load the light curve average for the specific wavelength.
        """
        self.mean_time = []
        self.mean = []
        self.mean_wavelegth = []

        if mean_file is None:
            path = os.path.dirname(sugar.__file__)
            mean_file = path + '/data_input/mean_gaussian_process.dat'

        nnumber_bin = len(self.lds.spectra_wavelength[self.sn_name[0]]['0'])

        data = np.loadtxt(mean_file)
        number_points = len(data[:,0]) / nnumber_bin
        y = np.zeros(number_points)
        time = np.zeros(number_points)
        wavelength = np.zeros(number_points)

        y = data[number_points * number_bin: (number_points * number_bin) + number_points, 2]
        time = data[number_points * number_bin: (number_points * number_bin) + number_points, 0]
        wavelength = data[number_points * number_bin: (number_points * number_bin) + number_points, 1]

        self.mean = y
        self.mean_time = time
        self.mean_wavelength = wavelength


    def build_difference_mean(self, mean_file=None):
        """
        Compute systematique difference between average and data.
        """
        self.diff = np.zeros(len(self.sn_name))

        if mean_file is None:
            path = os.path.dirname(sugar.__file__)
            mean_file = path + '/data_input/mean_gaussian_process.dat'

        data = np.loadtxt(mean_file)

        delta_mean = 0
        delta_lambda = 0

        for i in range(len(data[:, 0])):

            if data[:, 0][0] == data[:, 0][i]:
                delta_lambda +=1
            if data[:, 1][0] == data[:, 1][i]:
                delta_mean +=1

        self.load_data_bin(0)
        self.load_mean_bin(0)

        for i in range(len(self.sn_name)):

            phase = self.time[i]
            delta = len(phase)

            spectra = np.zeros(delta * delta_lambda)

            for t in range(len(self.lds.spectra[self.sn_name[i]].keys())):
                spectra[t * delta_lambda: (t+1) * delta_lambda] = self.lds.spectra[self.sn_name[i]]['%i'%t]

            mean_new_binning = np.zeros(delta_lambda * len(phase))

            for Bin in range(delta_lambda):
                interpolate_mean = cosmogp.mean.interpolate_mean_1d(self.mean_time,
                                                               data[:, 2][Bin * delta_mean: (Bin + 1) * delta_mean],
                                                               phase)
                mean_new_binning[Bin * delta: (Bin+1) * delta] = interpolate_mean

            reorder = np.arange(delta_lambda * delta).reshape(delta_lambda, delta).T.reshape(-1)
            mean_new_binning = mean_new_binning[reorder]

            self.diff[i] = np.mean(spectra-mean_new_binning)


class gp_sed:
    """Interpolate snia sed using gaussian process."""
    def __init__(self, grid_interpolation=np.linspace(-12,42,19), svd_method=False):
        """
        Inteporlation of sed with gaussian process.

        Will interpolate using cosmogp and a
        squared exponential kernel. It will fit
        hyperparameters in terms of wavelength by
        assuming no correlation in wavelength.
        """
        self.grid_interpolation = grid_interpolation
        self.svd_method = svd_method

        self.ldbg = load_data_bin_gp()
        self.ldbg.build_difference_mean()
        self.diff = self.ldbg.diff
        self.sigma = np.zeros(190)
        self.l = np.zeros(190)

    def gaussian_process_regression(self):
        """
        Fit hyperparameter and interpolation for the full sed.

        compute hyperparameter and full interpolation.
        compute covariance matrix also. 
        """
        for i in range(190):

            print i+1

            self.ldbg.load_data_bin(i)
            self.ldbg.load_mean_bin(i)

            gpr = cosmogp.gaussian_process_nobject(self.ldbg.y, self.ldbg.time, kernel='RBF1D',
                                                   y_err=self.ldbg.y_err, diff=self.diff, Mean_Y=self.ldbg.mean,
                                                   Time_mean=self.ldbg.mean_time, substract_mean=False)

            gpr.nugget = 0.03
            gpr.find_hyperparameters(hyperparameter_guess=[0.5,8.], nugget=False, svd_method=self.svd_method)
            gpr.get_prediction(new_binning=self.grid_interpolation, COV=True, svd_method=self.svd_method)
            self.sigma[i] = gpr.hyperparameters[0]
            self.l[i] = gpr.hyperparameters[1]

            #BP= build_pull(LCMC.Y,LCMC.Y_err,LCMC.TIME,LCMC.Time_Mean,LCMC.Mean,GP.hyperparameters['sigma'],GP.hyperparameters['l'])
            #BP.compute_pull(diFF=LCMC.difference)

            #dic={'Prediction':GP.Prediction,
            #     'Covariance':GP.covariance_matrix,
            #     'Hyperparametre':GP.hyperparameters,
            #     'Hyperparametre_cov':GP.hyperparameters_Covariance,
            #     'Time':GP.new_binning,
            #     'wavelength':N.mean(LCMC.wavelength[0]),
            #     'pull_per_supernovae':BP.pull,
            #     'global_pull':BP.PULL,
            #     'Moyenne_pull':BP.Moyenne_pull,
            #     'ecart_tupe_pull':BP.ecart_type_pull,
            #     'sn_name':LCMC.sn_name}


if __name__=="__main__":

    import time
    gp = gp_sed()
    A = time.time()
    gp.gaussian_process_regression()
    B = time.time()
