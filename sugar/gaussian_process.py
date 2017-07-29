"""interpolate sed of snia using gaussian process."""

import numpy as np
import cosmogp
import sugar
import cPickle
import os


class load_data_bin_gp:
    """load data for gp for one wavelength bin."""
    def __init__(self):
        """
        create data used for gp for a given bin.
        """
        self.lds = sugar.load_data_sugar()
        self.lds.load_spectra()

        self.wavelength = self.lds.spectra_wavelength[self.lds.sn_name[0]]['0']

        self.sn_name = self.lds.sn_name

        self.y = []
        self.y_err = []
        self.time = []

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

            self.y.append(y)
            self.y_err.append(y_err)
            self.time.append(time)


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

        self.wavelength = self.ldbg.wavelength
        
        self.diff = self.ldbg.diff
        
        self.sigma = np.zeros(len(self.wavelength))
        self.l = np.zeros(len(self.wavelength))

        self.pull_std = np.zeros(len(self.wavelength))
        self.pull_average = np.zeros(len(self.wavelength))
        self.pull_number_point = np.zeros(len(self.wavelength))

    def gaussian_process_regression(self):
        """
        Fit hyperparameter and interpolation for the full sed.

        compute hyperparameter and full interpolation.
        compute covariance matrix also. 
        """
        path = os.path.dirname(sugar.__file__)
        output_directory = path + '/data_output/gaussian_process/'
        
        for i in range(len(self.wavelength)):

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

            bp = cosmogp.build_pull(self.ldbg.y,self.ldbg.time,gpr.hyperparameters,
                                    y_err=self.ldbg.y_err, nugget=0.03, y_mean=self.ldbg.mean,
                                    x_axis_mean=self.ldbg.mean_time, kernel='RBF1D')
            bp.compute_pull(diff=self.diff, svd_method=False)

            self.pull_average[i] = bp.pull_average
            self.pull_std[i] = bp.pull_std
            self.pull_number_point[i] = len(bp.pull)

            filegp = open(output_directory+'prediction_bin_%i.pkl'%(i),'w')
            
            dic = {'prediction': gpr.Prediction,
                   'covariance': gpr.covariance_matrix,
                   'time': gpr.new_binning,
                   'wavelength':self.wavelength[i],
                   'sn_name':self.ldbg.sn_name}

            cPickle.dump(dic, filegp)
            filegp.close()
            del dic

    def write_output(self):
        """
        write output from gp interpolation.
        """
        path = os.path.dirname(sugar.__file__)
        output_directory = path + '/data_output/gaussian_process/'

        for sn in range(len(self.ldbg.sn_name)): 

            dic = cPickle.load(open(output_directory+'prediction_bin_0.pkl'))
            fichier = open(output_directory+self.ldbg.sn_name[sn]+'.predict','w')

            if type(dic['time']) == list:
                time = dic['time'][sn]
            else:
                time = dic['time']

            for Bin in range(len(self.wavelength)):
                print Bin
                dic=cPickle.load(open(output_directory + 'prediction_bin_%i.pkl'%(Bin)))
                for t in range(len(time)):
                    fichier.write('%.5f    %.5f    %.5f'%((time[t], dic['wavelength'], dic['prediction'][sn][t])))
                    for tt in range(len(time)):
                        if tt == (len(time)-1):
                            fichier.write('    %.5f \n'%(dic['covariance'][sn][t, tt]))
                        else:
                            fichier.write('    %.5f'%(dic['covariance'][sn][t, tt]))
            fichier.close()

        os.system('rm '+output_directory.replace(' ','%s '%('\\'))+ '*.pkl')
            


if __name__=="__main__":

    import time
    gp = gp_sed()
    A = time.time()
    gp.gaussian_process_regression()
    gp.write_output()
    B = time.time()
