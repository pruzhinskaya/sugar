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


    def load_mean_bin(self, number_bin, hsiao_empca=True):
        """
        Load the light curve average for the specific wavelength.
        """
        self.mean_time = []
        self.mean = []
        self.mean_wavelegth = []

        nnumber_bin = len(self.lds.spectra_wavelength[self.sn_name[0]]['0'])
        path = os.path.dirname(sugar.__file__)
        
        if hsiao_empca:
            mean_file = path + '/data_input/mean_gaussian_process.dat'

            data = np.loadtxt(mean_file)
            number_points = len(data[:,0]) / nnumber_bin
            y = np.zeros(number_points)
            time = np.zeros(number_points)
            wavelength = np.zeros(number_points)

            y = data[number_points * number_bin: (number_points * number_bin) + number_points, 2]
            time = data[number_points * number_bin: (number_points * number_bin) + number_points, 0]
            wavelength = data[number_points * number_bin: (number_points * number_bin) + number_points, 1]
        else:
            mean_file = cPickle.load(open(path + '/data_output/gaussian_process/mean_sed_snia_from_gaussian_process.pkl'))
            y = mean_file['bin%i'%(number_bin)]['mean']
            time = mean_file['bin%i'%(number_bin)]['time']
            wavelength = np.ones(len(y)) * mean_file['bin%i'%(number_bin)]['wavelength']
            
        self.mean = y
        self.mean_time = time
        self.mean_wavelength = wavelength


    def build_difference_mean(self, hsiao_empca=True):
        """
        Compute systematique difference between average and data.
        """
        self.diff = np.zeros(len(self.sn_name))
        path = os.path.dirname(sugar.__file__)
        
        if hsiao_empca:
            mean_file = path + '/data_input/mean_gaussian_process.dat'
            data = np.loadtxt(mean_file)
        else:
            data = cPickle.load(open(path + '/data_output/gaussian_process/mean_sed_snia_from_gaussian_process.pkl'))
            
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
                if hsiao_empca:
                    interpolate_mean = cosmogp.mean.interpolate_mean_1d(self.mean_time,
                                                                        data[:, 2][Bin * delta_mean: (Bin + 1) * delta_mean],
                                                                        phase)
                else:
                    interpolate_mean = cosmogp.mean.interpolate_mean_1d(data['bin%i'%(Bin)]['time'],
                                                                        data['bin%i'%(Bin)]['mean'],
                                                                        phase)
                mean_new_binning[Bin * delta: (Bin+1) * delta] = interpolate_mean

            reorder = np.arange(delta_lambda * delta).reshape(delta_lambda, delta).T.reshape(-1)
            mean_new_binning = mean_new_binning[reorder]

            self.diff[i] = np.mean(spectra-mean_new_binning)


class gp_sed:
    """Interpolate snia sed using gaussian process."""
    def __init__(self, grid_interpolation=np.linspace(-12,42,19),
                 svd_method=False, hsiao_empca=False):
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
        self.sn_name = self.ldbg.sn_name

        self.wavelength = self.ldbg.wavelength

        self.diff = self.ldbg.diff

        self.hsiao_empca = hsiao_empca
        
        self.sigma = np.zeros(len(self.wavelength))
        self.l = np.zeros(len(self.wavelength))

        self.pull_std = np.zeros(len(self.wavelength))
        self.pull_average = np.zeros(len(self.wavelength))
        self.pull_number_point = np.zeros(len(self.wavelength))

        self.dic = {}

        for sn in range(len(self.sn_name)):
            self.dic.update({self.sn_name[sn]:{}})
            for wave in range(len(self.wavelength)):
                self.dic[self.sn_name[sn]].update({'bin%i'%(wave):{}})

        self.new_mean_gp = {}
                
    def gaussian_process_regression(self):
        """
        Fit hyperparameter and interpolation for the full sed.

        compute hyperparameter and full interpolation.
        compute covariance matrix also. 
        """
        for i in range(len(self.wavelength)):

            print i+1,'/',len(self.wavelength)

            self.ldbg.load_data_bin(i)
            self.ldbg.load_mean_bin(i,hsiao_empca=self.hsiao_empca)

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

            std = np.zeros((len(self.sn_name),len(self.grid_interpolation)))
            
            for sn in range(len(self.sn_name)):

                dic = {'prediction': gpr.Prediction[sn],
                       'covariance': gpr.covariance_matrix[sn],
                       'time': gpr.new_binning,
                       'wavelength':self.wavelength[i]}
                self.dic[self.sn_name[sn]]['bin%i'%(i)].update(dic)
                std[sn] = np.sqrt(np.diag(gpr.covariance_matrix[sn]))

            self.new_mean_gp.update({'bin%i'%(i):{'mean':np.average(gpr.Prediction,weights=1./std**2,axis=0),
                                                  'time':gpr.new_binning,
                                                  'wavelength':self.wavelength[i]}})

    def write_output(self):
        """
        write output from gp interpolation.
        """
        path = os.path.dirname(sugar.__file__)
        output_directory = path + '/data_output/gaussian_process/'

        fichier = open(output_directory+'sed_snia_gaussian_process.pkl','w')
        cPickle.dump(self.dic, fichier)
        fichier.close()

        mean_file = open(output_directory+'mean_sed_snia_from_gaussian_process.pkl','w')
        cPickle.dump(self.new_mean_gp, mean_file)
        mean_file.close()

        gp_files = open(output_directory+'gp_info.dat','w')
        gp_files.write('#wavelength kernel_amplitude correlation_length pull_average pull_std pull_number_of_points \n')

        for wave in range(len(self.wavelength)):
            gp_files.write('%.5f %.5f %.5f %.5f %.5f %.5f \n'%((self.wavelength[wave],self.sigma[wave],self.l[wave],
                                                                self.pull_average[wave],self.pull_std[wave],self.pull_number_point[wave])))

        gp_files.close()

    def write_output_slow(self):
        path = os.path.dirname(sugar.__file__)

        for sn in range(len(self.sn_name)):
            print sn+1,'/',len(self.sn_name)
            fichier=open(path+'/data_output/gaussian_process/gp_predict/'+self.sn_name[sn]+'.predict','w')
            number_bin = len(self.dic[self.sn_name[sn]].keys())
            for Bin in range(number_bin):
                dic = self.dic[self.sn_name[sn]]['bin%i'%(Bin)]
                TIME = dic['time']
                for t in range(len(TIME)):
                    fichier.write('%.5f    %.5f    %.5f'%((TIME[t],dic['wavelength'],dic['prediction'][t])))
                    for tt in range(len(TIME)):
                        if tt == (len(TIME)-1):
                            fichier.write('    %.5f \n'%(dic['covariance'][t,tt]))
                        else:
                            fichier.write('    %.5f'%(dic['covariance'][t,tt]))
        fichier.close()

if __name__=="__main__":

    import time
    A = time.time()
    gp = gp_sed(hsiao_empca=True)
    gp.gaussian_process_regression()
    gp.write_output()
    gp.write_output_slow()
    #for i in range(5):
    #    del gp
    #    gp = gp_sed(hsiao_empca=False)
    #    gp.gaussian_process_regression()
    #    gp.write_output()
    B = time.time()
