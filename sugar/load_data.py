"""script that load the data used to train sugar. """

import numpy as np
import cPickle
import sugar
import os

path = os.path.dirname(sugar.__file__)

class load_data_sugar:
    """function that allowed to access to observed sed and spectral features."""

    def __init__(self, file_spectra=path + '/data_input/spectra_snia.pkl',
                 file_at_max=path + '/data_input/spectra_and_si_at_max.pkl', mjd_cut=55250.):
        """
        Load all the data needed for training sugar.

        will load:

        -all spectra
        -spectra at max
        -spectal indicators at max
        """
        self.file_spectra = file_spectra
        self.file_at_max = file_at_max
        self.mjd_cut = mjd_cut

        dic = cPickle.load(open(self.file_at_max))
        self.sn_name = dic.keys()
        filter_mjd = np.array([True] * len(self.sn_name))

        for i in range(len(self.sn_name)):
            if dic[self.sn_name[i]]['spectra']['days'] > self.mjd_cut:
                filter_mjd[i] = False
            if self.sn_name[i] =='SN2005cf':
                filter_mjd[i] = False

        self.sn_name = np.array(self.sn_name)[filter_mjd]
        self.sn_name.sort()

        self.spectra = {}
        self.spectra_variance = {}
        self.spectra_wavelength = {}
        self.spectra_phases = {}

        self.spectra_at_max = []
        self.spectra_at_max_variance = []
        self.spectra_at_max_wavelength = []
        self.spectra_at_max_phases = []

        self.spectral_indicators = []
        self.spectral_indicators_error = []

        self.X0 = []
        self.X1 = []
        self.C = []
        self.mb = []
        self.zhelio = []
        self.zcmb = []
        self.zerr = []
        self.mb_err = []
        self.X0_err = []
        self.X1_err = []
        self.C_err = []
        self.C_mb_cov = []
        self.X1_C_cov = []
        self.X1_mb_cov = []

    def load_spectra(self, file_spectra=None):
        """
        Load time sed of snia.

        will load spectra and all other infomation needed

        -spectra
        -spectra variance
        -spectra wavelength
        -spectra phases respective to salt2.4 phases
        """
        if file_spectra is None:
            file_spectra = self.file_spectra

        dic = cPickle.load(open(file_spectra))

        for i in range(len(self.sn_name)):

            self.spectra.update({self.sn_name[i]: {}})
            self.spectra_variance.update({self.sn_name[i]: {}})
            self.spectra_wavelength.update({self.sn_name[i]: {}})
            self.spectra_phases.update({self.sn_name[i]: {}})

            for t in range(len(dic[self.sn_name[i]].keys())):
                if dic[self.sn_name[i]]['%i'%t]['phase_salt2'] < 50.:
                    self.spectra[self.sn_name[i]].update({'%i'%t: dic[self.sn_name[i]]['%i'%t]['Y']})
                    self.spectra_variance[self.sn_name[i]].update({'%i'%t: dic[self.sn_name[i]]['%i'%t]['V']})
                    self.spectra_wavelength[self.sn_name[i]].update({'%i'%t: dic[self.sn_name[i]]['%i'%t]['X']})
                    self.spectra_phases[self.sn_name[i]].update({'%i'%t: dic[self.sn_name[i]]['%i'%t]['phase_salt2']})

    def load_spectra_at_max(self, file_at_max=None):
        """
        Load sed at max of snia.

        will load spectra at max and all other infomation needed

        -spectra
        -spectra variance
        -spectra wavelength
        -spectra phases respective to salt2.4 phases
        """
        if file_at_max is None:
            file_at_max = self.file_at_max

        dic = cPickle.load(open(file_at_max))

        for i in range(len(self.sn_name)):

            self.spectra_at_max.append(dic[self.sn_name[i]]['spectra']['Y'])
            self.spectra_at_max_variance.append(dic[self.sn_name[i]]['spectra']['V'])
            self.spectra_at_max_wavelength.append(dic[self.sn_name[i]]['spectra']['X'])
            self.spectra_at_max_phases.append(dic[self.sn_name[i]]['spectra']['phase_salt2'])

        self.spectra_at_max = np.array(self.spectra_at_max)
        self.spectra_at_max_variance = np.array(self.spectra_at_max_variance)
        self.spectra_at_max_wavelength = np.array(self.spectra_at_max_wavelength)
        self.spectra_at_max_phases = np.array(self.spectra_at_max_phases)


    def load_spectral_indicator_at_max(self, file_at_max=None, missing_data=True):
        """
        Load spectral indicators at max of snia.

        will load spectra features at max and all other infomation needed

        -spectral indicator
        -spectral indicator error
        """
        if file_at_max is None:
            file_at_max = self.file_at_max

        dic = cPickle.load(open(file_at_max))

        number_si = len(dic[self.sn_name[0]]['spectral_indicators'])

        indicator_data = np.zeros((len(self.sn_name), number_si))
        indicator_error = np.zeros((len(self.sn_name), number_si))

        for i in range(len(self.sn_name)):
            indicator_data[i] = dic[self.sn_name[i]]['spectral_indicators']
            indicator_error[i] = dic[self.sn_name[i]]['spectral_indicators_error']

        if missing_data:
            error = indicator_error[:, np.all(np.isfinite(indicator_data), axis=0)]
            data = indicator_data[:, np.all(np.isfinite(indicator_data), axis=0)]

        self.spectral_indicators = indicator_data
        self.spectral_indicators_error = indicator_error

        for sn in range(len(self.sn_name)):
            for si in range(number_si):
                if not np.isfinite(self.spectral_indicators[sn, si]):
                    self.spectral_indicators[sn, si] = np.average(data[:, si], weights=1. / error[:, si]**2)
                    self.spectral_indicators_error[sn, si] = 10**8

    def load_salt2_data(self):

        meta = cPickle.load(open(path+'/data_input/META_JLA.pkl'))

        for i in range(len(self.sn_name)):
            self.X0.append(meta[self.sn_name[i]]['salt2.X0'])
            self.X1.append(meta[self.sn_name[i]]['salt2.X1'])
            self.C.append(meta[self.sn_name[i]]['salt2.Color'])
            self.mb.append(meta[self.sn_name[i]]['salt2.RestFrameMag_0_B'])
            self.zhelio.append(meta[self.sn_name[i]]['host.zhelio'])
            self.zcmb.append(meta[self.sn_name[i]]['host.zcmb'])
            self.zerr.append(meta[self.sn_name[i]]['host.zhelio.err'])
            self.mb_err.append(meta[self.sn_name[i]]['salt2.RestFrameMag_0_B.err'])
            self.X0_err.append(meta[self.sn_name[i]]['salt2.X0.err'])
            self.X1_err.append(meta[self.sn_name[i]]['salt2.X1'])
            self.C_err.append(meta[self.sn_name[i]]['salt2.Color.err'])
            self.C_mb_cov.append(meta[self.sn_name[i]]['salt2.CovColorRestFrameMag_0_B'])
            self.X1_C_cov.append(meta[self.sn_name[i]]['salt2.CovColorX1'])
            self.X1_mb_cov.append(meta[self.sn_name[i]]['salt2.CovRestFrameMag_0_BX1'])

        self.X0 = np.array(self.X0)
        self.X1 = np.array(self.X1)
        self.C = np.array(self.C)
        self.mb = np.array(self.mb)
        self.zhelio = np.array(self.zhelio)
        self.zcmb = np.array(self.zcmb)
        self.zerr = np.array(self.zerr)
        self.mb_err = np.array(self.mb_err)
        self.X0_err = np.array(self.X0_err)
        self.X1_err = np.array(self.X1_err)
        self.C_err = np.array(self.C_err)
        self.C_mb_cov = np.array(self.C_mb_cov)
        self.X1_C_cov = np.array(self.X1_C_cov)
        self.X1_mb_cov = np.array(self.X1_mb_cov)
                                                                                        

if __name__=='__main__':

    lds = load_data_sugar()
    lds.load_spectra()
    lds.load_spectra_at_max()
    lds.load_spectral_indicator_at_max()
    lds.load_salt2_data()
