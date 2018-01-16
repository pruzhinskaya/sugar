"""Fit Av, Rv paramers and associated error for Rv."""

import numpy as np
import cPickle
import copy
import os
import sugar
from sugar.extinction import CCMextinctionParameters as ccm_param

class load_data_to_build_sed_max:

    def __init__(self, path_input='data_input', path_output='data_output/', filtre=True):

        self.path_output = path_output
        self.path_input = path_input

        dicpca = cPickle.load(open(os.path.join(self.path_output,'emfa_3_sigma_clipping.pkl')))
        pca_sn_name = np.array(dicpca['sn_name'])

        if filtre:
            FILTRE = dicpca['filter']
        else:
            FILTRE = np.array([True]*len(pca_sn_name))

        self.pca_error = dicpca['error'][FILTRE]
        self.pca_data = dicpca['data'][FILTRE]
        self.pca_val = dicpca['val']
        self.pca_vec = dicpca['vec']
        self.pca_norm = dicpca['norm']
        self.pca_Norm_data = dicpca['Norm_data'][FILTRE]
        self.pca_Norm_err = dicpca['Norm_err'][FILTRE]
        self.sn_name = pca_sn_name[FILTRE]
        self.FILTRE = FILTRE

    def compute_EM_PCA_data(self, number_eigenvector=3):

        dat = self.pca_Norm_data
        err = self.pca_Norm_err

        new_base = sugar.passage(dat,err,self.pca_vec,sub_space=number_eigenvector)
        cov_new_err = sugar.passage_error(err,self.pca_vec,number_eigenvector)

        self.x = new_base
        self.covx = cov_new_err
        
    def load_spectra_max(self):

        lds = sugar.load_data_sugar(path_input=self.path_input)
        lds.load_spectra_at_max()
        lds.load_salt2_data()

        self.FILTRE = np.array([True]*len(lds.sn_name))
        for sn in range(len(lds.sn_name)):
            if lds.sn_name[sn] not in self.sn_name:
                self.FILTRE[sn] = False
        self.y = lds.spectra_at_max[self.FILTRE]
        self.var = lds.spectra_at_max_variance[self.FILTRE]
        self.wavelength = lds.spectra_at_max_wavelength[0]
        self.covy = np.zeros((len(self.var),len(self.wavelength),len(self.wavelength)))
        zcmb = lds.zcmb[self.FILTRE]
        zerr = lds.zerr[self.FILTRE]

        for sn in range(len(self.covy)):
            dmz = (5./np.log(10)) * np.sqrt(zerr[sn]**2 + 0.001**2) / zcmb[sn]
            self.covy[sn] = np.eye(len(self.wavelength))*self.var[sn] + dmz**2 *np.ones((len(self.wavelength),len(self.wavelength)))

class rv_fitting(sugar.sugar_fitting):
    
    def __init__(self, x, y, covx, covy,
                 wavelength, fit_disp_matrix=False,
                 control=False):

        sugar.sugar_fitting.__init__(self, x, y, covx, covy,
                                     wavelength, size_bloc=None,
                                     fit_grey=False, fit_gamma=True,
                                     fit_disp_matrix=fit_disp_matrix,
                                     sparse=False, control=control)

        self.av = np.zeros(self.nsn)
        self.inv_rv = 1./3.1
        self.rv = None
        self.rv_error = None
        self.rv_jack = np.zeros(self.nsn)

    def comp_rv(self):

        a, b = ccm_param(self.wavelength)
        self.separate_component()
        gamma = copy.deepcopy(self.gamma_lambda)
        
        up = np.sum(gamma*b) - np.sum(a*b) + np.sum(b*gamma) - np.sum(b*a)
        down = 2 * np.sum(b*b)
        self.inv_rv = up / down

    def comp_av(self):

        a, b = ccm_param(self.wavelength)
        S = a + b*self.inv_rv

        av = np.zeros(self.nsn)

        Filtre = np.array([True]*(self.ncomp+1))
        Filtre[(1+self.grey)] = False

        for sn in range(self.nsn):

            residual = self.y[sn] - np.dot(self.A[:,Filtre], np.matrix(self.h[sn,Filtre]).T).T 
            up = ((np.dot(np.matrix(residual), np.dot(self.wy[sn], np.matrix(S).T))) + (np.dot(np.matrix(S), np.dot(self.wy[sn], np.matrix(residual).T))))
            down = 2.*(np.dot(np.matrix(S), np.dot(self.wy[sn], np.matrix(S).T)))
            av = up[0]/down[0]

        self.av = av

    def jacknife(self):

        h_copy = copy.deepcopy(self.h)
        x_copy = copy.deepcopy(self.x)
        y_copy = copy.deepcopy(self.y)
        wx_copy = copy.deepcopy(self.wx)
        wy_copy = copy.deepcopy(self.wy)
        nsn_copy = copy.deepcopy(self.nsn)
        A_copy = copy.deepcopy(self.A)
        self.fit_disp_matrix = False
        
        self.nsn -= 1 
        print 'START JACK'
        for sn in range(nsn_copy):
            filtre = np.array([True]*nsn_copy)
            filtre[sn] = False
            self.h = copy.deepcopy(h_copy[filtre])
            self.x = copy.deepcopy(x_copy[filtre])
            self.y = copy.deepcopy(y_copy[filtre])
            self.wx = copy.deepcopy(wx_copy[filtre])
            self.wy = copy.deepcopy(np.array(wy_copy)[filtre])
            self.A = copy.deepcopy(A_copy)

            self.separate_component()
            self.merge_component()

            self.run_fit(maxiter=1)
            self.comp_rv()
            self.rv_jack[sn] = 1./self.inv_rv


    def find_rv(self, jacknife=True):

        #self.init_fit()
        self.run_fit()
        self.comp_av()
        self.comp_rv()
        self.rv = 1./self.inv_rv

        if jacknife:
            self.jacknife()


class make_sed_max(load_data_to_build_sed_max):

    def __init__(self, npca=3, path_input='data_input/', path_output='data_output/', filtre=True):

        load_data_to_build_sed_max.__init__(self, path_input=path_input, path_output=path_output, filtre=filtre)
        self.compute_EM_PCA_data(number_eigenvector=npca)
        self.load_spectra_max()
        
    def launch_rv_fitting(self,jack=True):

        rv_fit = rv_fitting(self.x, self.y, self.covx, self.covy,
                            self.wavelength, fit_disp_matrix=True,
                            control=False)
        self.rv_fit = rv_fit
        rv_fit.find_rv(jacknife=jack)
        self.rv_fit = rv_fit

if __name__ == '__main__':

    msm = make_sed_max(npca=3)
    msm.launch_rv_fitting(jack=True)
