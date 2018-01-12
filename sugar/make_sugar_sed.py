"""compute the sugar model."""

from scipy.sparse import block_diag
from scipy.sparse import coo_matrix
from sugar.multilinearfit import *
import numpy as np
import sugar
import cPickle
import copy
import os

class load_data_to_build_sugar:

    def __init__(self, path_output='data_output/', path_output_gp='data_output/gaussian_process/', filtre=True):

        self.path_output = path_output
        self.path_output_gp = path_output_gp

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

        dic_model=cPickle.load(open(os.path.join(self.path_output,'sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl')))
        self.sn_name_Av = dic_model['sn_name']
        self.Av = dic_model['Av_cardelli']
        self.Rv = dic_model['RV']
        
        self.rep_GP = self.path_output_gp

        self.N_sn = len(self.sn_name)

        self.number_bin_phase = 0
        self.number_bin_wavelength = 0
        A=np.loadtxt(self.rep_GP+self.sn_name[0]+'.predict')
        phase = A[:,0]
        wavelength = A[:,1]
        self.TX = wavelength
        for i in range(len(wavelength)):
            if wavelength[i]==wavelength[0]:
                self.number_bin_phase += 1

        self.number_bin_wavelength = len(wavelength)/self.number_bin_phase

    def compute_EM_PCA_data(self,number_eigenvector=3):

        dat = self.pca_Norm_data
        err = self.pca_Norm_err

        new_base = sugar.passage(dat,err,self.pca_vec,sub_space=number_eigenvector)
        cov_new_err = sugar.passage_error(err,self.pca_vec,number_eigenvector)

        self.data = new_base
        self.Cov_error = cov_new_err

    def load_spectra_GP(self,sn_name):

        A = np.loadtxt(self.rep_GP+sn_name+'.predict')
        Y = A[:,2]

        for j,sn_av in enumerate(self.sn_name_Av):
            if sn_name == sn_av:
                Y_cardelli_corrected = (Y-(self.Av[j]*sugar.extinctionLaw(A[:,1],Rv=self.Rv)))
        return Y,Y_cardelli_corrected


    def load_phase_wavelength(self,sn_name):
        
        A = np.loadtxt(self.rep_GP+sn_name+'.predict')
        phase = A[:,0]
        wavelength = A[:,1]
        del A
        
        return phase,wavelength


    def load_cov_matrix(self,sn_name):

        A = np.loadtxt(self.rep_GP+sn_name+'.predict')
        size_matrix = self.number_bin_phase*self.number_bin_wavelength
        COV = np.zeros((size_matrix,size_matrix))

        for i in range(self.number_bin_wavelength):
            cov = A[(i*self.number_bin_phase):((i+1)*self.number_bin_phase),3:]
            COV[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase] = cov

        return COV
        
    def load_spectra(self):

        self.phase,self.X = self.load_phase_wavelength(self.sn_name[0])
        self.Y_cardelli_corrected_cosmo_corrected = np.zeros((len(self.sn_name),len(self.X)))
        self.Y_cosmo_corrected = np.zeros((len(self.sn_name),len(self.X)))
        self.CovY = []
        
        for i,sn in enumerate(self.sn_name):
            print sn
            self.Y_cosmo_corrected[i], self.Y_cardelli_corrected_cosmo_corrected[i] = (self.load_spectra_GP(sn))

            Cov = self.load_cov_matrix(sn)
            COV = []
            for i in range(self.number_bin_wavelength):
                COV.append(coo_matrix(Cov[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase]))
            self.CovY.append(block_diag(COV))


class make_sugar(load_data_to_build_sugar):

    def __init__(self, path_output='data_output/', path_output_gp='data_output/gaussian_process/', filtre=True):

        load_data_to_build_sugar.__init__(self, path_output=path_output, path_output_gp=path_output_gp, filtre=filtre)
        self.compute_EM_PCA_data()
        self.load_spectra()

    def launch_sed_fitting(self):

        sedfit = sugar.sugar_fitting(self.data, self.Y_cardelli_corrected_cosmo_corrected,
                                     self.Cov_error, self.CovY,self.X,
                                     size_bloc=self.number_bin_phase,
                                     fit_grey=True, sparse=True)

        del self.data
        del self.Y_cardelli_corrected_cosmo_corrected
        del self.Cov_error
        del self.CovY

        self.sedfit = sedfit        
        self.sedfit.init_fit()
        self.sedfit.run_fit()
        self.sedfit.separate_component()
        
    def write_model_output(self):

        pkl = os.path.join(self.path_output,'sugar_model.pkl')

        dic = {'alpha':self.sedfit.alpha,
               'm0':self.sedfit.m0,
               'delta_m_grey':self.sedfit.delta_m_grey,
               'X':self.X,
               'h':self.sedfit._h,
               'sn_name':self.sn_name,
               'chi2':self.sedfit.chi2_save,
               'dof':self.sedfit.dof}
        
        fichier = open(pkl,'w')
        cPickle.dump(dic, fichier)
        fichier.close()
        

if __name__ == '__main__':

    ld = make_sugar()
    ld.launch_sed_fitting()
    ld.write_model_output()
