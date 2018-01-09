"""Realize emfa on spectrale features space."""

import numpy as np
import emfa
import sugar
import cPickle
import os

def sort_eigen(eigenval,eigenvec):
    """
    sort eigenvec and eigenval.
    """

    eigenvec_sort = np.zeros(np.shape(eigenvec))
    eigenval_sort = np.zeros(len(eigenval))
    
    eigenval_list = list(eigenval)
    eigenvec_list = list(eigenvec.T)

    for i in range(len(eigenval_sort)):

        ind_max = eigenval_list.index(max(eigenval_list))
        eigenvec_sort[:,i] = eigenvec_list[ind_max]
        eigenval_sort[i] = eigenval_list[ind_max]
        
        del eigenval_list[ind_max]
        del eigenvec_list[ind_max]

    return eigenval_sort, eigenvec_sort
                                                        

class emfa_si_analysis:
    """
    do emfa on spectral features space.

    si : 2D numpy array, array of spectrale 
    features
    
    si_err : 2D numpy array, array of spectrale 
    features errors measurments

    sn_name : 1D numpy array, array of associated 
    SNIa name

    missing_data : boolean, if True will fit with
    missing data 
    """
    def __init__(self,si,si_error,sn_name,missing_data=True):

        self.number_si = len(si[0])
        
        if missing_data:
            
            error = si_error[:,np.all(np.isfinite(si),axis=0)]
            data = si[:,np.all(np.isfinite(si_error),axis=0)]

            self.sn_name = sn_name
            self.error = si_error
            self.data = si

            for sn in range(len(self.sn_name)):
                for SI in range(self.number_si):
                    if not np.isfinite(self.data[sn,SI]):
                        self.data[sn,SI] = np.average(data[:,SI],weights=1./error[:,SI]**2)
                        self.error[sn,SI] = 10**8
        else:

            self.sn_name = sn_name[np.all(np.isfinite(si),axis=0)]
            self.error = si_error[:,np.all(np.isfinite(si),axis=0)]
            self.data = si[:,np.all(np.isfinite(si),axis=0)]
            
        self.filter = np.array([True]*len(self.data))

    def _center(self,wnorm=True):

        if wnorm:
            center = np.average(self.data[self.filter].T,axis=1,weights=1./self.error[self.filter].T**2)
        else:
            center = np.mean(self.data[self.filter].T,axis=1) 
        self.data_center = ( self.data - center.T )
        self.error_center =  self.error
        self.Covar = 1./(np.sum(self.filter)-1) * np.dot(self.data_center[self.filter].T,self.data_center[self.filter])
        

    def _Norm_varr(self,Filter=None):
        if Filter is not None :
            norm = np.sqrt(np.var(self.data_center[Filter].T,axis=1))
        else:
            norm = np.sqrt(np.var(self.data_center.T,axis=1))

        self.norm = norm

        return norm
    

    def _filter_and_prepare_data(self,chi2emfa=False):

        self._center()
        norm = self._Norm_varr(self.filter)
        
        dat = (self.data_center[self.filter] /norm)
        err = (self.error_center[self.filter]/norm)
        DAT = (self.data_center /norm)
        ERR = (self.error_center/norm)

        if chi2emfa:
            
            emfa_algo = emfa.EMPCA(dat,1./err**2)
            emfa_algo.converge(13,niter=150,center=True)
            new_varr = np.dot(emfa_algo.Lambda,emfa_algo.Lambda.T)
            chi2 = np.zeros(len(self.data_center)) 
        
            for sn in range(len(self.data_center)):
                chi2[sn] = np.dot(np.dot(DAT[sn],np.linalg.inv(new_varr+np.diag(ERR[sn]**2))),DAT[sn].reshape((len(DAT[sn]),1)))[0]
                self.chi2_empca = chi2
               
        else:
            Covar = 1./(np.sum(self.filter)-1) * np.dot(dat.T,dat)
            chi2 = np.diag(np.dot(np.dot(DAT,np.linalg.inv(Covar)),DAT.T))
            self.chi2_pca = chi2
        
        ndf = len(self.data_center[0])
        
        self.filter = self.filter & (chi2<ndf + 3 * np.sqrt(2*ndf))

        self._center()

        

    def _iterative_filter(self,chi2emfa=True):

        a = [sum(self.filter)+1,sum(self.filter)+2]
        i = 0
        
        while a[i+1] != a[i]:
            self._filter_and_prepare_data(chi2emfa=chi2emfa)
 
            a.append(sum(self.filter))
            i += 1

            
    def _bic_number_eigenvector(self):

        self.BIC = np.zeros(len(self.data_center[0]))
        self.Log_L = np.zeros(len(self.data_center[0]))
        self.chi2_Tot = np.zeros(len(self.data_center[0]))

        dat = (self.data_center[self.filter]/self._Norm_varr(self.filter))
        err = (self.error_center[self.filter]/self._Norm_varr(self.filter))

        for i in range(len(self.data_center[0])):
            emfa_algo = emfa.EMPCA(dat,1./err**2)
            emfa_algo.converge(i+1,niter=500)
            self.Log_L[i],self.chi2_Tot[i]=emfa_algo.log_likelihood(z_solved=True)

            #self.BIC[i]=-2*self.Log_L[i]+(sum(self.filter)+13+((i+1)*(2*13-i-1+1))/2.)*np.log(sum(self.filter))


    def _em_fa(self,pkl_file,bic=False):
        
        dat = (self.data_center[self.filter]/self._Norm_varr(self.filter))
        err = (self.error_center[self.filter]/self._Norm_varr(self.filter))

        emfa_algo = emfa.EMPCA(dat,1./err**2)
        emfa_algo.converge(len(self.data_center[0]),niter=500)
        varr = np.dot(emfa_algo.Lambda,emfa_algo.Lambda.T)

        val,vec = np.linalg.eig(varr)
        self.EM_FA_Cov = varr
        self.val = val.real
        self.vec = vec.real

        self.val, self.vec = sort_eigen(self.val, self.vec)

        self.Norm_data = (self.data_center/self._Norm_varr(self.filter))
        self.Norm_err = (self.error_center/self._Norm_varr(self.filter))

        if bic:
            self._bic_number_eigenvector()
            
        toto=self.__dict__
        File=open(pkl_file,'w')
        cPickle.dump(toto,File)
        File.close()


    def emfa(self,emfa_output_pkl, sigma_clipping=False,
             chi2emfa=True, bic=False):

        self._center()
        if sigma_clipping:
            self._iterative_filter(chi2emfa=True)
        self._em_fa(emfa_output_pkl,bic=bic)

def run_emfa_analysis(path_input, path_output, sigma_clipping=False):


    snia = sugar.load_data_sugar(path_input=path_input)
    snia.load_spectral_indicator_at_max()
    
    si_analysis = emfa_si_analysis(snia.spectral_indicators,
                                   snia.spectral_indicators_error,
                                   snia.sn_name,missing_data=True)

        
    output_file = os.path.join(path_output,'emfa_output.pkl')

    si_analysis.emfa(output_file,sigma_clipping=sigma_clipping,
                     chi2emfa=True,bic=False)

if __name__ == '__main__':

    run_emfa_analysis('data_input/','data_output/', sigma_clipping=True)
