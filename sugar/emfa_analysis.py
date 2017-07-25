"""Realize emfa on spectrale features space."""

import numpy as N
import copy
import emfa as EM_manu


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

    Use : 
    
    (TO FINISH)

    emfa = emfa_si_analysis(si,si_error,sn_name)
    emfa.emfa()

    """
    def __init__(self,si,si_error,sn_name,missing_data=True):

        self.number_si = len(si[0])
        
        if Missing_data:
            
            error = indicator_error[:,N.all(N.isfinite(si),axis=0)]
            data = indicator_data[:,N.all(N.isfinite(si_error),axis=0)]

            self.sn_name = sn_name
            self.error = si_error
            self.data = si

            for sn in range(len(self.sn_name)):
                for SI in range(self.number_si):
                    if not N.isfinite(self.data[sn,SI]):
                        self.data[sn,SI] = N.average(data[:,SI],weights=1./error[:,SI]**2)
                        self.error[sn,SI] = 10**8
        else:

            self.sn_name = sn_name[N.all(N.isfinite(si),axis=0)]
            self.error = indicator_error[:,N.all(N.isfinite(si),axis=0)]
            self.data = indicator_data[:,N.all(N.isfinite(si),axis=0)]
            
        self.filter = N.array([True]*len(self.data))

    def _center(self,wnorm=True):

        if wnorm:
            center = N.average(self.data[self.filter].T,axis=1,weights=1./self.error[self.filter].T**2)
        else:
            center = N.mean(self.data[self.filter].T,axis=1) 
        self.data_center = ( self.data - center.T )
        self.error_center =  self.error
        self.Covar = 1./(N.sum(self.filter)-1) * N.dot(self.data_center[self.filter].T,self.data_center[self.filter])
        

    def _Norm_varr(self,Filter=None):
        if Filter != None :
            norm = N.sqrt(N.var(self.data_center[Filter].T,axis=1))
        else:
            norm = N.sqrt(N.var(self.data_center.T,axis=1))

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
            
            emfa = EM_manu.EMPCA(dat,1./err**2)
            emfa.converge(13,niter=150,center=True)
            new_varr = N.dot(emfa.Lambda,emfa.Lambda.T)
            chi2 = N.zeros(len(self.data_center)) 
        
            for sn in range(len(self.data_center)):
                chi2[sn] = N.dot(N.dot(DAT[sn],N.linalg.inv(new_varr+N.diag(ERR[sn]**2))),DAT[sn].reshape((len(DAT[sn]),1)))[0]
                self.chi2_empca = chi2
               
        else:
            Covar = 1./(N.sum(self.filter)-1) * N.dot(dat.T,dat)
            chi2 = N.diag(N.dot(N.dot(DAT,N.linalg.inv(Covar)),DAT.T))
            self.chi2_pca = chi2
        
        ndf = len(self.data_center[0])
        
        self.filter = self.filter & (chi2<ndf + 3 * N.sqrt(2*ndf))

        self._center()

        

    def _iterative_filter(self,chi2emfa=True):

        a = [sum(self.filter)+1,sum(self.filter)+2]
        i = 0
        
        while a[i+1] != a[i]:
            self._filter_and_prepare_data(chi2emfa=chi2emfa)
 
            a.append(sum(self.filter))
            i += 1

            
    def _bic_number_eigenvector(self):

        self.BIC = N.zeros(len(self.data_center[0]))
        self.Log_L = N.zeros(len(self.data_center[0]))
        self.chi2_Tot = N.zeros(len(self.data_center[0]))

        dat = (self.data_center[self.filter]/self._Norm_varr(self.filter))
        err = (self.error_center[self.filter]/self._Norm_varr(self.filter))

        for i in range(len(self.data_center[0])):
            EMPCA = EM_manu.EMPCA(dat,1./err**2)
            EMPCA.converge(i+1,niter=500)
            self.Log_L[i],self.chi2_Tot[i]=EMPCA.log_likelihood(z_solved=True)

            #self.BIC[i]=-2*self.Log_L[i]+(sum(self.filter)+13+((i+1)*(2*13-i-1+1))/2.)*N.log(sum(self.filter))


    def _em_fa(self,pkl_file,bic=False):
        
        dat = (self.data_center[self.filter]/self._Norm_varr(self.filter))
        err = (self.error_center[self.filter]/self._Norm_varr(self.filter))

        emfa = EM_manu.EMPCA(dat,1./err**2)
        emfa.converge(len(self.data_center[0]),niter=500)
        varr = N.dot(emfa.Lambda,emfa.Lambda.T)

        val,vec = N.linalg.eig(varr)
        self.EM_FA_Cov = varr
        self.val = val.real
        self.vec = vec.real

        self.Norm_data = (self.data_center/self._Norm_varr(self.filter))
        self.Norm_err = (self.error_center/self._Norm_varr(self.filter))

        if bic:
            self._bic_number_eigenvector()
            
        toto=self.__dict__
        File=open(pkl_file,'w')
        cPickle.dump(toto,File)
        File.close()


    def emfa(self,emfa_output_pkl,chi2emfa=True,bic=False):

        self._center()
        self._iterative_filter(chi2emfa=True)
        self._em_fa(emfa_output_pkl,bic=bic)



