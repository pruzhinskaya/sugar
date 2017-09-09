"""fit sed using ODR algorithm."""

import numpy as np
import scipy.odr as odrs
from scipy import optimize
import sugar
import copy
import os


class load_data_sed_fitting:
    """
    load data for estimating sugar sed.

    TO DO
    """

    def __init__(self):

        return None

class multilinearfit:
    """
    n-dimentional linear fitting to init sed fitting.
    
    - x: array_like

    Components witch depends y. It can be strecth and color or what do you want.
    x have N lines and n collones.
    Where N are the numbers of points/number of supernovae and n the number of components
    
    - y: array_like
    
    Data where do you want constrain a multilinear model. It can be the absolute Magnitude in B band.
    yerr have N elements
    Where N are the numbers of points/number of supernovae.
    
    - xerr: array_like
    
    Error on x.
    xerr have N lines and n collones.
    Where N are the numbers of points/number of supernovae and n the number of components
    
    - yerr: array_like
    
    Error on y.
    yerr have N elements
    Where N are the numbers of points/number of supernovae.
    
    - covx: array_like
    
    Covriance error matrix for x.
    covx have N matrix n times n.
    Where N are the numbers of points/number of supernovae and n the number of components
    
    
    - Beta00: tuple_like
    
    You can initialize the fit with Beta00
    Beta00[0]=intercept
    Beta00[1:]=slope(s)
    
    
    - To run the fit :
    
    Multi=Multilinearfit(x,y,xerr=None,yerr=None,covx=None,Beta00=None)
    Multi.Multilinearfit(adddisp=False)
    
    if you want a chi2 by degrees of freedom reduce to 1 put the option adddisp=True
    
    the result are:
    
    intercept : Multi.M0
    slope(s)  : Multi.alpha
    
    if you want RMS or wRMS of the fit:
    Multi.comp_stat()
    """
    def __init__(self,x,y,xerr=None,yerr=None,covx=None,Beta00=None):

        X=x.T
        if xerr is not None:
            xerr=xerr.T
        self.X=X
        self.y=y
        self.Beta00=Beta00

        if xerr is None:
            self.xerr = np.ones(np.shape(X))
        else:
            self.xerr = xerr
            
        if yerr is None:
            self.yerr = np.ones(len(y))
        else:
            self.yerr = np.array(yerr)
            
        if covx is None:
            self.covx=None
        else:
            self.covx = covx.T

    def _lin_fcn(self,B,x):
        a,b=B[0],B[1:]
        b.shape=(b.shape[0],1)
        return a + (x*b).sum(axis=0)

    def _lin_fjb(self,B, x):
        a = np.ones(x.shape[-1], float)
        res = np.concatenate((a, x.ravel()))
        res.shape = (B.shape[-1], x.shape[-1])
        return res

    def _lin_fjd(self,B, x):
        b = B[1:]
        b = np.repeat(b, (x.shape[-1],)*b.shape[-1],axis=0)
        b.shape = x.shape
        return b

    def _lin_est(self,data):
        if len(data.x.shape) == 2:
            m = data.x.shape[0]
        else:
            m = 1
        return np.ones((m + 1,), float)
    
    def Multilinearfit(self,adddisp=False,PRINT=False):


        Fct=lambda B,X : self._lin_fcn(B,X)
        jac=lambda B,X : self._lin_fjb(B,X)
        jacd=lambda B,X : self._lin_fjd(B,X)

        if self.Beta00 is None:
        
            if type(self.X[0])==np.float64:
                BETA0=(1,0)
            else:    
                BETA0=(len(self.X),)
                for i in range(len(self.X)):
                    BETA0+=(len(self.X)-i-1,)

        else:
            BETA0=self.Beta00
    
        if self.covx is not  None:
            dataXZ=odrs.RealData(self.X,self.y,sy=self.yerr,covx=self.covx)
        else:
            dataXZ=odrs.RealData(self.X,self.y,sx=self.xerr,sy=self.yerr)
        estXZ=self._lin_est(dataXZ)

        MODELXZ=odrs.Model(Fct, fjacb=jac,fjacd=jacd, estimate=estXZ)
        odrXZ=odrs.ODR(dataXZ,MODELXZ,beta0=BETA0,maxit=1000,sstol=0.)
        output=odrXZ.run()
        BETA1=(output.beta[0],)
        for i in range(len(output.beta)-1):
            BETA1+=(output.beta[i+1],)

        odrXZ=odrs.ODR(dataXZ,MODELXZ,beta0=BETA1,maxit=1000,sstol=0.)
        output=odrXZ.run()

        
        alpha=np.zeros(len(output.beta[1:]))
        for correction in range(len(output.beta[1:])):
            alpha[correction]=output.beta[correction+1]
        M0=output.beta[0]
        chi2_ODR=output.sum_square

        self.disp=0.
        self.alpha=alpha
        self.M0=output.beta[0]
        self.dof=len(self.y)-len(output.beta[1:])-1.

        if adddisp:
            calls=0
            self.pouet=0.
            if (chi2_ODR/(self.dof))<1.:
                if PRINT:
                    print 'ok'
            else:
                while abs((chi2_ODR/(self.dof))-1.)>0.001:
                    
                    if calls<100:
                        if PRINT:
                            print 'search of dispertion : %i'%(calls+1)
                        self._compute_dispertion()

                        if self.covx is not None:
                            dataXZ=odrs.RealData(self.X,self.y,sy=np.sqrt(self.yerr**2+self.disp**2),covx=self.covx)
                        else:
                            dataXZ=odrs.RealData(self.X,self.y,sx=self.xerr,sy=np.sqrt(self.yerr**2+self.disp**2))
                        estXZ=self._lin_est(dataXZ)

                        MODELXZ=odrs.Model(Fct, fjacb=jac,fjacd=jacd, estimate=estXZ)
                        odrXZ=odrs.ODR(dataXZ,MODELXZ,beta0=BETA0,maxit=1000,sstol=0.)
                        output=odrXZ.run()
                        
                        chi2_ODR=output.sum_square
                        for correction in range(len(output.beta[1:])):
                            self.alpha[correction]=output.beta[correction+1]
                        self.M0=output.beta[0]
                        calls+=1                        

                    else:
                        print 'error : calls limit are exceeded'
                        break
        
        y_corrected=copy.deepcopy(self.y)
        y_plus=copy.deepcopy(self.y)
        y_error_corrected=np.zeros(len(self.y))      
        
        VARx=np.dot(self.X,self.X.T)/(len(self.y)-1)
        
        self.xplus=output.xplus.T

        for sn in range(len(self.y)):
            for correction in range(len(self.alpha)):
                if len(self.alpha)==1:
                    y_corrected[sn] -=  self.alpha*self.X.T[sn] 
                    y_plus[sn] -= self.alpha*self.xplus[sn]
                else:
                    y_corrected[sn] -= self.alpha[correction]*self.X.T[sn][correction] 
                    y_plus[sn] -= self.alpha[correction]*self.xplus[sn][correction]
    
                if self.covx is not None:
                    for k in range(len(self.alpha)):
                        if correction>k :
                            continue
                        else:
                            y_error_corrected[sn] += self.alpha[correction]*self.alpha[k]*self.covx.T[sn,correction,k]
    
        self.y_plus=y_plus
        self.y_corrected=y_corrected
        self.y_error_corrected=y_error_corrected
        self.alpha=self.alpha
        self.M0=self.M0   
        self.dM=np.sqrt(self.yerr**2+self.disp**2)
        self.output=output

    def _compute_dispertion(self):
        self.disp=optimize.fmin(self._disp_function,self.disp,disp=0)[0]

    def _disp_function(self,d):
        residu=np.zeros(len(self.y))
        VAR=np.zeros(len(self.y))
        A=np.matrix(np.concatenate([[1.0],[self.alpha[i] for i in range(len(self.alpha))]]))
        Cov=np.zeros((len(self.y),len(self.alpha)+1,len(self.alpha)+1))

        for sn in range(len(self.y)):
            Cov[sn][0][0]=self.yerr[sn]**2+d**2
            if self.covx is None:
                if type(self.xerr.T[sn])!=np.float64:
                    Cov[:,1:,1:][sn]=np.diag(self.xerr.T[sn]**2)
            else:
                Cov[:,1:,1:][sn]=self.covx.T[sn]
    
            residu[sn]=self.y[sn]-self.M0
              
            for correction in range(len(self.alpha)):
                if type(self.X.T[sn])!=np.float64:
                    residu[sn]-=self.alpha[correction]*self.X.T[sn][correction]
                else:
                    residu[sn]-=self.alpha*self.X.T[sn]

            if type(self.xerr.T[sn])!=np.float64 and self.covx is None:
                VAR[sn]=self.yerr[sn]**2+d**2+self.alpha**2*self.xerr.T[sn]**2
            else:
                VAR[sn]=np.dot(A,np.dot(Cov[sn],A.T))                                       
                
        chi2 = (residu**2/VAR).sum()
        return abs((chi2/self.dof)-1.)


class sugar_fitting:
    """
    sed fitting.

    TO DO
    """

    def __init__(self,x, y, covx, covy,
                 wavelength, size_bloc=None,
                 fit_grey=False):

        self.fit_grey = fit_grey
        self.size_bloc = size_bloc
        
        self._x = x
        self._h = copy.deepcopy(self._x)
        self.y = y
        self.covx = covx
        self.covy = covy

        self.nsn = len(self.y[:,0])
        self.grey = 1 if self.fit_grey else 0
        self.ncomp = len(self._x[0]) + self.grey
        self.nslopes = len(self._x[0]) + 1
        self.nbin = len(self.y[0])
        self.dof = (self.nsn*self.nbin) - (self.nbin*(self.nslopes)) - self.grey*self.nsn
        
        # the one is for the mean spectrum        
        self.x = np.zeros((self.nsn,self.ncomp+1))
        self.x[:,0] = 1
        self.x[:,(1+self.grey):] = self._x
        
        self.wx = np.zeros((self.nsn,self.ncomp,self.ncomp))
        self.dy = np.zeros((self.nsn,self.nbin))
        self.wy = []

        for sn in range(self.nsn):
            self.wx[:,(self.grey):,(self.grey):][sn] = np.linalg.inv(self.covx[sn])
            self.wy.append(np.linalg.inv(self.covy[sn]))
            self.dy[sn] = np.sqrt(np.diag(self.covy[sn]))
            

        self.A = np.zeros((self.nbin,self.ncomp+1))
        self.h = copy.deepcopy(self.x)
        self.alpha = np.ones((self.nbin,self.ncomp))
        self.m0 = np.zeros(self.nbin)
        self.delta_m_grey = np.zeros(self.nsn)

        self.A[:,0]=self.m0
        self.A[:,1:]=self.alpha

        self.chi2 =None

    def init_fit(self, alpha=None, M0=None, delta_m_grey=None):
        
        self.separate_component()

        if alpha is not None:
           self.alpha = alpha 
        
        for Bin in range(self.nbin):
            print Bin+1,'/',self.nbin
            init = sugar.multilinearfit(self._x,self.y[:,Bin],yerr=self.dy[:,Bin],covx=self.covx)
            init.Multilinearfit(adddisp=True)
            self.alpha[Bin] = init.alpha
            self.m0[Bin] = init.M0
            
        if M0 is not None:
            self.m0 = M0

        if delta_m_grey is not None:
            self.delta_m_grey = delta_m_grey
        
        self.merge_component()
                                                                                      
    def comp_chi2(self):
        """
        Compute the ODR Chi2. 
        This is this chi2 which minimized
        in this chi2. 
        """
        residuy = self.y - np.dot(self.A,self.h.T).T
        residux = self.x[:,1:] - self.h[:,1:]
        self.chi2 = np.einsum("ij,ijk,ik->",residuy,self.wy,residuy) + np.einsum("ij,ijk,ik->",residux,self.wx,residux)


    def e_step(self):
        """
        Compute the true value of the data,
        and the grey offset
        """

        A=np.zeros(np.shape(self.wx))
        B=np.zeros((self.nsn,self.ncomp))
        H=np.zeros((self.nsn,self.ncomp))
        Y=np.zeros_like(self.y)
        
        self.separate_component()
        
        for sn in range(self.nsn):
            
            Y[sn]=self.y[sn] - self.m0
            T=np.dot(self.alpha.T,self.wy[sn].dot(self.alpha))+self.WX[j]
            A[j]=np.linalg.inv(T)
            B[j]=(np.dot(self.alpha.T,self.WY[j].dot(np.matrix(Y[j]).T))+np.dot(self.WX[j],np.matrix(self.data[j,1:]).T)).T
              
            H[j]=(np.dot(A[j],np.matrix(B[j]).T)).T
              
        if self.Parallel:
            h=np.zeros((self.N_sn,self.N_comp))
            self.comm.Allgatherv(H,h)
            self.comm.Barrier()
        else:
            h=H
              
        self.h[:,1:]=h

        if self.delta_M_grey:
            # mean of grey = 0
            mean_grey=copy.deepcopy(np.mean(self.h[:,1]))
            self.h[:,1]-=mean_grey
            self.A[:,0]+=mean_grey
            chi2=self.chi2
            self.comp_chi2()
            if abs(self.chi2-chi2)>1e-6:
                print 'ANDALOUSE GREY'
                print "chi2 avant %f chi2 apres %f"%((chi2,self.chi2))
            # decorrelation grey xplus
            if self.Color or self.intrinsic:
                self.comp_chi2()
                chi2=self.chi2
                self.decorrelate_grey_h()
                self.comp_chi2()
                if abs(self.chi2-chi2)>1e-6:
                    print 'ANDALOUSE'
                    print "chi2 avant %f chi2 apres %f"%((chi2,self.chi2))
            
 
    def decorrelate_grey_h(self):

        h=copy.deepcopy(self.xplus)
        self.add_cst_Av=0
        self.add_cst_xplus=np.zeros(len(self.xplus[0]))
        self.cov_grey_xplus=np.zeros(len(self.xplus[0]))
              
              
        for ncomp in range(len(self.cov_grey_xplus)):
            self.cov_grey_xplus[ncomp]=np.cov(self.delta_M_GREY,h[:,ncomp])[0,1]

        add_cst=np.dot(np.linalg.inv(np.cov(h.T)),self.cov_grey_xplus)
        self.add_cst_xplus=add_cst

    def m_step(self):

        if self.SPARSE:
            raise NameError('numpy array is needed for this fonction')


        if self.delta_M_grey:
            if self.CCM and not self.Color:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*np.ones(np.shape(self.Y)).T).T-(self.h[self.Filtre_parallel][:,2]*(self.A[:,2]*np.ones(np.shape(self.Y))).T).T
            else:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*np.ones(np.shape(self.Y)).T).T
        else:
            if self.CCM and not self.Color:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*(self.A[:,1]*np.ones(np.shape(self.Y))).T).T
            else:
                Y=copy.deepcopy(self.Y)

        self.A_vector=np.zeros(self.N_slopes*self.N_bin)
        hh_kron_W=np.zeros((len(self.A_vector),len(self.A_vector)))
        sum_WYh=np.zeros(np.shape(self.A[:,self.filter_grey_CCM]))
        self.sum_WYh_vector=np.zeros(len(self.A_vector))

        for j in range(self.Number_loop):
            if self.Parallel:
                sn=self.START+j
            else:
                sn=j
            h_ht=np.dot(np.matrix(self.h[sn,self.filter_grey_CCM]).T,np.matrix(self.h[sn,self.filter_grey_CCM]))
            hh_kron_W+=np.kron(h_ht,self.WY[j])
            sum_WYh+=np.dot(self.WY[j],np.dot(np.matrix(Y[j]).T,np.matrix(self.h[sn,self.filter_grey_CCM])))

        if self.Parallel:
            self.hh_kron_W=np.zeros((len(self.A_vector),len(self.A_vector)))
            self.sum_WYh=np.zeros(np.shape(self.A[:,self.filter_grey_CCM]))

            self.comm.Allreduce(hh_kron_W,self.hh_kron_W)
            self.comm.Allreduce(sum_WYh,self.sum_WYh)
            self.comm.Barrier()

        else:
            self.hh_kron_W=hh_kron_W
            self.sum_WYh=sum_WYh

         # deroulage des matrices en vecteur

        for i in range(self.N_slopes):
            self.sum_WYh_vector[i*self.N_bin:][:self.N_bin]=self.sum_WYh[:,i]

        X_cho=linalg.cho_factor(self.hh_kron_W)
        self.A_vector=linalg.cho_solve(X_cho,self.sum_WYh_vector)

        new_slopes=np.zeros(np.shape(self.A[:,self.filter_grey_CCM]))
        for i in range(self.N_slopes):
            new_slopes[:,i]=self.A_vector[i*self.N_bin:][:self.N_bin]
        self.A[:,self.filter_grey_CCM]=new_slopes
    
    def run_fit(self):

        return 'to do'

    def separate_component(self):

        self.m0 = copy.deepcopy(self.A[:,0])
        self.alpha = copy.deepcopy(self.A[:,1+self.grey:])
        self._x = copy.deepcopy(self.x[:,1+self.grey:])
        self._h = copy.deepcopy(self.h[:,1+self.grey:])
        if self.fit_grey:
            self.delta_m_grey = copy.deepcopy(self.h[:,1])

    def merge_component(self):

        self.A[:,0] = copy.deepcopy(self.m0)
        self.A[:,1+self.grey:] = copy.deepcopy(self.alpha)
        self.x[:,1+self.grey:] = copy.deepcopy(self._x)
        self.h[:,1+self.grey:] = copy.deepcopy(self._h)
        if self.fit_grey:
            self.h[:,1] = copy.deepcopy(self.delta_m_grey)
        

if __name__=='__main__':

    print 'to finish'
