"""fit sed using ODR algorithm."""

import numpy as np
import scipy.odr as odrs
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
                        estXZ=_lin_est(dataXZ)

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


class sed_fitting:
    """
    sed fitting.

    TO DO
    """

    def __init__(self):

        return None


if __name__=='__main__':

    print 'to finish'
