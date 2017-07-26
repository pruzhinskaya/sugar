#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
## Filename:          multilinearfit.py 
## Version:           $Revision: 1.48 $
## Description:       
## Author:            $Author: leget $
## Created at:        $Date: 2016/06/19 15:12:58 $
## Modified at:       01-10-2013 18:11:52
## $Id: multilinearfit.py,v 1.48 2016/06/19 15:12:58 leget Exp $
################################################################################

"""
"""

__author__  = "Pierre-Francois Leget <leget@clermont.in2p3.fr>"
__version__ = '$Id: multilinearfit.py,v 1.48 2016/06/19 15:12:58 leget Exp $'




import sys
import scipy.odr as O
import scipy.odr.models as M
from scipy import optimize,linalg,sparse
from scipy.sparse import block_diag
from scipy.misc import derivative
import pylab as P
import numpy as N
import scipy as S
from ToolBox import Hubblefit as H
from ToolBox import Astro
import copy
import cPickle
import scipy.interpolate as inter
import emfa_covariant as EMfa_cov


def extract_block_diag(A,size_bloc,number_bloc):

    start=(size_bloc)*number_bloc
    end=((size_bloc)*number_bloc)+size_bloc


    Non_zeros=N.array(S.sparse.extract.find(A))
    Filtre=(Non_zeros[0]>=start)
    Filtre=(Filtre & (Non_zeros[0]<end))

    Non_zeros=Non_zeros[:,Filtre]

    blocks=N.zeros((size_bloc,size_bloc))

    T=0

    for i in range(size_bloc):
        for j in range(size_bloc):
            if Non_zeros[0][T]==start+i and Non_zeros[1][T]==start+j:
                blocks[i,j]=Non_zeros[2][T]
                if T!=len(Non_zeros[2])-1:
                    T+=1


    return blocks




class comp_wRMS:

    def __init__(self,data,weight,Parallel=False,N_sn=None,Communicator=None):

        self.data=data

        if N_sn is not None:
            self.N_sn=N_sn
        else:
            self.N_sn=N_sn
        
        
        if Parallel:
            try: from mpi4py import MPI
            except ImportError : raise ImportError('you need to have mpi4py on your computer to use this option')
            self.MPI=MPI
            self.Parallel=True

            if Communicator is not None:
                self.comm= Communicator
            else:
                self.comm = self.MPI.COMM_WORLD
            size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.size=size
            self.njob=int(self.N_sn/size)

            if self.N_sn%size!=0:
                ValueError('Number of observation (supernovae) % MPI.COMM_WORLD.Get_size() need to be equal zeros (equal %i)'%(self.N_sn%size))

            self.START=self.njob*self.rank
            self.Number_loop=self.njob

        else:
            self.Parallel=False
            self.Number_loop=self.N_sn


        if self.Parallel:
                          
            self.data=[]
            self.weight=[]

            for j in range(self.njob):
                sn=self.START+j
                
                if len(weight)==self.njob:
                    self.weight.append(weight[j])
                else:
                    self.weight.append(weight[sn])
                if len(data)==self.njob:
                    self.data.append(data[j])
                else:
                    self.data.append(data[sn])

            if len(self.weight)!=1:
                rankF = N.sum(N.array(weight),axis=0)
            else:
                rankF = N.array(weight)
            sum_w=N.zeros(N.shape(rankF))
            self.comm.Allreduce(rankF,sum_w)
            self.comm.Barrier()
            self.inv_sum_W=N.linalg.inv(sum_w)
            self.data=N.array(self.data)
            self.weight=N.array(self.weight)
        else:
            self.weight=weight
            self.inv_sum_W=N.linalg.inv(N.sum(weight,axis=0))

    def comp_mean(self):

        Sum=N.zeros((len(self.data[0]),1))

        for j in range(self.Number_loop):
            D=self.data[j].reshape(len(self.data[j]),1)
            Sum+=N.dot(self.weight[j],D)
        if self.Parallel:
            SUM=N.zeros((len(self.data[0]),1))
            self.comm.Allreduce(Sum,SUM)
            self.comm.Barrier()
        else:
            SUM=Sum

        self.mean=N.dot(self.inv_sum_W,SUM)

    def comp_residu(self):

        self.comp_mean()
        self.residu=copy.deepcopy(self.data)
        self.residu-=self.mean.T


    def comp_wRMS(self):

        self.comp_residu()
        Sum=N.zeros((len(self.data[0]),1))
        for j in range(self.Number_loop):
            D=self.residu[j].reshape(len(self.residu[j]),1)
            Sum+=N.dot(self.weight[j],D*D)

        if self.Parallel:
            SUM=N.zeros((len(self.data[0]),1))
            self.comm.Allreduce(Sum,SUM)
            self.comm.Barrier()
        else:
            SUM=Sum

        self.wRMS=N.sqrt(N.dot(self.inv_sum_W,SUM))
 


def median_stats(a, weights=None, axis=None, scale=1.4826, corrected=True):
    """Compute [weighted] median and :func:`nMAD` of array *a* along
        *axis*. Weighted computation is implemented for *axis* = None
        only. If *corrected*, apply finite-sample correction from Croux &
        Rousseeuw (1992)."""
    
    if weights is not None:
        if axis is not None:
            raise NotImplementedError("implemented on 1D-arrays only")
        else:
            med  = wpercentile(a, 50., weights=weights)
            nmad = wpercentile(N.abs(a - med), 50., weights=weights) * scale
    else:
        med = N.median(a, axis=axis)
        if axis is None:
            umed = med                      # Scalar
    if weights is not None:
        if axis is not None:
            raise NotImplementedError("implemented on 1D-arrays only")
        else:
            med  = wpercentile(a, 50., weights=weights)
            nmad = wpercentile(N.abs(a - med), 50., weights=weights) * scale
    else:
        med = N.median(a, axis=axis)
        if axis is None:
            umed = med                      # Scalar
        else:
            umed = N.expand_dims(med, axis) # Same ndim as a
        nmad = N.median(N.absolute(a - umed), axis=axis) * scale

    if corrected:
        # Finite-sample correction on nMAD (Croux & Rousseeuw, 1992)
        if axis is None:
            n = N.size(a)
        else:
            n = N.shape(a)[axis]
        if n<=9:
            c = [0,0,1.196,1.495,1.363,1.206,1.200,1.140,1.129,1.107][n]
        else:
            c = n/(n-0.8)
        nmad *= c
    
    return med,nmad


#==============================================================================
#interpolation with spline
#==============================================================================

def Interpolation_spline(x,y,Number_of_points):
     
     SPLINE=inter.InterpolatedUnivariateSpline(x,y)

     X=N.linspace(x[0],x[-1],Number_of_points)
     Y=SPLINE(X)

     return X,Y


#==============================================================================
#MultiLinear fit based on a Orthogonal Distance Regression (ODR)
#==============================================================================


# this 4 next function are usefull to me for create multilinear fonction for ODR


def _lin_fcn(B,x):
     a,b=B[0],B[1:]
     b.shape=(b.shape[0],1)

     return a + (x*b).sum(axis=0)


def _lin_fjb(B, x):
     a = N.ones(x.shape[-1], float)
     res = N.concatenate((a, x.ravel()))
     res.shape = (B.shape[-1], x.shape[-1])
     return res
 
def _lin_fjd(B, x):
     b = B[1:]
     b = N.repeat(b, (x.shape[-1],)*b.shape[-1],axis=0)
     b.shape = x.shape
     return b
 
def _lin_est(data):
     # Eh. The answer is analytical, so just return all ones.
     # Don't return zeros since that will interfere with
     # ODRPACK's auto-scaling procedures.
 
     if len(data.x.shape) == 2:
          m = data.x.shape[0]
     else:
          m = 1
 
     return N.ones((m + 1,), float)
 



class Multilinearfit:
    
     def __init__(self,x,y,xerr=None,yerr=None,covx=None,Beta00=None):
          """
          Find slope(s) and intercept for a multilinear model with ODR package

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


          X=x.T

          if xerr is not None:
               xerr=xerr.T
          self.X=X
          self.y=y
          self.Beta00=Beta00

          if xerr is None: 
               self.xerr = N.ones(N.shape(X))
          else: 
               self.xerr = N.array(xerr)

          if yerr is None: 
               self.yerr = N.ones(len(y))
          else: 
               self.yerr = N.array(yerr)

          if covx is None: 
               self.covx=None
          else: 
               self.covx = covx.T


     def Multilinearfit(self,adddisp=False,PRINT=False):


          Fct=lambda B,X : _lin_fcn(B,X)
          jac=lambda B,X : _lin_fjb(B,X)
          jacd=lambda B,X : _lin_fjd(B,X)

          if self.Beta00 is None:
        
               if type(self.X[0])==N.float64:
                    BETA0=(1,0)
               else:    
                    BETA0=(len(self.X),)
                    for i in range(len(self.X)):
                         BETA0+=(len(self.X)-i-1,)

          else:
               BETA0=self.Beta00
    
          if self.covx is not  None:
               dataXZ=O.RealData(self.X,self.y,sy=self.yerr,covx=self.covx)
          else:
               dataXZ=O.RealData(self.X,self.y,sx=self.xerr,sy=self.yerr)
          estXZ=_lin_est(dataXZ)

          MODELXZ=O.Model(Fct, fjacb=jac,fjacd=jacd, estimate=estXZ)
          odrXZ=O.ODR(dataXZ,MODELXZ,beta0=BETA0,maxit=1000,sstol=0.)
          output=odrXZ.run()
          BETA1=(output.beta[0],)
          for i in range(len(output.beta)-1):
               BETA1+=(output.beta[i+1],)

          odrXZ=O.ODR(dataXZ,MODELXZ,beta0=BETA1,maxit=1000,sstol=0.)
          output=odrXZ.run()

        
          alpha=N.zeros(len(output.beta[1:]))
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
                                   dataXZ=O.RealData(self.X,self.y,sy=N.sqrt(self.yerr**2+self.disp**2),covx=self.covx)
                              else:
                                   dataXZ=O.RealData(self.X,self.y,sx=self.xerr,sy=N.sqrt(self.yerr**2+self.disp**2))
                              estXZ=_lin_est(dataXZ)

                              MODELXZ=O.Model(Fct, fjacb=jac,fjacd=jacd, estimate=estXZ)
                              odrXZ=O.ODR(dataXZ,MODELXZ,beta0=BETA0,maxit=1000,sstol=0.)
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
          y_error_corrected=N.zeros(len(self.y))      
        
          VARx=N.dot(self.X,self.X.T)/(len(self.y)-1)

          self.xplus=output.xplus.T

          for sn in range(len(self.y)):
            
               for correction in range(len(self.alpha)):

                    if len(self.alpha)==1:
                         y_corrected[sn] -=  self.alpha*self.X.T[sn] 
                         #y_corrected[sn] -=  self.alpha[correction]*self.X[0][sn] 
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
          self.dM=N.sqrt(self.yerr**2+self.disp**2)
          self.output=output
      

     def _compute_dispertion(self):
          self.disp=optimize.fmin(self._disp_function,self.disp,disp=0)[0]
          #self.disp=optimize.fsolve(self._disp_function,self.disp)

     def _disp_function(self,d):

          residu=N.zeros(len(self.y))
          VAR=N.zeros(len(self.y))
          A=N.matrix(N.concatenate([[1.0],[self.alpha[i] for i in range(len(self.alpha))]]))
          Cov=N.zeros((len(self.y),len(self.alpha)+1,len(self.alpha)+1))

        

          for sn in range(len(self.y)):
               Cov[sn][0][0]=self.yerr[sn]**2+d**2
               if self.covx is None:
                    if type(self.xerr.T[sn])!=N.float64:
                         Cov[:,1:,1:][sn]=N.diag(self.xerr.T[sn]**2)
               else:
                    Cov[:,1:,1:][sn]=self.covx.T[sn]
    
               residu[sn]=self.y[sn]-self.M0
              
               for correction in range(len(self.alpha)):
                    if type(self.X.T[sn])!=N.float64:
                         residu[sn]-=self.alpha[correction]*self.X.T[sn][correction]
                    else:
                         residu[sn]-=self.alpha*self.X.T[sn]

               if type(self.xerr.T[sn])!=N.float64 and self.covx is None:
                    VAR[sn]=self.yerr[sn]**2+d**2+self.alpha**2*self.xerr.T[sn]**2
               else:
                    VAR[sn]=N.dot(A,N.dot(Cov[sn],A.T))                                       
              
          chi2 = (residu**2/VAR).sum()
        

          return abs((chi2/self.dof)-1.)
    

     def comp_stat(self):
         
        #compute STD, RMS, WRMS, chi2
        
          residu=N.zeros(len(self.y))
          residu_plus=N.zeros(len(self.y))
          VAR=N.zeros(len(self.y))

          A=N.matrix(N.concatenate([[1.0],[self.alpha[i] for i in range(len(self.alpha))]]))

          Cov=N.zeros((len(self.y),len(self.alpha)+1,len(self.alpha)+1))

          for sn in range(len(self.y)):

               Cov[sn][0][0]=self.dM[sn]**2
               if self.covx is None:
                    for i in range(len(self.alpha)):
                         if type(self.xerr.T[sn])!=N.float64:
                              Cov[sn][i+1][i+1]=self.xerr.T[sn,i]**2
               else:
                    Cov[:,1:,1:][sn]=self.covx.T[sn]
    
               residu[sn]=self.y[sn]-self.M0
               residu_plus[sn]=self.y[sn]-self.M0
               
               for correction in range(len(self.alpha)):
             
                    if type(self.xerr.T[sn])!=N.float64:
                         residu[sn]-=self.alpha[correction]*self.X.T[sn][correction]
                        
                    else:
                         residu[sn]-=self.alpha*self.X.T[sn]
                    
                    if len(self.alpha)==1:
                         residu_plus[sn] -= self.alpha[correction]*self.xplus[sn]
                       
                    else:
                         residu_plus[sn]-=self.alpha[correction]*self.xplus[sn][correction]

               if type(self.xerr.T[sn])!=N.float64:
                    VAR[sn]=N.dot(A,N.dot(Cov[sn],A.T))
               else:
                    VAR[sn]=self.dM[sn]**2+self.alpha**2*self.xerr.T[sn]**2


          self.residu = residu
          self.residu_plus = residu_plus
         
          self.Cov_debug=Cov
          self.A_debug=A
          self.VAR= VAR
                
          self.chi2 = (residu**2/VAR).sum()
          self.chi2_plus = (residu_plus**2/VAR).sum()
         
          self.RMS=H.comp_rms(residu, self.dof, err=False)
          self.RMS_plus=H.comp_rms(residu_plus, self.dof, err=False)

          self.WRMS,self.WRMS_err=H.comp_rms(residu, self.dof, err=True, variance=VAR)
          self.WRMS_plus=H.comp_rms(residu_plus, self.dof, err=False, variance=VAR)





#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


#==============================================================================
# Global MultiLinear fit (Multilinearfit multi-band)
#==============================================================================


    ############################################################
    ############################################################
    # TO DO 
    # - compute orthgonnal projection error 
    # - update map chi2 (data or not and CovY or not and 
    #   parallelization or not) 
    ############################################################
    ############################################################


class global_fit:


     #################################################################################################### 
     #################################################################################################### 
     # Load fundamental data to run the fit and determination of supernovae, band and composante numbers 
     #################################################################################################### 
     #################################################################################################### 

     def __init__(self,Y,wavelength,data=None,CovX=None,dY=None,CovY=None,dm_z=None,alpha0=None,reddening_law=None,M00=None,H0=None,B_V=None,Delta_M0=None,Disp_matrix_Init=None,RV_init=None,Color=True,delta_M_grey=True,CCM=True,EMFA_disp_matrix=False,Communicator=None,Parallel=False):
          """
          Search of the spectral energy distribution of supernovae for one phase. This search is base on a multilinear model and used a Orthogonal Distance Regression to find this distribution.

          - Y: array_like
          
          Y are the spectra of the supernovae (in magnitude).
          Y have N lines and N_bin collones
          Where N are the number of supernovae and N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)
          
          - wavelength: array_like

          The mean wavelength for each band 
          wavelength have N_bin elements
          Where N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)

          - data: array_like

          Components witch depends Y. It can be strecth,color, EWCa H&K, EWSi 4128 or what do you want. 
          data have N lines and n collones.
          Where N are the number of supernovae and n the number of components 
          If data==None --> Only reddening and/or grey fit 

          - CovX: array_like 

          Covriance error matrix for data. 
          CovX have N matrix n times n.
          Where N are the number of supernovae and n the number of components 
          If CovX==None --> Only reddening and/or grey fit 

          - dY: array_like                                                                                                                                 
                                                                                                                                                         
          dY are errors on spectra of the supernovae (in magnitude).                                                                                       
          dY have N lines and N_bin collones 
          Where N are the number of supernovae and N_bin the number of bands 
          (If you are in photometry for a given phase with UBVRI filter N_bin=5)      
          
          - CovY: array_like or could be a scypi sparse matrix (if it's Block diagonal)                                                                                                                                                                                                                           
          Error covariance matrix for Y (spectra)
          CovY have N matrix N_bins times N_bins 
          Where N are the number of supernovae and N_bins the number of bands
          (If you are in photometry for a given phase with UBVRI filter N_bin=5, and
          the shape of CovY are 5*5)      
                                                                       
          - dm_z: array_like

          error in magnitude due to error on redshift and peculiar velocity.
          dmz have N elements
          Where N are the number of supernovae

          - alpha0: array_like

          Initialize the fit with intrinsic slopes
          alpha0 have N_bin 
          Where N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)
          
          - reddening_law: array_like

          Initialize the fit with a reddening law
          reddening_law have N_bin elements
          Where N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)
          
          - M00: array_like

          Initialize the fit with a mean spectrum (in magnitude)
          M00 have N_bin elements
          Where N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)


          - H0: array_like

          Initialize the fit with the true value of data
          H0 have N lines and n collones.
          Where N are the number of supernovae and n the number of components 


          - B_V: array_like
          
          Initialize the fit with E(B-V) or Av
          B_V have N elements
          Where N are the number of supernovae

          - Delta_M0: array_like

          Initialize the fit with a grey offset 
          Delta_M0 have N elements
          Where N are the number of supernovae

          - Disp_matrix_Init: array_like

          Initialize the fit with a dispertion matrix (see Chotard & al 2011)  
          Disp_matrix_Init have N_Bin lines and N_bin collones
          Where N_bin the number of bands (If you are in photometry with UBVRI like filter N_bin=5)

          - if Color: 

          The reddening law are adjusted like in Chotard & al. 2011 with a Av parameter by supernovae

          - if delta_M_grey:

          A grey offset are adjusted by supernovae

          - if CCM:

          The reddening law is supposed to be a Cardelli law  (see Cardelli & al. 1989) and just Av parameter by supernovae and a mean Rv are adjusted


          - if Color and CCM :

          Exactly like in Chotard & al 2011

          - if Parallel :
          
          Used the mpi4py package to parallelize each loop that depends of observation (supernovae). Accelerate the convergence if Y and CovY 
          are realy big. (typically if len(Y)>1000) 

          - To run the fit :

          GF=global_fit(Y,wavelength,data=None,CovX=None,
                        dY=None,CovY=None,dm_z=None,alpha0=None,
                        reddening_law=None,M00=None,H0=None,
                        B_V=None,Delta_M0=None,Disp_matrix_Init=None,
                        Color=True,delta_M_grey=True,CCM=True)

          GF.run_Global_fit(MAX_ITER=N.inf,Init=False,Addisp=False,JACK=False)
          
          If you want a chi2 by degrees of freedom reduce to 1 put the option adddisp=True (a dispertion are adjuste like in Chotard & al 2011)
          If you want the statistical error on slopes put the option JACK=True
          If you want a Initialisation with Mulitilinearfit to help global_fit to converge put Init=True
          MAX_ITER are the maximal number of iteration In the Expectation-step and the Minimization-step
          

          To have the result:
          
          GF.separate_slopes_and_data()

          mean spectrum : GF.M0
          intrinsic slope(s) : GF.Alpha
          reddening_law : GF.reddening_law                
          grey offset : GF.delta_M_GREY
          Av : GF.Av
          true value of data : GF.xplus
          Rv : 1./GF.inv_Rv
          dispertion matrix : (GF.disp_added**2)*GF.disp_matrix

          if JACK:

          statistical error on mean spectrum : GF.M0_err_Jackknife
          statistical error on intrinsic slope(s) : GF.alpha_err_Jackknife
          statistical error on reddening_law : GF.reddening_law_err_Jackknife
          statistical error on Rv : GF.RV_err_Jackknife 
          

          if you want RMS or wRMS of the fit:

          GF.comp_stat()


          """

        
        ################################################# 
        # Load data
        #################################################  

          self.Color=Color
          self.delta_M_grey=delta_M_grey
          self.CCM=CCM
          self.RV_init=RV_init
          
          self.N_sn=len(Y[:,0])
          self.N_bin=len(Y[0])
          if M00 is not None:
              self.M0=M00
          else:
               self.M0=N.mean(Y,axis=0)

          if CovY is not None and type(CovY[0])==S.sparse.coo.coo_matrix:
              self.SPARSE=True
          else:
              self.SPARSE=False


          
          if Parallel:
              try: from mpi4py import MPI
              except ImportError : raise ImportError('you need to have mpi4py on your computer to use this option')
              self.MPI=MPI
              self.Parallel=True
              
              if Communicator is not None:
                  self.comm= Communicator
              else:
                  self.comm = self.MPI.COMM_WORLD
              size = self.comm.Get_size()
              self.rank = self.comm.Get_rank()
              self.size=size
              self.njob=int(self.N_sn/size)
              
              if self.N_sn%size!=0:
                  ValueError('Number of observation (supernovae) % MPI.COMM_WORLD.Get_size() need to be equal zeros (equal %i)'%(self.N_sn%size))

              self.START=self.njob*self.rank
              self.Number_loop=self.njob

          else:
              self.Parallel=False
              self.Number_loop=self.N_sn

          if data is None:
              self.intrinsic=False
          else:
              self.intrinsic=True

          if dm_z is not None:
               self.peculiar_velocity=True
          else:
               self.peculiar_velocity=False
        
          if self.delta_M_grey:
               self.grey=1
          else:
               self.grey=0

          if CovY is None:
              self.DIAG=True
          else:
              self.DIAG=False

          if self.intrinsic:
              if CovX is None:
                  self.CovX=N.ones(shape(data))
              else:
                  self.CovX=CovX
          else:
              self.CovX=None

          self.wavelength=wavelength


     
          if self.intrinsic:
              if self.Color or self.CCM:
                  self.N_comp=len(data[0])+1+self.grey
                  self.N_slopes=len(data[0])+2
              else:
                  self.N_comp=len(data[0])+self.grey
                  self.N_slopes=len(data[0])+1
          else:
              if self.Color or self.CCM:
                  self.N_comp=1+self.grey
                  self.N_slopes=2
              else:
                  self.N_comp=self.grey
                  self.N_slopes=1

          if self.CCM and not self.Color:
              self.N_slopes-=1

          if self.N_comp==0:
              ValueError('Warning !: N_comp=0 (N_comp=number of intrinsic components + color + grey), used this class is pointless ...')
                        
          # the one is for the mean spectrum 

          self.data=N.zeros((self.N_sn,self.N_comp+1))
          self.data[:,0]=1

          if self.Color or self.CCM:
              if self.intrinsic:
                  self.data[:,(2+self.grey):]=data
          else:
              if self.intrinsic:
                  self.data[:,(1+self.grey):]=data


          ################################################# 
          # Compute weigth matrix
          #################################################  


          self.WX=N.zeros((self.N_sn,self.N_comp,self.N_comp))

          if Disp_matrix_Init is not None:
               self.disp_matrix=Disp_matrix_Init
               self.disp_added=1.
          else:
               if self.SPARSE:
                   self.disp_matrix=0
               else:
                   self.disp_matrix=N.zeros((self.N_bin,self.N_bin))
               self.disp_added=1.

          if CovY is None:
              
              self.Y=Y        
              self.dY=dY
              self.dm_z=dm_z

              self.CovY=N.zeros((self.N_sn,self.N_bin,self.N_bin))
              self.WY=N.zeros((self.N_sn,self.N_bin,self.N_bin))

              for sn in range(self.N_sn):
              
                  #################################
                  # peculiar velocity or not
                  #################################
                  if self.dm_z is not None:
                      self.CovY[sn]=N.diag(self.dY[sn]**2)+((self.dm_z[sn]**2)*N.ones(N.shape(self.CovY[sn])))

                      ####################################
                      # Sherman-Morrison matrix inversion
                      ####################################
                      A=N.diag(self.dY[sn]**(-2))
                      B=N.matrix(self.dm_z[sn]*N.ones(self.N_bin))
                      A_num=N.dot(A,N.dot(B.T,N.dot(B,A)))
                      A_den=float(1.+N.dot(B,N.dot(A,B.T)))
                      self.WY[sn]=A-(A_num*(1./A_den))
                      
                  else:
                      self.CovY[sn]=N.diag(self.dY[sn]**2)
                      self.WY[sn]=N.diag(self.dY[sn]**-2)

          else:
          
              if self.Parallel:

                  self.Y=[]
                  if dm_z is not None:
                      self.dm_z=[]
                  else:
                      self.dm_z=None
                  self.CovY=[]
                  self.WY=[]

                  for j in range(self.njob):
                      sn=self.START+j

                      if dm_z is not None:
                          if len(CovY)==self.njob:
                              self.CovY.append(CovY[j]+((dm_z[sn]**2)*N.ones(N.shape(CovY[j]))))
                          else:
                              self.CovY.append(CovY[sn]+((dm_z[sn]**2)*N.ones(N.shape(CovY[sn]))))
                      else:
                          if len(CovY)==self.njob:  
                              self.Y.append(Y[sn])  
                              if self.dm_z is not None:  
                                  self.dm_z.append(dm_z[sn])  
                              self.CovY.append(CovY[j])  
                          else:
                              self.CovY.append(CovY[sn])
                              self.Y.append(Y[sn])
                              if self.dm_z is not None:
                                  self.dm_z.append(dm_z[sn])
                                  
                      if self.SPARSE:
                          for i in range(len(self.CovY)):
                              w=[]
                              for j in range(190):
                                  w.append(N.linalg.inv(extract_block_diag(self.CovY[i],19,j)))
                          self.WY.append(block_diag(w))
                      else:
                          self.WY.append(N.linalg.inv(self.CovY[j]))

                  #self.WY=N.array(self.WY)
                  #self.CovY=N.array(self.CovY)

                  self.Y=N.array(self.Y)
                  del Y
                  del CovY
                  del dm_z
                  self.comm.Barrier()
                     
              else:

                  self.Y=Y        
                  self.dY=dY
                  self.dm_z=dm_z
                  self.CovY=CovY
                  self.WY=N.zeros(N.shape(self.CovY))
                  for sn in range(self.N_sn):
                 
                      if self.dm_z is not None:
                          self.CovY[sn]+=((self.dm_z[sn]**2)*N.ones(N.shape(self.CovY[sn])))

                      self.WY[sn]=N.linalg.inv(self.CovY[sn])                  


          for sn in range(self.N_sn):  
               if self.Color or self.CCM:
                   if self.intrinsic:
                       self.WX[:,(1+self.grey):,(1+self.grey):][sn]=N.linalg.inv(self.CovX[sn])
               else:
                   if self.intrinsic:
                       self.WX[:,(self.grey):,(self.grey):][sn]=N.linalg.inv(self.CovX[sn])
                


          ############################################################# 
          # determinate degrees of freedom
          #############################################################  
      
          if self.Color:
               if self.delta_M_grey:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*(self.N_comp-1))-self.N_bin-2*self.N_sn
               else:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*self.N_comp)-self.N_bin-self.N_sn

          if self.CCM and self.Color==False:
               if self.delta_M_grey:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*(self.N_comp-2))-self.N_bin-self.N_sn-1
               else:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*self.N_comp-1)-self.N_bin-1
          
          else:
               if self.delta_M_grey:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*(self.N_comp-1))-self.N_bin-self.N_sn
               else:
                    self.dof=(self.N_sn*self.N_bin)-(self.N_bin*self.N_comp)-self.N_bin


          ############################################################# 
          # Variable initialization 
          #############################################################  


          ###################################################################### 
          # Intrinsic slopes + slopes of ones for delta_M_grey (if Delta_M_grey)
          ######################################################################  

        
          if alpha0 is not None:
               self.alpha=N.ones((self.N_bin,self.N_comp))

               if self.Color or self.CCM:
                    self.alpha[:,(1+self.grey):]=alpha0
               else:
                    self.alpha[:,(0+self.grey):]=alpha0
          else:
               self.alpha=N.ones((self.N_bin,self.N_comp))

          if self.Color or self.CCM:
              self.a_cardelli,self.b_cardelli=Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)
              if reddening_law is not None:
                  self.alpha[:,(0+self.grey)]=reddening_law
              else:
                  if self.RV_init is None:
                      self.inv_Rv=1./3.1
                  else:
                      self.inv_Rv=1./self.RV_init

                  self.alpha[:,(0+self.grey)]=self.a_cardelli+(self.inv_Rv*self.b_cardelli)       
#                  self.alpha[:,(0+self.grey)]=N.ones(self.N_bin)

               
          if H0 is not None:
               self.h=copy.deepcopy(self.data)
               if self.Color or self.CCM: 
                    self.h[:,(2+self.grey):]=H0
               else:
                    self.h[:,(1+self.grey):]=H0

               
          else:
               self.h=copy.deepcopy(self.data)
    
          if self.Color or self.CCM:
               if B_V is not None:
                    self.h[:,(1+self.grey)]=B_V
               else:
                   self.h[:,(1+self.grey)]=N.zeros(self.N_sn)


          self.filter_grey_CCM=N.array([True]*(self.N_comp+1))

          if self.delta_M_grey:
               self.filter_grey_CCM[1]=False
               if Delta_M0 is not None:
                    self.h[:,1]=Delta_M0
               else:
                    self.h[:,1]=0

          if self.CCM and not self.Color:
               self.filter_grey_CCM[(1+self.grey)]=False

          if self.RV_init is None:
              self.inv_Rv=None
          else:
              self.inv_Rv=1./self.RV_init

          self.A=N.zeros((self.N_bin,self.N_comp+1))
          self.A[:,0]=self.M0
          self.A[:,1:]=self.alpha
                        
    
          #self.AALPHA=[]
          #self.MM0=[]
          #self.HH=[]
          self.CHI2=[]
          self.CHI2_TRADITIONNAL=[]

          self.Filtre_parallel=N.array([True]*self.N_sn)

          if self.Parallel:
              self.Filtre_parallel=N.array([False]*self.N_sn)
              del data
              WX=[]
              data=[]
              for j in range(self.Number_loop):
                  sn=self.START+j
                  WX.append(self.WX[sn])
                  data.append(self.data[sn])
                  self.Filtre_parallel[sn]=True
              del self.WX
              del self.data
              self.WX=N.array(WX)
              self.data=N.array(data)
          
          self.EMFA_disp_matrix=EMFA_disp_matrix
          self.LAMBDA_INIT=None
          self.NITER_emfa=1000 
    
     ####################################################################################
     ####################################################################################
     # Initialization of intrinsic slopes and mean spectrum 
     ####################################################################################
     ####################################################################################


     def initialization_slopes_and_mean_spectrum(self,PRINT=True):
         
         """
         initialization with the ODR package from scipy 
         of intrinsic component and mean spectrum. This 
         function help the to find the global minimum 
         of the chi2.

         """

         if not self.intrinsic:
             return
         else:
              self.separate_slopes_and_data()
              
              print'**********************************************************************'
              print '             Initialization of intrinsic slope(s):'
              print'********************************************************************** \n'

              if self.Parallel:
                  Data=N.zeros((self.N_sn,len(self.Data[0])))
                  Y=N.zeros((self.N_sn,len(self.Y[0])))
                  dY=N.zeros((self.N_sn,len(self.Y[0])))

                  self.comm.Allgatherv(self.Data,Data)
                  self.comm.Allgatherv(self.Y,Y)
                  if self.DIAG:
                      self.comm.Allgatherv(self.dY,dY)
                  else:
                      dy=[]
                      for t in range(len(self.CovY)):
                          dy.append(N.sqrt(self.CovY[t].diagonal()))
                      dy=N.array(dy)
                      self.comm.Allgatherv(dy,dY)
                      del dy
                  self.comm.Barrier()

              else:
                  Data=self.Data
                  Y=self.Y
                  if self.DIAG:
                      dY=self.dY
                  else:
                      dY=N.sqrt(N.diag(self.CovY))


              for Bin in range(self.N_bin):
                  INIT=Multilinearfit(Data,Y[:,Bin],yerr=dY[:,Bin],covx=self.CovX)
                  #INIT=Multilinearfit(Data,Y[:,Bin],yerr=None,covx=None)
                  INIT.Multilinearfit(adddisp=True)
                  self.Alpha[Bin]=INIT.alpha
                  self.M0[Bin]=INIT.M0
                  if PRINT:
                      description='Initialization of bin %i/%i'%((Bin+1,self.N_bin))
                      print(description) 
                      sys.stdout.write("\033[F")
    
                      print(description+'\n')
              print'**********************************************************************'
              print'                      End of initialization ! '
              print'********************************************************************** \n'

              self.Data_controle=Data
              self.Y_controle=Y
              self.dy_controle=dY
              del Data
              del Y
              del dY

              self.merge_slopes_and_data()
    
     ####################################################################################
     ####################################################################################
     # etapes E et M et calcul Chi2 ODR du fit
     ####################################################################################
     ####################################################################################

    
     ############################################################# 
     # separate mean spectrum and slopes 
     #############################################################  

     def separate_alpha_M0(self):
          self.M0=self.A[:,0]
          self.alpha=self.A[:,1:]


    ############################################################# 
    # Compute chi2
    #############################################################  

     def comp_chi2(self,reddening_law=None,Av=None):

          """
          Compute the ODR Chi2. This is this chi2 which 
          minimized in this chi2. The option reddening_law and 
          Av are used when you do a cardelli fit

          """

          A=copy.deepcopy(self.A)
          h=copy.deepcopy(self.h)

          
          if reddening_law is not None:
               A[:,(1+self.grey)]=reddening_law
               
          if Av is not None:
               h[:,(1+self.grey)]=Av

          Chi2=0.
          self.CCHI=[]
          for j in range(self.Number_loop):
              if self.Parallel:
                  sn=self.START+j
              else:
                  sn=j

              self.residu1=self.Y[j]-N.dot(A,N.matrix(h[sn]).T).T 
              self.residu2=self.data[j,1:]-h[sn,1:]
              
              self.CCHI.append(N.dot(N.matrix(self.residu1),self.WY[j].dot(N.matrix(self.residu1).T))+N.dot(N.matrix(self.residu2),N.dot(self.WX[j],N.matrix(self.residu2).T)))
              Chi2+= N.dot(N.matrix(self.residu1),self.WY[j].dot(N.matrix(self.residu1).T))+N.dot(N.matrix(self.residu2),N.dot(self.WX[j],N.matrix(self.residu2).T))
              #Chi2+= N.dot(N.matrix(self.residu1),N.dot(self.WY[j].toarray(),N.matrix(self.residu1).T))+N.dot(N.matrix(self.residu2),N.dot(self.WX[j],N.matrix(self.residu2).T))

          if self.Parallel:
              rankF = N.array(Chi2[0,0])
              chi2 = N.zeros(1)
              self.comm.Allreduce(rankF,chi2, op=self.MPI.SUM)
              self.chi2=chi2[0]
              self.comm.Barrier()
          else:
              self.chi2=Chi2[0,0]


     ############################################################# 
     # Compute orthogonal projection (E-step)
     #############################################################  

     def compute_h(self):

          """
          Compute the true value of the data,
          Av and the grey offset
 
          """

          A=N.zeros(N.shape(self.WX))
          B=N.zeros((self.Number_loop,self.N_comp))
          H=N.zeros((self.Number_loop,self.N_comp))
          Y=N.zeros(N.shape(self.Y))
        
          self.separate_alpha_M0()
          
          for j in range(self.Number_loop):
              if self.Parallel:
                  sn=self.START+j
              else:
                  sn=j
              Y[j]=self.Y[j] - self.M0
              T=N.dot(self.alpha.T,self.WY[j].dot(self.alpha))+self.WX[j]
              A[j]=N.linalg.inv(T)
              B[j]=(N.dot(self.alpha.T,self.WY[j].dot(N.matrix(Y[j]).T))+N.dot(self.WX[j],N.matrix(self.data[j,1:]).T)).T
              
              H[j]=(N.dot(A[j],N.matrix(B[j]).T)).T
              
          if self.Parallel:
              h=N.zeros((self.N_sn,self.N_comp))
              self.comm.Allgatherv(H,h)
              self.comm.Barrier()
          else:
              h=H
              
          self.h[:,1:]=h

          if self.Color or self.CCM:
               # mean of Av = 0

               mean_Av=copy.deepcopy(N.mean(self.h[:,(1+self.grey)]))
               self.h[:,(1+self.grey)]-=mean_Av
               self.A[:,0]+=mean_Av*self.A[:,(1+self.grey)]

               
               # decorrelation Av xplus 

               if self.intrinsic:
                   self.comp_chi2()
                   chi2=self.chi2
                   self.decorrelate_Av_h()
                   self.comp_chi2()
                   if abs(self.chi2-chi2)>1e-6:
                       print 'ANDALOUSE AV'
                       print "chi2 avant %f chi2 apres %f"%((chi2,self.chi2))

               if self.Color:
               # scale of Av fixed on the median wavelength
                   if len(self.wavelength)%2==1:
                       Ind_med=list(self.wavelength).index(N.median(self.wavelength))
                   else:
                       Ind_med=(len(self.wavelength)/2)-1

                   Red_law_med=copy.deepcopy(self.A[:,(1+self.grey)][Ind_med])
               
                   self.h[:,(1+self.grey)] *= Red_law_med
                   self.A[:,(1+self.grey)] /= Red_law_med
               


          if self.delta_M_grey:

               # mean of grey = 0

               mean_grey=copy.deepcopy(N.mean(self.h[:,1]))
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
            
        
     ################################################################ 
     # Decorrelate the grey offset and the orthogonal projection (h) 
     ################################################################  

     def decorrelate_grey_h(self):

         self.separate_slopes_and_data()
         self.compute_cov_grey_h()

         if self.intrinsic:
             for ncomp in range(len(self.xplus[0])):
                 self.delta_M_GREY-=self.xplus[:,ncomp]*self.add_cst_xplus[ncomp]
                 self.Alpha[:,ncomp]+=self.add_cst_xplus[ncomp]
    
         if self.Color: #and self.CCM==False:
             self.delta_M_GREY-=self.Av*self.add_cst_Av
             self.reddening_law+=self.add_cst_Av

         self.merge_slopes_and_data()



     def compute_cov_grey_h(self):

          if self.Color:
              h=N.zeros((self.N_sn,len(self.xplus[0])+1))
              h[:,0]=copy.deepcopy(self.Av)
              h[:,1:]=copy.deepcopy(self.xplus)
              self.add_cst_Av=0
              self.add_cst_xplus=N.zeros(len(self.xplus[0]))
              self.cov_grey_xplus=N.zeros(len(self.xplus[0])+1)
          else:
              h=copy.deepcopy(self.xplus)
              self.add_cst_Av=0
              self.add_cst_xplus=N.zeros(len(self.xplus[0]))
              self.cov_grey_xplus=N.zeros(len(self.xplus[0]))
              
              
          for ncomp in range(len(self.cov_grey_xplus)):
              self.cov_grey_xplus[ncomp]=N.cov(self.delta_M_GREY,h[:,ncomp])[0,1]

          add_cst=N.dot(N.linalg.inv(N.cov(h.T)),self.cov_grey_xplus)
          
          if self.Color:
              self.add_cst_Av=add_cst[0]
              self.add_cst_xplus=add_cst[1:]
          else:
              self.add_cst_xplus=add_cst

     ################################################################ 
     # Decorrelate Av and orthogonal projection (h) 
     ################################################################  

     def decorrelate_Av_h(self):

         self.separate_slopes_and_data()
         self.compute_cov_Av_h()

         if self.intrinsic:
             for ncomp in range(len(self.xplus[0])):
                 self.Av-=self.xplus[:,ncomp]*self.add_cst_xplus_Av[ncomp]
                 self.Alpha[:,ncomp]+=self.add_cst_xplus_Av[ncomp]*self.reddening_law

#         if self.delta_M_grey: 
#             self.delta_M_GREY-=self.Av*self.add_cst_Av
#             self.reddening_law+=self.add_cst_Av

         self.merge_slopes_and_data()



     def compute_cov_Av_h(self):

#          if self.Color:
#              h=N.zeros((self.N_sn,len(self.xplus[0])+1))
#              h[:,0]=copy.deepcopy(self.Av)
#              h[:,1:]=copy.deepcopy(self.xplus)
#              self.add_cst_Av=0
#              self.add_cst_xplus=N.zeros(len(self.xplus[0]))
#              self.cov_grey_xplus=N.zeros(len(self.xplus[0])+1)
#          else:
         h=copy.deepcopy(self.xplus)
         #self.add_cst_Av=0
         self.add_cst_xplus_Av=N.zeros(len(self.xplus[0]))
         self.cov_Av_xplus=N.zeros(len(self.xplus[0]))
              
              
         for ncomp in range(len(self.cov_Av_xplus)):
             self.cov_Av_xplus[ncomp]=N.cov(self.Av,h[:,ncomp])[0,1]

         if len(self.cov_Av_xplus)==1:
             add_cst=self.cov_Av_xplus/(N.std(h)**2)
         else:
             add_cst=N.dot(N.linalg.inv(N.cov(h.T)),self.cov_Av_xplus)
             
#          
#          if self.Color:
#              self.add_cst_Av=add_cst[0]
#              self.add_cst_xplus=add_cst[1:]
#          else:
         self.add_cst_xplus_Av=add_cst



     ################################################################ 
     # Compute gradient of slopes or slopes (M-step 1/2)
     ################################################################

     def compute_slopes(self):

         if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')


         if self.delta_M_grey:
             if self.CCM and not self.Color:
                 Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*N.ones(N.shape(self.Y)).T).T-(self.h[self.Filtre_parallel][:,2]*(self.A[:,2]*N.ones(N.shape(self.Y))).T).T
             else:
                 Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*N.ones(N.shape(self.Y)).T).T
         else:
             if self.CCM and not self.Color:
                 Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*(self.A[:,1]*N.ones(N.shape(self.Y))).T).T
             else:
                 Y=copy.deepcopy(self.Y)

         self.A_vector=N.zeros(self.N_slopes*self.N_bin)
         hh_kron_W=N.zeros((len(self.A_vector),len(self.A_vector)))
         sum_WYh=N.zeros(N.shape(self.A[:,self.filter_grey_CCM]))
         self.sum_WYh_vector=N.zeros(len(self.A_vector))

         for j in range(self.Number_loop):
             if self.Parallel:
                 sn=self.START+j
             else:
                 sn=j
             h_ht=N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]).T,N.matrix(self.h[sn,self.filter_grey_CCM]))
             hh_kron_W+=N.kron(h_ht,self.WY[j])
             sum_WYh+=N.dot(self.WY[j],N.dot(N.matrix(Y[j]).T,N.matrix(self.h[sn,self.filter_grey_CCM])))

         if self.Parallel:
             self.hh_kron_W=N.zeros((len(self.A_vector),len(self.A_vector)))
             self.sum_WYh=N.zeros(N.shape(self.A[:,self.filter_grey_CCM]))

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


     def compute_slopes_bloc_diagonal(self,size_bloc=19):

        if not self.Parallel:
            raise NameError('Parallelization is needed for this fonction')


        if self.delta_M_grey:
            if self.CCM and not self.Color:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*N.ones(N.shape(self.Y)).T).T-(self.h[self.Filtre_parallel][:,2]*(self.A[:,2]*N.ones(N.shape(self.Y))).T).T
            else:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*N.ones(N.shape(self.Y)).T).T
        else:
            if self.CCM and not self.Color:
                Y=copy.deepcopy(self.Y)-(self.h[self.Filtre_parallel][:,1]*(self.A[:,1]*N.ones(N.shape(self.Y))).T).T
            else:
                Y=copy.deepcopy(self.Y)

        self.A_vector=N.zeros(self.N_slopes*self.N_bin)

        new_slopes=N.zeros(N.shape(self.A[:,self.filter_grey_CCM]))

        if self.rank == 0:
            print self.rank, 'Starting Lambda full'

        h_ht_dict = {}
        
        for j in range(self.Number_loop):
            if self.Parallel:
                sn=self.START+j
            else:
                sn=j

            for i in range(self.N_bin/size_bloc):
                h_ht=N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]).T,N.matrix(self.h[sn,self.filter_grey_CCM]))
                
                if self.SPARSE:
                    W_sub=extract_block_diag(self.WY[j],size_bloc,i)
                else:
                    W_sub = self.WY[j][i*size_bloc:(i+1)*size_bloc, i*size_bloc:(i+1)*size_bloc]
                
                hh_kron_W_sub = linalg.kron(h_ht, W_sub)
                WYh = N.dot(W_sub, N.dot(N.matrix(Y[j,i*size_bloc:(i+1)*size_bloc]).T,
                                                    N.matrix(self.h[sn,self.filter_grey_CCM])))
                target = i % self.size
                if self.rank == target:
                    hh_kron_W_sum =  N.copy(hh_kron_W_sub)
                    WYh_sum = N.copy(WYh)
                    
                if target != self.rank:
                    self.comm.Send([hh_kron_W_sub, self.MPI.DOUBLE],
                                   dest=target, tag=i)
                    self.comm.Send([WYh, self.MPI.DOUBLE],
                                   dest=target, tag=1000+i)
    
                if target == self.rank:
                    for r in range(self.size):
                        if r == self.rank: continue
                        Buffer = N.empty_like(hh_kron_W_sub)
                        Buffer2 = N.empty_like(WYh)
                        self.comm.Recv([Buffer, self.MPI.DOUBLE],
                                       source=r, tag=i)
                        self.comm.Recv([Buffer2, self.MPI.DOUBLE],
                                       source=r, tag=1000+i)
                        hh_kron_W_sum += Buffer
                        WYh_sum += Buffer2
                    h_ht_dict[i] = [hh_kron_W_sum, WYh_sum]
                    
            for wl in h_ht_dict.keys():
                hh_kron_W_sum, W_sum = h_ht_dict[wl]
                sum_WYh_vector = N.zeros(size_bloc*self.N_slopes)
                for i in xrange(self.N_slopes):   
                    sum_WYh_vector[i*size_bloc:][:size_bloc]=W_sum[:,i].ravel()

                X_cho = linalg.cho_factor(hh_kron_W_sum)
                slopes_solve = linalg.cho_solve(X_cho, sum_WYh_vector)
                for i in xrange(self.N_slopes):
                    new_slopes[wl*size_bloc:(wl+1)*size_bloc,i] = slopes_solve[i*size_bloc:(i+1)*size_bloc]
            
            New_slopes=N.zeros(N.shape(new_slopes))

            self.comm.Allreduce(new_slopes, New_slopes, op = self.MPI.SUM)
            self.A[:,self.filter_grey_CCM]=New_slopes



     def Compute_gradiant(self):
         
          if self.Parallel:
              raise NameError('The parallelization is not implemented in this fonction')
          
          if self.SPARSE:
              raise NameError('numpy array is needed for this fonction')


          grad_A_sn=N.zeros((self.N_sn,)+N.shape(self.A[:,self.filter_grey_CCM]))
          residu=N.zeros((self.N_sn,self.N_bin))

          if self.delta_M_grey:
               if self.CCM and not self.Color:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*N.ones(N.shape(self.Y)).T).T-(self.h[:,2]*(self.A[:,2]*N.ones(N.shape(self.Y))).T).T
               else:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*N.ones(N.shape(self.Y)).T).T
          else:
               if self.CCM and not self.Color:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*(self.A[:,1]*N.ones(N.shape(self.Y))).T).T
               else:
                    Y=copy.deepcopy(self.Y)
        
          for sn in range(self.N_sn):
            
               residu[sn]=Y[sn]-N.dot(self.A[:,self.filter_grey_CCM],N.matrix(self.h[sn,self.filter_grey_CCM]).T).T
            
               grad_A_sn[sn]=-2.*N.dot(self.WY[sn],N.dot(N.matrix(residu[sn]).T,N.matrix(self.h[sn,self.filter_grey_CCM])))

        
          self.grad_A = grad_A_sn.sum(axis=0)



    ########################################################################### 
    # Compute the epsilon to know how advance in the slopes space (M-step 2/2)
    ###########################################################################  


     def Compute_epsilon(self):

          if self.Parallel:
             raise NameError('The parallelization is not implemented in this fonction')

          if self.SPARSE:
              raise NameError('numpy array is needed for this fonction')


          A=N.zeros(self.N_sn)
          B=N.zeros(self.N_sn)

          if self.delta_M_grey:
               if self.CCM and not self.Color:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*N.ones(N.shape(self.Y)).T).T-(self.h[:,2]*(self.A[:,2]*N.ones(N.shape(self.Y))).T).T
               else:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*N.ones(N.shape(self.Y)).T).T
          else:
               if self.CCM and not self.Color:
                    Y=copy.deepcopy(self.Y)-(self.h[:,1]*(self.A[:,1]*N.ones(N.shape(self.Y))).T).T
               else:
                    Y=copy.deepcopy(self.Y)


          for sn in range(self.N_sn):

               A[sn]=N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]),N.dot(N.matrix(self.grad_A).T,N.dot(self.WY[sn],N.dot(N.matrix(self.grad_A),N.matrix(self.h[sn,self.filter_grey_CCM]).T))))
               B[sn]=N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]),N.dot(self.A[:,self.filter_grey_CCM].T,N.dot(self.WY[sn],N.dot(self.grad_A,N.matrix(self.h[sn,self.filter_grey_CCM]).T)))) + N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]),N.dot(self.grad_A.T,N.dot(self.WY[sn],N.dot(self.A[:,self.filter_grey_CCM],N.matrix(self.h[sn,self.filter_grey_CCM]).T)))) - N.dot(N.matrix(self.h[sn,self.filter_grey_CCM]),N.dot(self.grad_A.T,N.dot(self.WY[sn],N.matrix(Y[sn]).T))) - N.dot(N.matrix(Y[sn]),N.dot(self.WY[sn],N.dot(self.grad_A,N.matrix(self.h[sn,self.filter_grey_CCM]).T)))
        
          self.epsilon=-(B.sum()/(2.*A.sum()))


     ################################
     #  Compute Rv (& Av if Color)
     ################################

     def comp_Av_cardelli(self):

          if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')



          a,b = Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)
          S=a+b*self.inv_Rv

          if self.CCM and self.Color:

              Av_cardelli=N.zeros(self.Number_loop)

              Filtre=N.array([True]*(self.N_comp+1))
              Filtre[(1+self.grey)]=False
       
              for j in range(self.Number_loop):

                  if self.Parallel:
                      sn=self.START+j
                  else:
                      sn=j

                  Y_corr=self.Y[j]-N.dot(self.A[:,Filtre],N.matrix(self.h[sn,Filtre]).T).T 
                  up=((N.dot(N.matrix(Y_corr),N.dot(self.WY[j],N.matrix(S).T)))+(N.dot(N.matrix(S),N.dot(self.WY[j],N.matrix(Y_corr).T))))
                  down=2.*(N.dot(N.matrix(S),N.dot(self.WY[j],N.matrix(S).T)))
                  Av_cardelli[j]=up[0]/down[0]
                  
              if self.Parallel:
                  self.Av_cardelli = N.zeros(self.N_sn)
                  self.comm.Allgatherv(Av_cardelli,self.Av_cardelli)
                  self.comm.Barrier()

              else:
                  self.Av_cardelli=Av_cardelli

          else:
              print 'You must used this function only when you want to do a agnostic fit AND a cardelli fit'



     def comp_Rv(self,On_Gamma=True):

         if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')

         if not On_Gamma:
             a,b = Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)
             self.separate_slopes_and_data()
         
             if self.CCM and self.Color:
                 # les Av ne sont pas alligner sur la loi de cardelli, il faut que je mette le meme point zeros
                 Av=copy.deepcopy(self.Av_cardelli)
             else:
                 Av=copy.deepcopy(self.Av)

             Filtre=N.array([True]*(self.N_comp+1))
             
             Filtre[(1+self.grey)]=False
                           
             up=0.
             down=0.
        
             for j in range(self.Number_loop):

                 if self.Parallel:
                     sn=self.START+j
                 else:
                     sn=j

                 S=self.Y[j]-N.dot(self.A[:,Filtre],N.matrix(self.h[sn,Filtre]).T).T 
                 up+=(Av[sn])*N.dot(N.matrix(S),N.dot(self.WY[j],N.matrix(b).T))+(Av[sn])*N.dot(N.matrix(b),N.dot(self.WY[j],N.matrix(S).T))-(Av[sn]**2)*N.dot(N.matrix(a),N.dot(self.WY[j],N.matrix(b).T))-(Av[sn]**2)*N.dot(N.matrix(b),N.dot(self.WY[j],N.matrix(a).T))
                 down+=(Av[sn]**2)*N.dot(N.matrix(b),N.dot(self.WY[j],N.matrix(b).T))

             if self.Parallel:
                 up = N.array(up[0,0])
                 down = N.array(down[0,0])
                 UP = N.zeros(1)
                 DOWN = N.zeros(1)
                 self.comm.Allreduce(up,UP, op=self.MPI.SUM)
                 self.comm.Allreduce(down,DOWN, op=self.MPI.SUM)
                 self.comm.Barrier()
                 UP=UP[0]
                 DOWN=DOWN[0]
             else:
                 UP=up[0,0]
                 DOWN=down[0,0]
             
             inv_Rv=0.5*(UP/DOWN)
             self.inv_Rv=inv_Rv

         else:
             a,b = Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)
             self.separate_slopes_and_data()
             slope=copy.deepcopy(self.reddening_law)
             #ind_med=N.where(slope==1)[0][0]

             up=N.sum(slope*b)-N.sum(a*b)+N.sum(b*slope)-N.sum(b*a)
             down=2*N.sum(b*b)
             self.inv_Rv=up/down

     def converge_CCM_fit(self,PRINT=True,ON_GAMMA=True):

          if self.SPARSE:
              raise NameError('numpy array is needed for this fonction')


          if PRINT:
               print'**********************************************************************'
               print '                   Cardelli Fit                                      '
               print'********************************************************************** \n'



          
          self.CHI2_CCM=[]
          a,b = Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)

          Iteration=0
          critere=[]
          self.inv_Rv=1./3.1
          self.reload_WY()
          
          self.comp_chi2(a+b*self.inv_Rv,Av=self.Av)

          critere.append(N.inf)
          critere.append(self.chi2)
          self.CHI2_CCM.append(self.chi2)

          if not ON_GAMMA:
                          
              while critere[Iteration]-critere[Iteration+1]>1e-7:
                  self.comp_Av_cardelli()

                  self.comp_chi2(reddening_law=a+b*self.inv_Rv,Av=self.Av_cardelli)
                  self.CHI2_CCM.append(self.chi2)
                  self.comp_Rv(On_Gamma=ON_GAMMA)
               
                  self.comp_chi2(reddening_law=a+b*self.inv_Rv,Av=self.Av_cardelli)
                  critere.append(self.chi2)
                  self.CHI2_CCM.append(self.chi2)
               
                  if PRINT:
                      description='Step:%i    Chi2:%f '%((Iteration+1,self.chi2))
                      print(description)
                      sys.stdout.write("\033[F")

                  if self.CHI2_CCM[-2]-self.CHI2_CCM[-1]<0:
                      print 'Problem !!!!!'

                  Iteration+=1
                
                
                  if Iteration>200:
                      break
              print(description+'\n')

          else:
              print 'Je passe par nouvelle routine'
              self.comp_Rv(On_Gamma=True)
              self.comp_Av_cardelli()




          
          if PRINT: 

               print'**********************************************************************'
               print '                  End of Cardelli Fit                                '
               print'********************************************************************** \n'


     ############################################################# 
     # E-step
     #############################################################  

     def Expectation_step(self):

          self.compute_h()
          
          if self.CCM and not self.Color:
              if self.RV_init is None:
                  self.comp_Rv()
                  #if not self.Color:
                  self.alpha[:,(0+self.grey)]=copy.deepcopy(self.a_cardelli+(self.inv_Rv*self.b_cardelli))
                  self.A[:,(1+self.grey)]=copy.deepcopy(self.a_cardelli+(self.inv_Rv*self.b_cardelli))
            

     ############################################################# 
     # M-step
     #############################################################  


     def Minimisation_step(self,Cholesky=True,BLOC=False):
         
         if Cholesky:
             if BLOC:
                 self.compute_slopes_bloc_diagonal(size_bloc=19)
             else:
                 self.compute_slopes()
                 new_slopes=N.zeros(N.shape(self.A[:,self.filter_grey_CCM]))
                 for i in range(self.N_slopes):
                     new_slopes[:,i]=self.A_vector[i*self.N_bin:][:self.N_bin]
                 self.A[:,self.filter_grey_CCM]=new_slopes

         else:
             self.Compute_gradiant()
             self.Compute_epsilon()
             
             self.A[:,self.filter_grey_CCM]+=self.epsilon*self.grad_A



     ############################################################# 
     # Chi2 minimization by EM step
     #############################################################  

     def converge_chi2(self,Max_iter=500,DISP_ADDed=None,CHOLESKY=True,BLOC_DIAGONAL=False,PRINT=True):
          
          self.reload_WY(DISP_ADDED=DISP_ADDed)
          E_step=0
          M_step=0

          Iteration1=0
          critere1=[]
          self.comp_chi2()
          critere1.append(N.inf)
          critere1.append(self.chi2)
          self.CHI2.append(self.chi2)
          

          if PRINT:
               print'**********************************************************************'
               print '                   Start of minimization:'
               print'********************************************************************** \n'
               

          
          self.Expectation_step() 
          E_step+=1
          #self.HH.append(copy.deepcopy(self.h))
          self.comp_chi2()               
          self.CHI2.append(self.chi2)
 
          if self.Parallel:
              ARRET=1e-3
          else:
              ARRET=1e-7

          while critere1[Iteration1]-critere1[Iteration1+1]>ARRET:
           
               if CHOLESKY:
                   
                   if BLOC_DIAGONAL:
                       self.Minimisation_step(Cholesky=CHOLESKY,BLOC=BLOC_DIAGONAL)
                   else:
                       self.Minimisation_step(Cholesky=CHOLESKY)
                   M_step+=1

                   self.comp_chi2()
                   self.CHI2.append(self.chi2)
                   self.separate_alpha_M0()
                   #self.AALPHA.append(copy.deepcopy(self.alpha))
                   #self.MM0.append(copy.deepcopy(self.M0))
                    
                      
                   if PRINT:
                       description='E-step:%i    M-step:%i    Chi2:%f    Chi2/dof:%f'%((E_step,M_step,self.chi2,self.chi2/self.dof))
                       print(description)
                       sys.stdout.write("\033[F")

                   if self.CHI2[-2]-self.CHI2[-1]<0:
                       print 'Problem in M-step'

               else:

                   Iteration2=0
                   critere2=[]
                   self.comp_chi2()
                   critere2.append(N.inf)
                   critere2.append(self.chi2)

                   while critere2[Iteration2]-critere2[Iteration2+1]>1e-6:
    
                        self.Minimisation_step(Cholesky=CHOLESKY)
                        M_step+=1
                        self.comp_chi2()
               
                        self.CHI2.append(self.chi2)
                        
    
                        #self.AALPHA.append(copy.deepcopy(self.alpha))
                        #self.MM0.append(copy.deepcopy(self.M0))
                    
        
                        self.separate_alpha_M0()
        
                        Iteration2+=1
                        critere2.append(self.chi2)
    
                        if PRINT:
                             description='E-step:%i    M-step:%i    Chi2:%f    Chi2/dof:%f'%((E_step,M_step,self.chi2,self.chi2/self.dof))
                             print(description)
                             sys.stdout.write("\033[F")
    
                        if Iteration2==Max_iter:
                             break
        
                        if self.CHI2[-2]-self.CHI2[-1]<0:
                             print 'Problem in M-step'
    
                        self.separate_slopes_and_data()
                   
    
               self.Expectation_step() 
               E_step+=1
               #self.HH.append(copy.deepcopy(self.h))
               self.comp_chi2()
               self.separate_slopes_and_data()
      
               self.CHI2.append(self.chi2)
               Iteration1+=1
               critere1.append(self.chi2)
               
               if PRINT:
                    description='E-step:%i    M-step:%i    Chi2:%f    Chi2/dof:%f'%((E_step,M_step,self.chi2,self.chi2/self.dof))
                    print(description)
                    sys.stdout.write("\033[F")

               if self.CHI2[-2]-self.CHI2[-1]<0:
                    print 'Problem in E-step'

               if Iteration1==Max_iter:
                    break
               
               if self.Parallel and Iteration1%10000==0:
                   if self.rank==0:
                       name_pkl='save_inter_sed_%i.pkl'%(Iteration1)
                       dic_data={'Alpha':self.Alpha,
                                 'xplus':self.xplus,
                                 'M0':self.M0,
                                 'Grey':self.delta_M_GREY,
                                 'disp_matrix':self.disp_matrix,
                                 'chi2':self.CHI2}
                       File=open(name_pkl,'w')        
                       cPickle.dump(dic_data,File)
                       File.close()
                       del dic_data
                       
                   self.comm.Barrier()

          self.Expectation_step()
          #self.HH.append(copy.deepcopy(self.h))
          self.comp_chi2()
          self.CHI2.append(self.chi2)

          if PRINT:
               print(description+'\n')
               print'**********************************************************************'
               print '                    End of Minimization !'
               print'********************************************************************** \n'



     def Compute_traditional_chi2(self,disp_added=None):



          if disp_added is None:
               disp_added=self.disp_added

            
          residuals = []
          self.W = []

          if self.SPARSE:
              self.COVY=[]
              for i in range(self.Number_loop):
                  self.COVY.append(copy.deepcopy(self.CovY[i].toarray()))
          else:
              self.COVY=copy.deepcopy(self.CovY)

          Chi2=0.

          self.separate_slopes_and_data()

          for j in range(self.Number_loop):
              if self.Parallel:
                  sn=self.START+j
              else:
                  sn=j
              residuals.append(copy.deepcopy(self.Y[j]))
              if self.intrinsic:
                  for cor in range(len(self.Alpha[0])):
                      residuals[j]-=self.Alpha[:,cor]*self.Data[j,cor]

              residuals[j]-=self.M0

              if self.Color or self.CCM:
                  residuals[j]-=self.Av[sn]*self.reddening_law

              if self.delta_M_grey:
                  residuals[j]-=self.delta_M_GREY[sn]
         
              if self.intrinsic:
                  self.COVY[j]+=N.dot(self.Alpha,N.dot(self.CovX[sn],self.Alpha.T))

              self.W.append(N.linalg.inv(self.COVY[j]+disp_added**2*self.disp_matrix))
              
              Chi2+=N.dot(N.matrix(residuals[j]),N.dot(self.W[j],N.matrix(residuals[j]).T))

          self.W=N.array(self.W)
          self.COVY=N.array(self.COVY)

          if self.Parallel:
              rankF = N.array(Chi2)
              self.chi2_traditional = N.zeros(1)
              self.comm.Allreduce(rankF,self.chi2_traditional, op=self.MPI.SUM)
              residuals = N.array(residuals)
              self.residuals = N.zeros((self.N_sn,self.N_bin))
              self.comm.Allgatherv(residuals,self.residuals)
              self.comm.Barrier()
           
          else:
              self.residuals=N.array(residuals)
              self.chi2_traditional=sum(Chi2)


     #############################################################       
     #############################################################
    
     #############################################################
     #  Compute dispertion matrix
     #############################################################        



     def measured_dispersion_matrix(self):

          if self.SPARSE:
              raise NameError('numpy array is needed for this fonction')


          if self.EMFA_disp_matrix and not self.Parallel:
              W=N.zeros(N.shape(self.COVY))
              for i in range(len(self.residuals)):
                  W[i]=N.linalg.inv(self.COVY[i])
              emfa=EMfa_cov.EMfa_covariant(self.residuals,W)
              emfa.converge(10,niter=self.NITER_emfa,Lambda_init=self.LAMBDA_INIT)

              self.LAMBDA_INIT=emfa.Lambda
              self.NITER_emfa=50
              return N.dot(emfa.Lambda,emfa.Lambda.T)


          
          else:
              if self.Parallel:
                  COV=N.zeros(N.shape(self.disp_matrix))
                  cov=N.sum(self.COVY,axis=0)
                  self.comm.Allreduce(cov,COV)
                  self.comm.Barrier()
              else:
                  COV=N.sum(self.COVY,axis=0)

              measured_matrix=( N.dot(self.residuals.T,self.residuals) - COV ) /len(self.residuals)
              # remove negative values from the matrix
              valps,vecps=N.linalg.eig(measured_matrix)
              # Solve for numerical blow-up
              valps=N.real(valps)
              vecps=N.real(vecps)
              valps[N.nonzero(valps<0)]=[0]*len(valps[N.nonzero(valps<0)])
              return N.dot(N.dot(vecps,N.diag(valps)),vecps.T)


    
     def _comp_REML_approx(self):
          if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')


          '''computes 2* log*REML wher REML stands for restricted maximum likelihood'''

          reml=0
        
          for j in range(self.Number_loop) :
               if self.Parallel:
                   sn=self.START+j
               else:
                   sn=j
               residuals=(self.residuals[sn])
               reml += N.dot(residuals.T,N.dot(self.W[j],residuals))
               # the determinant blows up for large matrices, so go to decomposition
               valps=N.linalg.eigvals(self.W[j])
               reml += -N.sum(N.log(valps[N.nonzero(valps>0)]))

          if self.Parallel:
              REML=N.zeros(1)
              self.comm.Allreduce(reml,REML, op=self.MPI.SUM)
              REML=REML[0]
              self.comm.Barrier()              
          else:
              REML=reml
          return REML


     def _comp_REML(self):

         if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')

          # compute the full REML like Guy & al. 2010
         
         reml=0
         self.separate_alpha_M0()
         self.Compute_traditional_chi2()
         print 'start to compute full REML \n'
         
         for j in range(self.Number_loop) :
             if self.Parallel:
                 sn=self.START+j
             else:
                 sn=j

             residuals=(self.residuals[sn])
             reml += N.dot(residuals.T,N.dot(self.W[j],residuals))
             reml -= N.log(N.linalg.det(self.W[j]))
             reml += N.log(N.linalg.det(N.dot(self.alpha.T,N.dot(self.W[j],self.alpha))))
             
         if self.Parallel:
             REML=N.zeros(1)
             self.comm.Allreduce(reml,REML, op=self.MPI.SUM)
             REML=REML[0]
             self.comm.Barrier()
         else:
             REML=reml

         print 'end to compute full REML'                              
         
         return REML
        

     def disp_function(self,d):
          self.converge_chi2(Max_iter=5000,DISP_ADDed=d,PRINT=False)
          #self.Compute_traditional_chi2(disp_added=d)
          return ((self.chi2/self.dof)-1.)



     def build_dispersion_matrix(self,MAX_iter=150,Normalise_chi2=False):

          if self.SPARSE:
             raise NameError('numpy array is needed for this fonction')


          self.disp_added=1.0
     
          self.Compute_traditional_chi2()
        
          controle_reml=0
          n=0
          REML_old=N.inf
          if not self.Parallel:
              REML=self._comp_REML_approx()
          else:
              REML=MAX_iter+1
          REMLs=[REML_old,REML]
          self.REML_save=[]
          if not self.Parallel:
              self.REML_save.append(self._comp_REML_approx())
          else:
              self.REML_save.append(MAX_iter)


          A_copy=copy.deepcopy(self.A)
          Disp_copy=copy.deepcopy(self.disp_matrix)
          H_copy=copy.deepcopy(self.h)
          print "\nCurrent REML:"
          while ( N.abs(REML-REML_old)>0.001 ) and n<=2:

               print n, REML

               self.disp_matrix=self.measured_dispersion_matrix()
               
               if Normalise_chi2:
                    self.disp_added=optimize.fsolve(self.disp_function,self.disp_added,maxfev=200)
                    self.Compute_traditional_chi2()
               else:
                    self.converge_chi2(Max_iter=1,DISP_ADDed=None,PRINT=False,BLOC_DIAGONAL=False)
                    self.Compute_traditional_chi2()
               REML_old=REML
               if not self.Parallel:
                   REML=self._comp_REML_approx()
               else:
                   REML-=1

                              
               if REML<min(self.REML_save):
                   print 'prout'
                   A_copy=copy.deepcopy(self.A)
                   Disp_copy=copy.deepcopy(self.disp_matrix)
                   H_copy=copy.deepcopy(self.h)

               self.REML_save.append(REML)


               if N.abs(REML-REMLs[-2])<0.00001: n+=1
               REMLs.append(REML)

               controle_reml+=1
               if controle_reml>MAX_iter:
                    break

               if N.abs(REML-REMLs[-2])<0.00001: n+=1
               REMLs.append(REML)


          #self.A=A_copy
          #self.disp_matrix=Disp_copy
          #self.h=H_copy

          if Normalise_chi2:
              self.disp_added=optimize.fsolve(self.disp_function,self.disp_added,maxfev=200)
              self.Compute_traditional_chi2()
          else:
              self.converge_chi2(Max_iter=1,DISP_ADDed=None,PRINT=False,BLOC_DIAGONAL=False)
              self.Compute_traditional_chi2()
              
          if not self.Parallel:
              REML=self._comp_REML_approx()
          else:
              REML-=1

          self.REML_save.append(REML)
          print REML
          



     def reload_WY(self,DISP_ADDED=None):
          
          if DISP_ADDED is not None:
               disp_add=DISP_ADDED
          else:
               disp_add=self.disp_added
               
               
          if self.SPARSE:
              print 'be carefull you DO NOT add dispertion matrix (sparse matrix)'
               
          else:
              for j in range(self.Number_loop): 
                  if self.Parallel:
                      sn=self.START+j
                  else:
                      sn=j
                  self.WY[j]=N.linalg.inv(self.CovY[j]+((disp_add**2)*self.disp_matrix)) 
                  
              if self.Parallel:
                  self.comm.Barrier()

           
    
     #############################################################       
     #############################################################
    

     ############################################################
     #  compute RMS wRMS in terms of bands
     ############################################################
      

     def comp_stat(self):
         
         self.separate_slopes_and_data()
         self.Compute_traditional_chi2()

          #if self.Parallel:
              #WY=N.zeros((self.N_sn,self.N_bin,self.N_bin))
              #self.comm.Allgatherv(self.W,WY)
              #self.comm.Barrier()
         vy = []
         for i in range(self.Number_loop):
             vy.append(N.diag(self.COVY[i]))
             
         if self.Parallel:
             VY=N.zeros((self.N_sn,self.N_bin))
             vy=N.array(vy)
             self.comm.Allgatherv(vy,VY)
             self.comm.Barrier()

         else:
             VY=N.array(vy)
              #WY=self.WY

          #WRMS=comp_wRMS(self.residuals,WY)
          #WRMS.comp_wRMS()
          #self.WRMS=copy.deepcopy(WRMS.wRMS)
          #del WRMS
         self.WRMS=N.zeros(self.N_bin)
         dof=self.N_sn-len(self.Data[0])-1
         if self.Color:
             dof -=1
         for Bin in range(self.N_bin):
             self.WRMS[Bin]=H.comp_rms(self.residuals[:,Bin], dof, err=False, variance=VY[:,Bin])

     ############################################################
     #  compute statistical error on slopes with Jackknife
     ############################################################


     def Jackknife(self,MAX_iter=5000):
         
          self.separate_slopes_and_data()

          if self.intrinsic:
              self.alpha_Jackknife=N.zeros((self.N_sn,self.N_bin,len(self.Data[0])))
          else:
              self.alpha_Jackknife=0

          self.M0_Jackknife=N.zeros((self.N_sn,self.N_bin))
          self.reddening_law_Jackknife=N.zeros((self.N_sn,self.N_bin))
          self.RV_Jackknife=N.zeros(self.N_sn)

          self.Filter_Jackknife=N.array([True]*self.N_sn)
        
          print'**********************************************************************'
          print '                      Start of Jackknife:'
          print'********************************************************************** \n'
          

          for sn in range(self.N_sn):
            
               description='supernovae %i/%i' %((sn+1,self.N_sn))
               print(description)
               sys.stdout.write("\033[F")

               self.Filter_Jackknife[sn]=False

               if self.intrinsic:
                   Data=self.Data[self.Filter_Jackknife]
                   COVX=self.CovX[self.Filter_Jackknife]
                   xplus=self.xplus[self.Filter_Jackknife]
               else:
                   Data=None
                   COVX=None
                   xplus=None

               if self.DIAG:
                   DY=self.dY
                   COVY=None
               else:
                   DY=None
                   COVY=self.CovY

               GF=global_fit(self.Y[self.Filter_Jackknife],self.wavelength,data=Data,CovX=COVX,dY=DY,CovY=COVY,dm_z=self.dm_z,
                             alpha0=self.Alpha,reddening_law=self.reddening_law,M00=self.M0,H0=xplus,B_V=self.Av[self.Filter_Jackknife],
                             Delta_M0=self.delta_M_GREY[self.Filter_Jackknife],Disp_matrix_Init=(self.disp_added**2)*self.disp_matrix,
                             Color=self.Color,delta_M_grey=self.delta_M_grey,CCM=self.CCM,Parallel=self.Parallel)


               GF.converge_chi2(Max_iter=MAX_iter,PRINT=False) 
               
               GF.separate_slopes_and_data()

               if self.CCM and self.Color:
                    GF.converge_CCM_fit()

               if self.intrinsic:
                   self.alpha_Jackknife[sn]=GF.Alpha

               self.M0_Jackknife[sn]=GF.M0
               self.reddening_law_Jackknife[sn]=GF.reddening_law
               if self.CCM:
                    self.RV_Jackknife[sn]=1./GF.inv_Rv

               self.Filter_Jackknife[sn]=True

          print(description+'\n')
          print'**********************************************************************'
          print '                       End of Jackknife !'
          print'********************************************************************** \n'
 

        
     def compute_slopes_error_with_jackknife(self):

          self.separate_slopes_and_data()

          if self.intrinsic:
              self.alpha_err_Jackknife=N.zeros(N.shape(self.Alpha))
          else:
              self.alpha_err_Jackknife=0

          self.M0_err_Jackknife=N.zeros(self.N_bin)
          self.reddening_law_err_Jackknife=N.zeros(self.N_bin)
          self.RV_err_Jackknife=0.

          for sn in range(self.N_sn):

              if self.intrinsic:
                  for corr in range(len(self.Data[0])):
                      self.alpha_err_Jackknife[:,corr] += (self.alpha_Jackknife[sn][:,corr]-self.Alpha[:,corr])**2

              self.M0_err_Jackknife += (self.M0_Jackknife[sn] - self.M0)**2
              self.reddening_law_err_Jackknife += (self.reddening_law_Jackknife[sn] - self.reddening_law)**2

              if self.CCM:
                  self.RV_err_Jackknife += (self.RV_Jackknife[sn] - 1./self.inv_Rv)**2

          cst=(self.N_sn-1.)/float(self.N_sn)

          self.alpha_err_Jackknife = N.sqrt(cst*self.alpha_err_Jackknife)
          self.M0_err_Jackknife = N.sqrt(cst*self.M0_err_Jackknife)
          self.reddening_law_err_Jackknife = N.sqrt(cst*self.reddening_law_err_Jackknife)
          self.RV_err_Jackknife = N.sqrt(cst*self.RV_err_Jackknife)


     
     def profile_chi2(self):
        
         self.separate_slopes_and_data()
        
         
         if self.CCM or self.Color:
             AV=self.Av
             RED=self.reddening_law
        
         else:
             AV=None
             RED=None
 
         if self.delta_M_grey:
             DM_grey=self.delta_M_GREY
         else:
             DM_grey=None



         VC=Map_chi2(self.Data,self.Y,self.CovX,self.dY,self.wavelength,Dm_z=self.dm_z,
                     Alpha0=self.Alpha,Reddening_law=RED,MM00=self.M0,
                     HH0=self.xplus,BB_V=AV,delta_M0=DM_grey,inv_RV=self.inv_Rv,Disp_Matrix_Init=self.disp_added**2*self.disp_matrix,
                     COlor=self.Color,Delta_M_grey=self.delta_M_grey,ccm=self.CCM)

         if self.intrinsic:
             VC.variate_chi2_alpha()
             self.Alpha_variation=VC.alpha_variation
             self.chi2_Alpha_variation=VC.chi2_alpha_variation
         
         VC.variate_chi2_M0()
         self.M0_variation=VC.M0_variation
         self.chi2_M0_variation=VC.chi2_M0_variation
        
         if self.intrinsic:
             VC.variate_chi2_xplus()
             self.xplus_variation=VC.xplus_variation
             self.chi2_xplus_variation=VC.chi2_xplus_variation

        
         if self.Color:
             VC.variate_chi2_Red_law_agno()
             self.red_law_variation=VC.red_law_variation
             self.chi2_red_law_variation=VC.chi2_red_law_variation
            
         if self.Color or self.CCM:
             VC.variate_chi2_Av()
             self.Av_variation=VC.Av_variation
             self.chi2_Av_variation=VC.chi2_Av_variation
             
         if self.delta_M_grey:
             VC.variate_chi2_grey()
             self.grey_variation=VC.grey_variation
             self.chi2_grey_variation=VC.chi2_grey_variation

         if self.CCM:

             if self.Color:
                 AV=self.Av_cardelli
                 VC_bis=Map_chi2(self.Data,self.Y,self.CovX,self.dY,self.wavelength,Dm_z=self.dm_z,
                     Alpha0=self.Alpha,Reddening_law=RED,MM00=self.M0,
                     HH0=self.xplus,BB_V=AV,delta_M0=DM_grey,inv_RV=self.inv_Rv,Disp_Matrix_Init=self.disp_added**2*self.disp_matrix,
                     COlor=self.Color,Delta_M_grey=self.delta_M_grey,ccm=self.CCM)

                 #VC_bis=Map_chi2(self.Data,self.Y,self.CovX,self.dY,self.wavelength,dm_z=self.dm_z,
                 #                alpha0=self.Alpha,reddening_law=RED,M00=self.M0,
                 #                H0=self.xplus,B_V=AV,Delta_M0=DM_grey,inv_Rv=self.inv_Rv,Disp_matrix_Init=self.disp_added**2*self.disp_matrix,
                 #                Color=self.Color,delta_M_grey=self.delta_M_grey,CCM=self.CCM)

                 VC_bis.variate_chi2_inv_Rv()
                 self.inv_Rv_variation=VC_bis.inv_Rv_variation
                 self.chi2_inv_Rv_variation=VC_bis.chi2_inv_Rv_variation
 
             else:
                 VC.variate_chi2_inv_Rv()
                 self.inv_Rv_variation=VC.inv_Rv_variation
                 self.chi2_inv_Rv_variation=VC.chi2_inv_Rv_variation
 



        
    #############################
    #  separate the fit result
    #############################

     def separate_slopes_and_data(self,Everithing=False):

          self.M0=copy.deepcopy(self.A[:,0])
          self.reddening_law=N.zeros(self.N_bin)
          self.Av=N.zeros(self.N_sn)
          self.delta_M_GREY=N.zeros(self.N_sn)

          if not self.intrinsic:
              self.Alpha=None
              self.Data=None
              self.xplus=None
        
          if self.Color or self.CCM:
               if self.delta_M_grey:
                    self.delta_M_GREY=copy.deepcopy(self.h[:,1])
                    self.Av=copy.deepcopy(self.h[:,2])
                    self.reddening_law=copy.deepcopy(self.A[:,2])
                    if self.intrinsic:
                        self.Alpha=copy.deepcopy(self.A[:,3:])
                        self.Data=copy.deepcopy(self.data[:,3:])
                        self.xplus=copy.deepcopy(self.h[:,3:])
               else:
                    self.Av=copy.deepcopy(self.h[:,1])
                    self.reddening_law=copy.deepcopy(self.A[:,1])
                    if self.intrinsic:
                        self.Alpha=copy.deepcopy(self.A[:,2:])
                        self.Data=copy.deepcopy(self.data[:,2:])
                        self.xplus=copy.deepcopy(self.h[:,2:])
          else:
               if self.delta_M_grey:
                    self.delta_M_GREY=copy.deepcopy(self.h[:,1])
                    if self.intrinsic:
                        self.Alpha=copy.deepcopy(self.A[:,2:])
                        self.Data=copy.deepcopy(self.data[:,2:])
                        self.xplus=copy.deepcopy(self.h[:,2:])

               else:
                   if self.intrinsic:
                       self.Alpha=copy.deepcopy(self.A[:,1:])
                       self.Data=copy.deepcopy(self.data[:,1:])
                       self.xplus=copy.deepcopy(self.h[:,1:])

    #############################
    #  merge fit result
    #############################

     def merge_slopes_and_data(self):

          self.A[:,0]=copy.deepcopy(self.M0)

          if self.Color or self.CCM:
               if self.delta_M_grey:
                    self.h[:,1]=copy.deepcopy(self.delta_M_GREY)
                    self.h[:,2]=copy.deepcopy(self.Av)
                    self.A[:,2]=copy.deepcopy(self.reddening_law)
                    if self.intrinsic:
                        self.A[:,3:]=copy.deepcopy(self.Alpha)
                        self.data[:,3:]=copy.deepcopy(self.Data)
                        self.h[:,3:]=copy.deepcopy(self.xplus)
               else:
                    self.h[:,1]=copy.deepcopy(self.Av)
                    self.A[:,1]=copy.deepcopy(self.reddening_law)
                    if self.intrinsic:
                        self.A[:,2:]=copy.deepcopy(self.Alpha)
                        self.data[:,2:]=copy.deepcopy(self.Data)
                        self.h[:,2:]=copy.deepcopy(self.xplus)
          else:
               if self.delta_M_grey:
                    self.h[:,1]=copy.deepcopy(self.delta_M_GREY)
                    if self.intrinsic:
                        self.A[:,2:]=copy.deepcopy(self.Alpha)
                        self.data[:,2:]=copy.deepcopy(self.Data)
                        self.h[:,2:]=copy.deepcopy(self.xplus)

               else:
                   if self.intrinsic:
                       self.A[:,1:]=copy.deepcopy(self.Alpha)
                       self.data[:,1:]=copy.deepcopy(self.Data)
                       self.h[:,1:]=copy.deepcopy(self.xplus)




    #######################################
    # Run the fit with disp matrix or not
    #######################################

     def run_Global_fit(self,MAX_ITER=5000,Max_iter_disp=150,Init=False,Addisp=False,Bloc_diag=False,JACK=False,Norm_Chi2=False,Map_chi2=True):
          
          if Init:
               self.initialization_slopes_and_mean_spectrum()

          self.converge_chi2(Max_iter=MAX_ITER,BLOC_DIAGONAL=Bloc_diag)
        
          if Addisp:
           
               calls=0
               if (self.chi2/(self.dof))<1.:
                    print 'Chi2/dof is already more small than 1'
               else:

                    print self.chi2/(self.dof)
                    self.build_dispersion_matrix(MAX_iter=Max_iter_disp,Normalise_chi2=Norm_Chi2)

               self.Full_REML=self._comp_REML()

          if self.CCM and self.Color:
               self.converge_CCM_fit(PRINT=True)

          if JACK:
               self.Jackknife(MAX_iter=MAX_ITER)
               self.compute_slopes_error_with_jackknife()


          if Map_chi2:
              self.profile_chi2()

          self.separate_slopes_and_data()
                    


    
     ############################################################
     #  write the intermediate result and fit result in pkl file
     ############################################################


     def write_pkl_file(self,name_pkl):

          dic_data=self.__dict__
          File=open(name_pkl,'w')        
          cPickle.dump(dic_data,File)
          File.close()
          del dic_data

     ###########################
     #  control plot
     ###########################

     def plot_control_chi2(self):

          P.figure()
          X=N.linspace(1,len(self.CHI2),len(self.CHI2))
          P.plot(X,self.CHI2,'b',label=r'$\chi^2$ ODR')
          P.xlabel('iteration')
          P.ylabel(r'$\chi ^2$')
          P.legend()


     def plot_control_slopes(self,DisP=False):

          phot=['U','B','V','R','I']

          for Bin in range(self.N_bin):
               if self.N_bin==5:
                    self.plot_control_Bin(Bin,DISP=DisP)
               else:
                    if Bin%20==0:
                         self.plot_control_Bin(Bin,DISP=DisP)

          P.show()


     def plot_control_Bin(self,BIN,DISP=False):
          
          Color= self.Color or self.CCM

          if DISP:
               disp=self.disp_added*self.disp_matrix[BIN,BIN]
          else:
               disp=0.

          self.separate_slopes_and_data()

          if Color:
              if self.intrinsic:
                  corr= len(self.Data[0])+1  
              else:
                  corr= 1  
          else:
              if self.intrinsic:
                  corr= len(self.Data[0])
              else:
                  corr=0

          Y=N.zeros((corr,self.N_sn))


          if Color:
               Y[0]=copy.deepcopy(self.Y[:,BIN])
               for i in range(corr-1):
                   if self.intrinsic:
                       Y[0]-=self.Data[:,i]*self.Alpha[BIN,i]
               if self.delta_M_grey:
                    Y[0]-=self.delta_M_GREY
                    
               for i in range(corr-1):
                    Y[i+1]=copy.deepcopy(self.Y[:,BIN])
                    Y[i+1]-=self.Av*self.reddening_law[BIN]
                    for j in range(corr-1):
                         if i!=j:
                             if self.intrinsic:
                                 Y[i+1]-=self.Data[:,j]*self.Alpha[BIN,j]
                    if self.delta_M_grey:
                         Y[i+1]-=self.delta_M_GREY

          else:                   
               for i in range(corr):
                    Y[i]=copy.deepcopy(self.Y[:,BIN])
                    for j in range(corr):
                         if i!=j:
                             if self.intrinsic:
                                 Y[i]-=self.Data[:,j]*self.Alpha[BIN,j]
                    if self.delta_M_grey:
                         Y[i]-=self.delta_M_GREY

     
          


          for i in range(corr):
               if i==0 and Color:
                    P.figure(figsize=(10,10))
                    P.errorbar(self.Av,Y[i],linestyle='',xerr=None,yerr=N.sqrt(self.dY[:,BIN]**2+disp),ecolor='blue',alpha=0.7,marker='.',zorder=0)
                    P.scatter(self.Av,Y[i],c='b',s=10)
                    P.plot(self.Av,self.reddening_law[BIN]*self.Av+self.M0[BIN],'r')
                    P.xlabel('$A_V$',fontsize=16)
                    if self.delta_M_grey:
                         P.ylabel('$M(t,\lambda) - \sum_i \\alpha_i(t,\lambda) q_i - \Delta M_{grey}(t)$',fontsize=16)
                    else:
                         P.ylabel('$M(t,\lambda) - \sum_i \\alpha_i(t,\lambda) q_i $',fontsize=16)

                    P.title(r'$\lambda=%i \AA$'%(self.wavelength[BIN]))
                    P.gca().invert_yaxis()
               
               if self.intrinsic:
                   if i!=0 and Color:
                        P.figure(figsize=(10,10))
                        
                        P.errorbar(self.Data[:,i-1],Y[i],linestyle='',xerr=None,yerr=N.sqrt(self.dY[:,BIN]**2+disp),ecolor='blue',alpha=0.7,marker='.',zorder=0)
                        P.scatter(self.Data[:,i-1],Y[i],c='b',s=10)
                        P.plot(self.Data[:,i-1],self.Alpha[BIN,i-1]*self.Data[:,i-1]+self.M0[BIN],'r')
                        P.xlabel('$q_{%i}$'%(i),fontsize=16)
                        if self.delta_M_grey:
                             P.ylabel('$M(t,\lambda) - \sum_{i\\ne%i} \\alpha_i(t,\lambda) q_i -A_V S(\lambda) - \Delta M_{grey}(t)$'%(i),fontsize=16)
                        else:
                             P.ylabel('$M(t,\lambda) - \sum_{i\\ne%i} \\alpha_i(t,\lambda) q_i -A_V S(\lambda)$'%(i),fontsize=16)
                             
                        P.title(r'$\lambda=%i \AA$'%(self.wavelength[BIN]))
                        P.gca().invert_yaxis()
                   else:
                        P.figure(figsize=(10,10))
                        P.errorbar(self.Data[:,i],Y[i],linestyle='',xerr=None,yerr=N.sqrt(self.dY[:,BIN]**2+disp),ecolor='blue',alpha=0.7,marker='.',zorder=0)
                        P.scatter(self.Data[:,i],Y[i],c='b',s=10)
                        P.plot(self.Data[:,i],self.Alpha[BIN,i-1]*self.Data[:,i]+self.M0[BIN],'r')
                        
                        P.xlabel('$q_{%i}$'%(i+1),fontsize=16)
                        if self.delta_M_grey:
                             P.ylabel('$M(t,\lambda) - \sum_{i\\ne%i} \\alpha_i(t,\lambda) q_i - \Delta M_{grey}(t)$'%(i+1),fontsize=16)
                        else:
                             P.ylabel('$M(t,\lambda) - \sum_{i\\ne%i} \\alpha_i(t,\lambda) q_i $'%(i+1),fontsize=16)
                        P.title(r'$\lambda=%i \AA$'%(self.wavelength[BIN]))
    
    
                        P.gca().invert_yaxis()
    


     def plot_control_spectra(self,interactif=False,sn_name=None):
        
          Color=self.Color or self.CCM


          Mag_corrected=N.zeros(N.shape(self.Y))
          STD=N.zeros(self.N_bin)
          data=copy.deepcopy(self.data)
        
          if Color:
               data[:,(1+self.grey)]=self.h[:,(1+self.grey)]
            
          if self.delta_M_grey:
               data[:,1]=self.h[:,1]
            
          for sn in range(self.N_sn):
               Mag_corrected[sn]=self.Y[sn]-N.dot(self.A,N.matrix(data[sn]).T).T[0] +self.M0 
            
          for Bin in range(self.N_bin):
               STD[Bin]=N.std(Mag_corrected[:,Bin])

          P.figure()

          P.subplots_adjust(hspace=0.001)
          for sn in range(self.N_sn):

               P.subplot(2,1,1)
               P.plot(self.wavelength,Mag_corrected[sn]+19.2)
        
          P.title('spectrum corrected')

          P.ylabel('all spectrum (Mag AB + cst)')
          P.ylim(-2.8,5.2)
          P.xticks([2500.,9500.],['toto','pouet'])
          P.xlim(self.wavelength[0]-60,self.wavelength[-1]+60)
          P.gca().invert_yaxis()

          if interactif:
               from ToolBox import MPL
               X=N.ones(self.N_sn)*self.wavelength[0]
               scat=P.scatter(X,Mag_corrected[:,0]+19.2,s=20)
               browser=MPL.PointBrowser(X, Mag_corrected[:,0]+19.2,sn_name,scat)
        
          P.subplot(2,1,2)
          P.plot(self.wavelength,STD,'b',label=r'STD')
          P.ylabel('STD')
          P.xlabel('wavelength [$\AA$]')
          P.xlim(self.wavelength[0]-60,self.wavelength[-1]+60)
          P.ylim(0,0.25)
          P.legend()
          P.show()



     def plot_matrix(self,ylabel=r'$\rho$',title='correlation matrix',cmap=P.cm.jet):
    
          matrix=N.zeros(N.shape(self.disp_matrix))

          disp_matrix=self.disp_matrix*self.disp_added**2
          self.diag_std_disp_matrix=N.sqrt(N.diag(disp_matrix))

          for Bin_i in range(len(self.wavelength)):
               for Bin_j in range(len(self.wavelength)):
                    matrix[Bin_i,Bin_j]=disp_matrix[Bin_i,Bin_j]/(self.diag_std_disp_matrix[Bin_i]*self.diag_std_disp_matrix[Bin_j])
 


          values = [N.diag(matrix,k=i) for i in range(len(matrix))]
          
          means=map(N.mean,values)
          stds=map(N.std,values)
          med,nmad=N.array([median_stats(x) for x in values]).T
     
    
          #Plot the matrix
          wlength=[self.wavelength[0],self.wavelength[-1],self.wavelength[-1],self.wavelength[0]]
          fig = P.figure(dpi=150,figsize=(8,8))
          #fig = figure(figsize=(12,12))
          ax = fig.add_axes([0.08,0.09,0.88,0.88]) #title=title
          im = ax.imshow(matrix,cmap=cmap,extent=wlength,interpolation='nearest')
          cb = fig.colorbar(im)
          cb.set_label(ylabel,size='x-large')
          ax.set_xlabel(r'Wavelength [$\AA$]',size='large')
          ax.set_ylabel(r'Wavelength [$\AA$]',size='large')
    
     def plot_diag_matrix(self):

          P.figure()

          P.plot(self.wavelength,self.diag_std_disp_matrix)
          P.xlabel(r'Wavelength [$\AA$]')
          P.ylabel(r'$\sqrt{diag(D)}$')
          P.ylim(0,0.2)


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


#==============================================================================
# Map Chi2 ODR after a Global MultiLinear fit 
#==============================================================================


class Map_chi2(global_fit):

    def __init__(self,Data,YY,CovXX,dYY,Wavelength,
                 Dm_z=None,Alpha0=None,Reddening_law=None,
                 MM00=None,HH0=None,BB_V=None,delta_M0=None,
                 inv_RV=None,Disp_Matrix_Init=None,
                 COlor=True,Delta_M_grey=True,ccm=True):


        global_fit.__init__(self,YY,dYY,Wavelength,dm_z=Dm_z,data=Data,CovX=CovXX,
                            alpha0=Alpha0,reddening_law=Reddening_law,M00=MM00,H0=HH0,
                            B_V=BB_V,Delta_M0=delta_M0,Disp_matrix_Init=Disp_Matrix_Init,
                            Color=COlor,delta_M_grey=Delta_M_grey,CCM=ccm)


        if inv_RV is not None:
            self.inv_Rv=inv_RV
        self.Range=50

    
    def variate_chi2_alpha(self):
        
        if not self.intrinsic:
            return

        self.separate_slopes_and_data()
         
        Alpha_save=copy.deepcopy(self.Alpha)

        self.alpha_variation=N.zeros((self.N_bin,len(self.Alpha[0]),self.Range))
        self.chi2_alpha_variation=N.zeros((self.N_bin,len(self.Alpha[0]),self.Range))

        for Bin in range(self.N_bin):
            print Bin
            for Comp in range(len(self.Alpha[0])):
                self.alpha_variation[Bin,Comp]=N.linspace(Alpha_save[Bin,Comp]-3.*N.std(Alpha_save[:,Comp]),Alpha_save[Bin,Comp]+3.*N.std(Alpha_save[:,Comp]),self.Range)
                     
                for R in range(self.Range):
                     
                    self.Alpha[Bin,Comp]=copy.deepcopy(self.alpha_variation[Bin,Comp,R])
                    self.merge_slopes_and_data()
          
                    A=copy.deepcopy(self.A)
                    h=copy.deepcopy(self.h)
         
                    residu1=N.zeros((self.N_sn,self.N_bin))
                    residu2=N.zeros((self.N_sn,self.N_comp))
                    chi2=N.zeros(self.N_sn)
                     
                    for sn in range(self.N_sn):
                         
                        residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                        residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                        chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))

    
                    self.chi2_alpha_variation[Bin,Comp,R]=chi2.sum()

                self.Alpha=copy.deepcopy(Alpha_save)
                self.merge_slopes_and_data()

    def variate_chi2_M0(self):
         
        self.separate_slopes_and_data()
         
        M0_save=copy.deepcopy(self.M0)

        self.M0_variation=N.zeros((self.N_bin,self.Range))
        self.chi2_M0_variation=N.zeros((self.N_bin,self.Range))

        for Bin in range(self.N_bin):
            print Bin
            
            self.M0_variation[Bin]=N.linspace(M0_save[Bin]-3.*N.std(M0_save),M0_save[Bin]+3.*N.std(M0_save),self.Range)
                     
            for R in range(self.Range):
                     
                self.M0[Bin]=copy.deepcopy(self.M0_variation[Bin,R])
                self.merge_slopes_and_data()
          
                A=copy.deepcopy(self.A)
                h=copy.deepcopy(self.h)
         
                residu1=N.zeros((self.N_sn,self.N_bin))
                residu2=N.zeros((self.N_sn,self.N_comp))
                chi2=N.zeros(self.N_sn)
                     
                for sn in range(self.N_sn):
                         
                    residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                    residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                    chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))

    
                self.chi2_M0_variation[Bin,R]=chi2.sum()

            self.M0=copy.deepcopy(M0_save)
            self.merge_slopes_and_data()


    def variate_chi2_grey(self):
         
        self.separate_slopes_and_data()
         
        grey_save=copy.deepcopy(self.delta_M_GREY)

        self.grey_variation=N.zeros((self.N_sn,self.Range))
        self.chi2_grey_variation=N.zeros((self.N_sn,self.Range))

        for SN in range(self.N_sn):
            print SN
            
            self.grey_variation[SN]=N.linspace(grey_save[SN]-3.*N.std(grey_save),grey_save[SN]+3.*N.std(grey_save),self.Range)
                     
            for R in range(self.Range):
                     
                self.delta_M_GREY[SN]=copy.deepcopy(self.grey_variation[SN,R])
                self.merge_slopes_and_data()
          
                A=copy.deepcopy(self.A)
                h=copy.deepcopy(self.h)
         
                residu1=N.zeros((self.N_sn,self.N_bin))
                residu2=N.zeros((self.N_sn,self.N_comp))
                chi2=N.zeros(self.N_sn)
                     
                for sn in range(self.N_sn):
                         
                    residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                    residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                    chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))
                        
                    
                self.chi2_grey_variation[SN,R]=chi2.sum()

            self.delta_M_GREY=copy.deepcopy(grey_save)
            self.merge_slopes_and_data()


    def variate_chi2_Av(self):
         
        self.separate_slopes_and_data()
         
        Av_save=copy.deepcopy(self.Av)

        self.Av_variation=N.zeros((self.N_sn,self.Range))
        self.chi2_Av_variation=N.zeros((self.N_sn,self.Range))

        for SN in range(self.N_sn):
            print SN
            
            self.Av_variation[SN]=N.linspace(Av_save[SN]-3.*N.std(Av_save),Av_save[SN]+3.*N.std(Av_save),self.Range)
                     
            for R in range(self.Range):
                     
                self.Av[SN]=copy.deepcopy(self.Av_variation[SN,R])
                self.merge_slopes_and_data()
          
                A=copy.deepcopy(self.A)
                h=copy.deepcopy(self.h)
         
                residu1=N.zeros((self.N_sn,self.N_bin))
                residu2=N.zeros((self.N_sn,self.N_comp))
                chi2=N.zeros(self.N_sn)
                     
                for sn in range(self.N_sn):
                         
                    residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                    residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                    chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))
                        
                    
                self.chi2_Av_variation[SN,R]=chi2.sum()

            self.Av=copy.deepcopy(Av_save)
            self.merge_slopes_and_data()



    def variate_chi2_xplus(self):

        if not self.intrinsic:
            return
         
        self.separate_slopes_and_data()
         
        xplus_save=copy.deepcopy(self.xplus)

        self.xplus_variation=N.zeros((self.N_sn,len(self.Alpha[0]),self.Range))
        self.chi2_xplus_variation=N.zeros((self.N_sn,len(self.Alpha[0]),self.Range))

        for SN in range(self.N_sn):
            print SN
            for Comp in range(len(self.xplus[0])):
                self.xplus_variation[SN,Comp]=N.linspace(xplus_save[SN,Comp]-3.*N.std(xplus_save[:,Comp]),xplus_save[SN,Comp]+3.*N.std(xplus_save[:,Comp]),self.Range)
                     
                for R in range(self.Range):
                     
                    self.xplus[SN,Comp]=copy.deepcopy(self.xplus_variation[SN,Comp,R])
                    self.merge_slopes_and_data()
          
                    A=copy.deepcopy(self.A)
                    h=copy.deepcopy(self.h)
         
                    residu1=N.zeros((self.N_sn,self.N_bin))
                    residu2=N.zeros((self.N_sn,self.N_comp))
                    chi2=N.zeros(self.N_sn)
                     
                    for sn in range(self.N_sn):
                         
                        residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                        residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                        chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))
                        
                    
                    self.chi2_xplus_variation[SN,Comp,R]=chi2.sum()

                self.xplus=copy.deepcopy(xplus_save)
                self.merge_slopes_and_data()




        
    def variate_chi2_Red_law_agno(self):
         
        self.separate_slopes_and_data()
         
        red_law_save=copy.deepcopy(self.reddening_law)

        self.red_law_variation=N.zeros((self.N_bin,self.Range))
        self.chi2_red_law_variation=N.zeros((self.N_bin,self.Range))

        for Bin in range(self.N_bin):
            print Bin
            
            self.red_law_variation[Bin]=N.linspace(red_law_save[Bin]-3.*N.std(red_law_save),red_law_save[Bin]+3.*N.std(red_law_save),self.Range)
                     
            for R in range(self.Range):
                     
                self.reddening_law[Bin]=copy.deepcopy(self.red_law_variation[Bin,R])
                self.merge_slopes_and_data()
          
                A=copy.deepcopy(self.A)
                h=copy.deepcopy(self.h)
         
                residu1=N.zeros((self.N_sn,self.N_bin))
                residu2=N.zeros((self.N_sn,self.N_comp))
                chi2=N.zeros(self.N_sn)
                     
                for sn in range(self.N_sn):
                         
                    residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                    residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                    chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))

    
                self.chi2_red_law_variation[Bin,R]=chi2.sum()

            self.reddening_law=copy.deepcopy(red_law_save)
            self.merge_slopes_and_data()




    def variate_chi2_inv_Rv(self):
         
        self.separate_slopes_and_data()

        a_cardelli,b_cardelli=Astro.Extinction.extinctionParameters(self.wavelength,odonnell=False)

        red_law_save=copy.deepcopy(self.reddening_law)
        inv_Rv_save=copy.deepcopy(self.inv_Rv)

        self.inv_Rv_variation=N.linspace(inv_Rv_save-1,inv_Rv_save+1,self.Range)
        self.chi2_inv_Rv_variation=N.zeros(self.Range)
                     
        for R in range(self.Range):
                     
            self.reddening_law=copy.deepcopy(a_cardelli+self.inv_Rv_variation[R]*b_cardelli)
            self.merge_slopes_and_data()
          
            A=copy.deepcopy(self.A)
            h=copy.deepcopy(self.h)
         
            residu1=N.zeros((self.N_sn,self.N_bin))
            residu2=N.zeros((self.N_sn,self.N_comp))
            chi2=N.zeros(self.N_sn)
                     
            for sn in range(self.N_sn):
                
                residu1[sn]=self.Y[sn]-N.dot(A,N.matrix(h[sn]).T).T 
                residu2[sn]=self.data[sn,1:]-h[sn,1:]
                         
                chi2[sn]= N.dot(N.matrix(residu1[sn]),N.dot(self.WY[sn],N.matrix(residu1[sn]).T))+N.dot(N.matrix(residu2[sn]),N.dot(self.WX[sn],N.matrix(residu2[sn]).T))

            print chi2.sum()
            self.chi2_inv_Rv_variation[R]=chi2.sum()

        self.inv_Rv=copy.deepcopy(inv_Rv_save)
        self.reddening_law=copy.deepcopy(red_law_save)
        self.merge_slopes_and_data()




if __name__=="__main__":

     x=N.linspace(0,10,100)
     z=N.linspace(0,15,100)
     XZ=N.array((x,z))
     y=N.zeros(len(x))
     x_err=N.zeros(len(x))
     y_err=N.zeros(len(x))
    
    
     for i in range(len(x)):
          y[i]=(x[i]+3.*z[i]+5.)
          x_err[i]=2.*N.random.normal()
          y_err[i]=5.*N.random.normal()
          y[i]+=N.random.normal()*y_err[i]
          x[i]+=N.random.normal()*x_err[i]


     M=Multilinearfit(x,y,xerr=x_err,yerr=y_err,covx=None)
     M.Multilinearfit(adddisp=False)
    
    
     P.errorbar(x,y, linestyle='',xerr=x_err,yerr=y_err,ecolor='grey',alpha=1.0,zorder=0)
     P.scatter(x,y,zorder=100,s=50)
     P.plot(x,M.alpha*x+M.M0)

    

     P.show()
    

        
   
 
