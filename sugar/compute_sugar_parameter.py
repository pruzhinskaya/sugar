
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

class aligne_SED:

    def __init__(self,SPECTRA_PKL,sn,ALPHA,M0,Rv):

        self.dic=cPickle.load(open(SPECTRA_PKL))
        Phase=[]
        self.IND=[]
        for j in range(len(self.dic[sn].keys())):
            if self.dic[sn]['%i'%(j)]['phase_salt2']>-12 and self.dic[sn]['%i'%(j)]['phase_salt2']<42:
                Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                self.IND.append('%i'%(j))
        self.Phase=N.array(Phase)
        self.sn=sn
        self.Alpha=ALPHA
        self.M0=M0
        self.Rv=Rv
        
    def align_SED(self):
        Time=N.linspace(-12,42,19)
        DELTA=len(self.Phase)
        self.SED=N.ones((190*len(self.Phase),6))
        for Bin in range(190):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*19:(Bin+1)*19])
            self.SED[:,0][Bin*DELTA:(Bin+1)*DELTA]=SPLINE_Mean(self.Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.Alpha[:,i][Bin*19:(Bin+1)*19])
                self.SED[:,i+3][Bin*DELTA:(Bin+1)*DELTA]=SPLINE(self.Phase)
            self.SED[:,2][Bin*DELTA:(Bin+1)*DELTA]=Astro.Extinction.extinctionLaw(self.dic[self.sn]['0']['X'][Bin],Rv=self.Rv,law='CCM89')

        reorder = N.arange(190*DELTA).reshape(190, DELTA).T.reshape(-1)
        for i in range(len(self.SED[0])):
            self.SED[:,i]=self.SED[:,i][reorder]

    def align_spectra(self):
        self.Y=N.zeros(190*len(self.Phase))
        self.Y_err=N.zeros(190*len(self.Phase))
        DELTA=len(self.Phase)
        for Bin in range(DELTA):
            self.Y[Bin*190:(Bin+1)*190]=self.dic[self.sn][self.IND[Bin]]['Y']
            self.Y_err[Bin*190:(Bin+1)*190]=N.sqrt(self.dic[self.sn][self.IND[Bin]]['V'])



#==============================================================================
# Compute the SED parameters for a given supernovae
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

     def __init__(self,Y,SED,dY=None,CovY=None):

         self.Y=Y
         self.A=SED
         if dY is None and CovY is None:
             self.WY=N.eye(len(self.Y))
         else:
             if dY is not None:
                 self.WY=N.eye(len(self.Y))*1./dY**2
             if CovY is not None:
                 self.WY=N.linalg.inv(CovY)

         self.N_comp=len(self.A[0])-1
    
     ############################################################# 
     # Compute orthogonal projection (E-step)
     #############################################################  


     def separate_alpha_M0(self):
          self.M0=self.A[:,0]
          self.alpha=self.A[:,1:]

     def compute_h(self):

          """
          Compute the true value of the data,
          Av and the grey offset
 
          """
          self.separate_alpha_M0()

          A=N.zeros((self.N_comp,self.N_comp))
          B=N.zeros(self.N_comp)
          H=N.zeros(self.N_comp)
          Y=N.zeros(len(self.Y))
        
          
          Y=self.Y - self.M0
          T=N.dot(self.alpha.T,self.WY.dot(self.alpha))
          A=N.linalg.inv(T)
          B=N.dot(self.alpha.T,self.WY.dot(N.matrix(Y).T))
              
          H=(N.dot(A,N.matrix(B))).T
              
          self.h=H
          self.cov_h=A
          
if __name__=="__main__":

    #dic_at_max=cPickle.load(open('../BEDELL/model_training_SUGAR/model_at_max_3_eigenvector_without_grey.pkl'))
    #dic_sed=cPickle.load(open('../BEDELL/model_training_SUGAR/model_full_time_sed_with_grey_new_kern_twice.pkl'))
    #SPECTRA='../BEDELL/all_BEDELL_data_without_cosmology.pkl'
    #SN=cPickle.load(open('../BEDELL/all_BEDELL_data_without_cosmology.pkl'))
    #HH=[]
    #dic={}
    #for i,SNN in enumerate(SN.keys()):
    #
    #    ASED=aligne_SED(SPECTRA,SNN,dic_sed['alpha'],dic_sed['M0'],dic_at_max['RV'])
    #    ASED.align_SED()
    #    ASED.align_spectra()
    #
    #    GF=global_fit(ASED.Y,ASED.SED,dY=ASED.Y_err,CovY=None)
    #    GF.compute_h()
    #    print GF.h
    #    HH.append(GF.h)
    #    A=N.array(HH[i])
    #    h=A[0]
    #    dic.update({SNN:{'Av':h[1],
    #                   'Grey':h[0],
    #                    'x1':h[2],
    #                    'x2':h[3],
    #                    'x3':h[4]}})

    #File=open('../BEDELL/SUGAR_parameters_without_cosmology.pkl','w')
    #cPickle.dump(dic,File)
    #File.close()

    sps='/sps/snovae/user/leget'
    dic_at_max=cPickle.load(open(sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/model_at_max_3_eigenvector_without_grey_without_MFR_problem_test_RV.pkl'))
    dic_sed=cPickle.load(open(sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/SUGAR_validation_3_eigenvector_CABALLO_test_RV.pkl'))
    SPECTRA=sps+'/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_without_cosmology.pkl'

    HH=[]
    dic={}
    SN=dic_sed['sn_name']
    for i,SNN in enumerate(SN):

        ASED=aligne_SED(SPECTRA,SNN,dic_sed['alpha'],dic_sed['M0'],dic_at_max['RV'])
        ASED.align_SED()
        ASED.align_spectra()

        GF=global_fit(ASED.Y,ASED.SED,dY=ASED.Y_err,CovY=None)
        GF.compute_h()
        print GF.h
        HH.append(GF.h)
        A=N.array(HH[i])
        h=A[0]
        cov_h=GF.cov_h#[1:,1:]
        dic.update({SNN:{'Av':h[1],
                        'Grey':h[0],
                        'x1':h[2],
                        'x2':h[3],
                        'x3':h[4],
                         'cov_h':cov_h}})

    File=open(sps+'/CABALLO/SUGAR_validation/SUGAR_parameters_without_cosmology.pkl','w')
    cPickle.dump(dic,File)
    File.close()
