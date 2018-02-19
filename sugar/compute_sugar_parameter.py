"""compute qi, av, and grey offset from spectral fitting."""

import sys
import scipy.odr as O
import scipy.odr.models as M
from scipy import optimize,linalg,sparse
from scipy.sparse import block_diag
from scipy.misc import derivative
import pylab as P
import numpy as N
import scipy as S
import sugar
import copy
import cPickle
import scipy.interpolate as inter

class aligne_SED:
    """
    Aligne sugar model on observed sed.
    """
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
        Time=N.linspace(-12,48,21)
        DELTA=len(self.Phase)
        self.SED=N.ones((197*len(self.Phase),6))
        for Bin in range(197):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*21:(Bin+1)*21])
            self.SED[:,0][Bin*DELTA:(Bin+1)*DELTA]=SPLINE_Mean(self.Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.Alpha[:,i][Bin*21:(Bin+1)*21])
                self.SED[:,i+3][Bin*DELTA:(Bin+1)*DELTA]=SPLINE(self.Phase)
            self.SED[:,2][Bin*DELTA:(Bin+1)*DELTA]=sugar.extinctionLaw(self.dic[self.sn]['0']['X'][Bin],Rv=self.Rv)

        reorder = N.arange(197*DELTA).reshape(197, DELTA).T.reshape(-1)
        for i in range(len(self.SED[0])):
            self.SED[:,i]=self.SED[:,i][reorder]

    def align_spectra(self):
        self.Y=N.zeros(197*len(self.Phase))
        self.Y_err=N.zeros(197*len(self.Phase))
        DELTA=len(self.Phase)
        for Bin in range(DELTA):
            self.Y[Bin*197:(Bin+1)*197]=self.dic[self.sn][self.IND[Bin]]['Y']
            self.Y_err[Bin*197:(Bin+1)*197]=N.sqrt(self.dic[self.sn][self.IND[Bin]]['V'])


class global_fit:
     """
     Fit h factor on spectra in respect with sugar.
     """
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


    
    
    dic_at_max = cPickle.load(open('data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl'))
    dic_sed = cPickle.load(open('data_output/sugar_model.pkl'))
    SPECTRA = 'data_input/spectra_snia.pkl'

    HH = []
    dic = {}
    SN = dic_sed['sn_name']
    for i,SNN in enumerate(SN):
        print i 
        ASED=aligne_SED(SPECTRA,SNN,dic_sed['alpha'],dic_sed['m0'],dic_at_max['RV'])
        ASED.align_SED()
        ASED.align_spectra()

        GF=global_fit(ASED.Y,ASED.SED,dY=ASED.Y_err,CovY=None)
        GF.compute_h()
        print GF.h
        HH.append(GF.h)
        A=N.array(HH[i])
        h=A[0]
        cov_h=GF.cov_h
        dic.update({SNN:{'Av':h[1],
                         'grey':h[0],
                         'q1':h[2],
                         'q2':h[3],
                         'q3':h[4],
                         'cov_q':cov_h}})
        
    File=open('data_output/sugar_parameters.pkl','w')
    cPickle.dump(dic,File)
    File.close()
