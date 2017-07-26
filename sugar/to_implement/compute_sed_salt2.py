import numpy as N
import pylab as P
import cPickle
import scipy.interpolate as inter
import Gaussian_process as GP
from load_spectrophotometry import d_l
from ToolBox.Astro import Templates as T 
from ToolBox.Astro import Coords 


class generate_SED_SALT2:

    def __init__(self,META):
        
        self.DICO=cPickle.load(open(META))
        self.sn_name=self.DICO.keys()

        self.DIC_SALT2={}


    def phase_PF(self):
        Time=N.linspace(-13,43,57)
        self.SPECTRA={}
        for i,sn in enumerate(self.sn_name):
            YY=[]
            print sn
            dic_time={}
            x0=self.DICO[sn]['salt2.X0']
            x1=self.DICO[sn]['salt2.X1']
            Color=self.DICO[sn]['salt2.Color']
            redshift=self.DICO[sn]['host.zcmb']
            SALT=T.Templates(template='Salt2',X0=x0, X1=x1, c=Color)   
            self.SALT=SALT
            for j in range(len(Time)):
                x,y=SALT.spec_at_given_phase(Time[j])   
                Y=Coords.flbda2ABmag(x,y)-5.*N.log10(d_l(redshift,SNLS=True))+5.
                dic_time.update({'%i'%(Time[j]):{'x':x,
                                                 'y_flux':y,
                                                 'Y_cosmo':Y,
                                                 'X0':x0,
                                                 'X1':x1,
                                                 'C':Color,
                                                 'z':redshift}})
                YY.append(Y)
            YY=N.array(YY)
            self.SPECTRA.update({sn:{'Y':YY,
                                     'X':x}})

            self.DIC_SALT2.update({sn:dic_time})

        self.big_dico={'Time_series':self.SPECTRA,
                       'All':self.DIC_SALT2}

    def at_same_phase_spectra(self,ALL_data_pkl):

        dic_all=cPickle.load(open(ALL_data_pkl))
        sn_name=dic_all.keys()
        Time=N.linspace(-13,43,57)
        dic_time={}
        
        for i,sn in enumerate(sn_name):
            print i,sn 
            TS=self.big_dico['Time_series'][sn]['Y']
            X=self.big_dico['Time_series'][sn]['X']
            new_grid=[]
            DAYS=[-999]
            for t in range(len(dic_all[sn].keys())):
                DAYS.append(dic_all[sn]['%i'%(t)]['phase_salt2'])
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    new_grid.append(dic_all[sn]['%i'%(t)]['phase_salt2'])

            YY=N.zeros((len(new_grid),len(X)))
            
            for W in range(len(X)):
                SPLINEW=inter.InterpolatedUnivariateSpline(X[N.isfinite(TS[:,W])],TS[:,W][N.isfinite(TS[:,W])])
                TS[:,W][~N.isfinite(TS[:,W])]=SPLINEW(X[~N.isfinite(TS[:,W])])
                SPLINE=inter.InterpolatedUnivariateSpline(N.linspace(-13,43,57),TS[:,W])
                YY[:,W]=SPLINE(new_grid)

        
            dic_time.update({sn:{'x':X,
                                 'Y_cosmo':YY,
                                 'Time':new_grid,
                                 'X0':self.big_dico['All'][sn]['0']['X0'],
                                 'X1':self.big_dico['All'][sn]['0']['X1'],
                                 'C':self.big_dico['All'][sn]['0']['C'],
                                 'z':self.big_dico['All'][sn]['0']['z']}})

        self.DIC_SALT2=dic_time

    def write_pkl(self,pkl_name):

        File=open(pkl_name,'w')
        cPickle.dump(self.DIC_SALT2,File)
        File.close()



if __name__=='__main__': 

    MET='/sps/snovae/user/leget/CABALLO/META_JLA.pkl'
    #MET='/sps/snovae/user/leget/CABALLO/META.pkl'
    SED=generate_SED_SALT2(MET)
    SED.sn_name=['PTF09fox']
    SED.phase_PF()
    SED.at_same_phase_spectra('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl')
    #SED.write_pkl('/sps/snovae/user/leget/CABALLO/SED_SALT2_JLA_CABALLO_with_cosmology_test_manu.pkl')
