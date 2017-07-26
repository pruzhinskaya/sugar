import pylab as P
import matplotlib.gridspec as gridspec
import numpy as N
import scipy as S 
import multilinearfit as Multi
from SUGAR_photometry_fitting import MC_light_curve
import cPickle
import copy
from Load_time_spectra import d_l
from ToolBox import Hubblefit as H

 

class load_SALT2_data:

    def __init__(self,META,sn_name):

        self.dic=cPickle.load(open(META))
        self.sn_name=sn_name


    def load_photometry_and_data(self,Filter='B'):

        self.Photometry=N.zeros(len(self.sn_name))
        self.Photometry_err=N.zeros(len(self.sn_name))
        self.data=N.zeros((len(self.sn_name),2))
        self.Cov_data=N.zeros((len(self.sn_name),2,2))
        self.z=N.zeros(len(self.sn_name))
        self.z_err=N.zeros(len(self.sn_name))
        self.DayMax=N.zeros(len(self.sn_name))

        for i,sn in enumerate(self.sn_name):
            self.z[i]=self.dic[sn]['host.zcmb']
            self.Photometry[i]=self.dic[sn]['salt2.RestFrameMag_0_'+Filter]-5.*N.log(d_l(self.z[i],SNLS=True))/N.log(10.)+5.
            self.Photometry_err[i]=N.sqrt(self.dic[sn]['salt2.RestFrameMag_0_'+Filter+'.err']**2+0.03**2)
            self.data[i,0]=self.dic[sn]['salt2.X1']
            self.data[i,1]=self.dic[sn]['salt2.Color']
            self.Cov_data[i,0,0]=self.dic[sn]['salt2.X1.err']**2
            self.Cov_data[i,1,1]=self.dic[sn]['salt2.Color.err']**2
            self.Cov_data[i,0,1]=self.dic[sn]['salt2.CovColorX1']
            self.Cov_data[i,1,0]=self.dic[sn]['salt2.CovColorX1']
            self.z_err[i]=self.dic[sn]['host.zhelio.err']
            self.DayMax[i]=self.dic[sn]['salt2.DayMax']
            

    def cut_salt2(self,snid_pkl=None,sn_list=None,low_z=True,Phot=True):

        self.Filtre=N.array([True]*len(self.sn_name))

        if snid_pkl is not None: 
            dic_snid=cPickle.load(open(snid_pkl))
            self.sub_type=[]
            self.Rate=[]
            self.key_snid=['Ic-broad','Ia-pec','Ia-91T',
                           'Ia-91bg','IIn','Ic-norm',
                           'Gal','Ib-norm','IIP',
                           'II-pec','Ib-pec','IIb',
                           'Ia-csm','IIL','AGN','M-star']


        for i,sn in enumerate(self.sn_name):

            # cut low redshift
            if low_z:
                if self.z[i]<0.01:
                    self.Filtre[i]=False
                    print 'prout'                                                                                                                                                                      


            # cut auxiliary and bad
            if Phot:
                if self.dic[sn]['idr.subset']  in ['auxiliary','bad']:
                    self.Filtre[i]=False
                    print 'prout'


            if sn_list is not None and sn in sn_list:
                self.Filtre[i]=False
                print 'Good Bye ma poule'

            # cut day
            #if self.DayMax[i]<55166:
            #    self.Filtre[i]=False
            #    print 'Good Bye Francis'




            if snid_pkl is not None: 
                sub_type=[]
                BAD=False
                Rate=0
                for j,key in enumerate(dic_snid[sn]['spectra'].keys()):
                    TYPE=dic_snid[sn]['spectra'][key]['SNID']['snid.subtype']
                    sub_type.append(TYPE)
                    if TYPE in self.key_snid:
                        Rate+=1.
                        BAD=True
                if BAD and self.Filtre[i]:
                    if Rate/len(sub_type)>0.15:
                        sub_type.append(sn)
                        self.sub_type.append(sub_type)
                        self.Rate.append(Rate/len(sub_type))
                        self.Filtre[i]=False

        self.Photometry=self.Photometry[self.Filtre]
        self.Photometry_err=self.Photometry_err[self.Filtre]
        self.data=self.data[self.Filtre]
        self.Cov_data=self.Cov_data[self.Filtre]
        self.z=self.z[self.Filtre]
        self.z_err=self.z_err[self.Filtre]
        self.sn_name=N.array(self.sn_name)[self.Filtre]
        self.DayMax=self.DayMax[self.Filtre]
        
        
        


class load_SUGAR_data:

    def __init__(self,SUGAR_parameters,sn_name,SUGAR_asci,HYPERPARAMETER,Filter=0):
        
        self.dic=cPickle.load(open(SUGAR_parameters))
        self.sn_name=sn_name
        self.Filter=Filter
        self.SUGAR_asci=SUGAR_asci
        self.HYPERPARAMETER=HYPERPARAMETER


    def compute_photometry(self):
        
        self.Photometry=N.zeros(len(self.sn_name))
        
        self.SUGAR_LC=MC_light_curve(self.SUGAR_asci,self.HYPERPARAMETER,N.linspace(-3,3,7))
        
        for i,sn in enumerate(self.sn_name):
            print sn 
            Q=N.array([self.dic[sn]['x1'],self.dic[sn]['x2'],self.dic[sn]['x3']])
            #self.SUGAR_LC.build_fixed_light_curves(Q,self.dic[sn]['Av'],self.dic[sn]['Grey'])
            self.SUGAR_LC.build_fixed_light_curves(N.zeros(3),self.dic[sn]['Av'],0)
            self.Photometry[i]=self.SUGAR_LC.B_max[0]

    def _SUGAR_variate_q1(self,q1):
        Q=N.array([q1,self.Q[1],self.Q[2]])
        self.SUGAR_LC.build_fixed_light_curves(Q,self.Q[3],self.Q[4])
        #return self.SUGAR_LC.BVR[self.Filter][3]
        return self.SUGAR_LC.B_max[0]

    def _SUGAR_variate_q2(self,q2):
        Q=N.array([self.Q[0],q2,self.Q[2]])
        self.SUGAR_LC.build_fixed_light_curves(Q,self.Q[3],self.Q[4])
        #return self.SUGAR_LC.BVR[self.Filter][3]
        return self.SUGAR_LC.B_max[0]

    def _SUGAR_variate_q3(self,q3):
        Q=N.array([self.Q[0],self.Q[1],q3])
        self.SUGAR_LC.build_fixed_light_curves(Q,self.Q[3],self.Q[4])
        #return self.SUGAR_LC.BVR[self.Filter][3]
        return self.SUGAR_LC.B_max[0]

    def _SUGAR_variate_Av(self,Av):
        Q=N.array([self.Q[0],self.Q[1],self.Q[2]])
        self.SUGAR_LC.build_fixed_light_curves(Q,Av,self.Q[4])
        #return self.SUGAR_LC.BVR[self.Filter][3]
        return self.SUGAR_LC.B_max[0]

    def _SUGAR_variate_Grey(self,Grey):
        Q=N.array([self.Q[0],self.Q[1],self.Q[2]])
        self.SUGAR_LC.build_fixed_light_curves(Q,self.Q[3],Grey)
        #return self.SUGAR_LC.BVR[self.Filter][3]
        return self.SUGAR_LC.B_max[0]


    def compute_derivative(self):

        self.A=N.zeros(5)

        self.A[2]=S.misc.derivative(self._SUGAR_variate_q1, self.Q[0], dx=0.1)
        self.A[3]=S.misc.derivative(self._SUGAR_variate_q2, self.Q[1], dx=0.1)
        self.A[4]=S.misc.derivative(self._SUGAR_variate_q3, self.Q[2], dx=0.1)
        self.A[1]=S.misc.derivative(self._SUGAR_variate_Av, self.Q[3], dx=0.1)
        self.A[0]=S.misc.derivative(self._SUGAR_variate_Grey, self.Q[4], dx=0.01)


    def compute_photometry_error(self,DOIT=False):
        print ''
        self.Photometry_err=N.zeros(len(self.sn_name))
        if DOIT:
            for i,sn in enumerate(self.sn_name):
                print sn
                self.Q=N.array([self.dic[sn]['x1'],self.dic[sn]['x2'],self.dic[sn]['x3'],self.dic[sn]['Av'],self.dic[sn]['Grey']])
                self.compute_derivative()
                self.Photometry_err[i]=N.dot(self.A,N.dot(self.dic[sn]['cov_h'],self.A.reshape(len(self.A),1)))
    
            
    def compute_pkl_photometry(self,pkl_name):

        FILTRE=['B','V','R']

        dic={}
        for i,sn in enumerate(self.sn_name):
            dic.update({sn:{'%s'%(FILTRE[self.Filter]):self.Photometry[i],
                            '%s'%(FILTRE[self.Filter])+'_err':self.Photometry_err[i],
                            'Av':self.dic[sn]['Av']}})

        File=open(pkl_name,'w')
        cPickle.dump(dic,File)
        File.close()

    def load_data(self,UBVRI,PHOT=None):
        dic=cPickle.load(open(UBVRI))

        if PHOT is not None:
            dic_phot=cPickle.load(open(PHOT))
            self.Photometry_err=N.zeros(len(self.sn_name))
            self.Photometry=N.zeros(len(self.sn_name))

        self.data=N.zeros((len(self.sn_name),4))
        self.Cov_data=N.zeros((len(self.sn_name),4,4))
        self.z=N.zeros(len(self.sn_name))
        self.z_err=N.zeros(len(self.sn_name))

        Filtre=['B','V','R']

        for i,sn in enumerate(self.sn_name):
            self.data[i,0]=self.dic[sn]['Av']
            self.data[i,1]=self.dic[sn]['x1']
            self.data[i,2]=self.dic[sn]['x2']
            self.data[i,3]=self.dic[sn]['x3']
            self.Cov_data[i]=self.dic[sn]['cov_h'][1:,1:]
            self.z[i]=dic[sn]['host.zcmb']
            self.z_err[i]=dic[sn]['host.zhelio.err']

            if PHOT is not None:
                self.Photometry[i]=dic_phot[sn]['%s'%(Filtre[self.Filter])]
                self.Photometry_err[i]=dic_phot[sn]['%s_err'%(Filtre[self.Filter])]



class Hubble_diagram:

    def __init__(self,Phot,Phot_err,Data,Data_cov,z,z_err):

        self.Y=Phot
        dmz=5/N.log(10) * N.sqrt(z_err**2 + 0.001**2) / z
        self.Y_err=N.sqrt(Phot_err**2+dmz**2+0.03**2)
        self.data=Data
        self.data_cov=Data_cov
        self.z=z
        self.z_err=z_err

    def Make_hubble_diagram(self):

        self.HF=Multi.Multilinearfit(self.data,self.Y,yerr=self.Y_err,xerr=None,covx=self.data_cov)
        
        self.HF.Multilinearfit(adddisp=True)
        self.HF.comp_stat()
        mb=self.Y+5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)-5.
        self.residuals=mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5.

    def plot_mass_step(self,sn_name,HOST_pkl):
        
        HOST=cPickle.load(open(HOST_pkl))
        mb=self.HF.y_corrected+5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)-5.
        data= mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5.
        error= N.sqrt(self.HF.y_error_corrected**2+self.Y_err**2)

        Masse=N.zeros(len(data))
        Masse_minus=N.zeros(len(data))
        Masse_plus=N.zeros(len(data))
        Filtre_M=N.array([True]*len(data))

        for i in range(len(sn_name)):
            sn=sn_name[i]
            if sn in HOST.keys() :
                Masse[i]=HOST[sn]['mchost.mass']
                Masse_minus[i]=HOST[sn]['mchost.mass_m.err']
                Masse_plus[i]=HOST[sn]['mchost.mass_p.err']

                if HOST[sn]['mchost.mass']==0:
                    Filtre_M[i]=False
            else :
                Filtre_M[i]=False


        MASS_SUP=(Masse[Filtre_M]>10)
        Xmass_SUP=N.linspace(10,14,10)
        Xmass_low=N.linspace(6,10,10)
        Masse_err=[Masse_minus[Filtre_M],Masse_plus[Filtre_M]]
        
        M_SI_SUP=N.average(data[Filtre_M][MASS_SUP],weights=1./error[Filtre_M][MASS_SUP]**2)
        wRMS_sup=H.comp_rms(data[Filtre_M][MASS_SUP]-M_SI_SUP, 10, err=False, variance=error[Filtre_M][MASS_SUP]**2)
        M_SI_SUP=N.average(data[Filtre_M][MASS_SUP],weights=1./(error[Filtre_M][MASS_SUP]**2+wRMS_sup**2))
        M_SI_SUP_err=N.sqrt((1./N.sum(1./(error[Filtre_M][MASS_SUP]**2+wRMS_sup**2))))
        
        M_SI_low=N.average(data[Filtre_M][~MASS_SUP],weights=1./error[Filtre_M][~MASS_SUP]**2)
        wRMS_low=H.comp_rms(data[Filtre_M][~MASS_SUP]-M_SI_low, 10, err=False, variance=error[Filtre_M][~MASS_SUP]**2)
        M_SI_low=N.average(data[Filtre_M][~MASS_SUP],weights=1./(error[Filtre_M][~MASS_SUP]**2+wRMS_low**2))
        M_SI_low_err=N.sqrt((1./N.sum(1./(error[Filtre_M][~MASS_SUP]**2+wRMS_low**2))))

        print M_SI_SUP-M_SI_low, ' +/- ', N.sqrt(M_SI_SUP_err**2+M_SI_low_err**2)
        pull=abs(M_SI_SUP-M_SI_low)/N.sqrt(M_SI_SUP_err**2+M_SI_low_err**2)
        X_SI=N.linspace(min(data)-N.std(data),max(data)+N.std(data),100)
        P.figure()

        P.text(11,0.4,'$\sigma=%.3f$'%(pull),fontsize=14)
        P.scatter(Masse[Filtre_M],data[Filtre_M],c='b')
        P.errorbar(Masse[Filtre_M],data[Filtre_M],linestyle='', yerr=error[Filtre_M],xerr=Masse_err,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.plot(10*N.ones(2),[-0.6,0.6],'k-.',linewidth=2)
        P.plot(Xmass_low,N.ones(len(Xmass_low))*M_SI_low,'b')
        P.fill_between(Xmass_low,N.ones(len(Xmass_low))*M_SI_low-M_SI_low_err,N.ones(len(Xmass_low))*M_SI_low+M_SI_low_err,color='b',alpha=0.5)
        P.plot(Xmass_SUP,N.ones(len(Xmass_SUP))*M_SI_SUP,'b')
        P.fill_between(Xmass_SUP,N.ones(len(Xmass_SUP))*M_SI_SUP-M_SI_SUP_err,N.ones(len(Xmass_SUP))*M_SI_SUP+M_SI_SUP_err,color='b',alpha=0.5)

        P.ylim(min(data)-N.std(data),max(data)+2*N.std(data))
        P.xlabel('$\log(M/M_{\odot})$',fontsize=16)
        P.xlim(7,12)
        P.ylabel(r'$\mu-\mu_{\Lambda CDM}$',fontsize=15)
        P.ylim(-0.6,0.6)

    def plot_result_before(self,SALT=True):
        
        P.figure(figsize=(9,9))
        mb=self.Y+5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)-5.

        gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
        #P.subplots_adjust(hspace=0.001)
        P.subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.95,hspace=0.001)

        P.subplot(gs[0])

        wwRMS,wwRMS_ERR=H.comp_rms(mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5., 10, err=True, variance=self.Y_err**2)

        P.scatter(self.z,mb-self.HF.M0,label='wRMS = %.3f$ \pm$ %.3f'%((wwRMS,wwRMS_ERR)))
        P.errorbar(self.z,mb-self.HF.M0 , xerr=None ,
                   yerr=N.sqrt(self.HF.y_error_corrected**2+self.Y_err**2),linestyle='',alpha=0.5,marker='.',zorder=0)
        P.plot(N.linspace(0.001,0.11,10),5.*N.log(d_l(N.linspace(0.001,0.11,10),SNLS=True))/N.log(10.)-5.)
        P.xlim(0.01,0.11)
        P.ylim(32.2,39)
        if SALT:
            P.ylabel(r'$\mu=m_B^*-M_0$',fontsize=20) 
            P.title('Hubble diagram (%i supernovae)'%(len(self.z)))
        else:
            P.ylabel(r'$\mu=m_B^*-M_0$',fontsize=20)
            P.title('SUGAR Hubble diagram (%i supernovae)'%(len(self.z)))
            
        P.semilogx()
        P.xticks([2500.,9500.],['toto','pouet'])
        P.xlim(0.01,0.11)
        P.legend(loc=4)

        P.subplot(gs[1])

        P.scatter(self.z,mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5.)
        P.errorbar(self.z,mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5., xerr=None ,
                   yerr=N.sqrt(self.HF.y_error_corrected**2+self.Y_err**2),linestyle='',alpha=0.5,marker='.',zorder=0)
        P.plot(N.linspace(0.001,0.11,10),N.zeros(10))
        P.xlim(0.01,0.11)
        P.ylim(-1,1)
        P.ylabel(r'$\mu-\mu_{\Lambda CDM}$',fontsize=20)
        P.semilogx()
        P.xlabel('z',fontsize=20)


    def plot_result_after(self,SALT=True):
        
        P.figure(figsize=(9,9))
        mb=self.HF.y_corrected+5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)-5.

        gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
        P.subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.95,hspace=0.001)
        P.subplot(gs[0])

        #P.scatter(self.z,mb-self.HF.M0,c='b',label='wRMS = %.3f $\pm$ %.3f,   $\sigma_{int}$=%.3f'%((self.HF.WRMS,self.HF.WRMS_err,self.HF.disp)))
        P.scatter(self.z,mb-self.HF.M0,c='b',label='wRMS = %.3f $\pm$ %.3f'%((self.HF.WRMS,self.HF.WRMS_err)))
        P.errorbar(self.z,mb-self.HF.M0 , xerr=None ,
                   yerr=N.sqrt(self.HF.y_error_corrected**2+self.Y_err**2),linestyle='',alpha=0.5,marker='.',zorder=0)
        P.plot(N.linspace(0.001,0.11,10),5.*N.log(d_l(N.linspace(0.001,0.11,10),SNLS=True))/N.log(10.)-5.)
        P.xlim(0.01,0.11)
        P.ylim(32.2,39)
        if SALT:
            P.ylabel(r'$\mu=m_B^*-M_0+\alpha x_1 -\beta c$',fontsize=20) 
            P.title('SALT2 Hubble diagram (%i supernovae)'%(len(self.z)))
        else:
            P.ylabel(r'$\mu=m_B^*-M_0-\sum_i^{n=3}a_i q_i -b A_V$',fontsize=20)
            P.title('SUGAR Hubble diagram (%i supernovae)'%(len(self.z)))
            
        P.semilogx()
        P.xticks([2500.,9500.],['toto','pouet'])
        P.xlim(0.01,0.11)
        P.legend(loc=4)

        P.subplot(gs[1])
        self.residuals=mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5.
        P.scatter(self.z,mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5.,c='b')
        P.errorbar(self.z,mb-self.HF.M0-5.*N.log(d_l(self.z,SNLS=True))/N.log(10.)+5., xerr=None ,
                   yerr=N.sqrt(self.HF.y_error_corrected**2+self.Y_err**2),linestyle='',alpha=0.5,marker='.',zorder=0)
        P.plot(N.linspace(0.001,0.11,10),N.zeros(10))

        P.xlim(0.01,0.11)
        P.ylim(-1,1)
        P.ylabel(r'$\mu-\mu_{\Lambda CDM}$',fontsize=20)
        P.semilogx()
        P.xlabel('z',fontsize=20)


    def plot_slopes(self):

        residu=N.zeros((len(self.data[0]),len(self.data[:,0])))

        for i in range(len(self.data[0])):
            residu[i]=copy.deepcopy(self.Y-self.HF.M0)
            for j in range(len(self.data[0])):
                if i!=j:
                    residu[i]-=self.data[:,j]*self.HF.alpha[j]
            
        P.figure()
        XX=N.linspace(N.min(self.data[:,0])-0.1,N.max(self.data[:,0])+0.1,20)
        P.scatter(self.data[:,0],residu[0])
        P.plot(XX,XX*self.HF.alpha[0],linewidth=3,label='b=%.2f, $R_V$=%.2f'%((self.HF.alpha[0],1./(self.HF.alpha[0]-1.))))
        P.plot(XX,XX*1.38314,'r-.',linewidth=3,label='b=%.2f, $R_V$=%.2f'%((1.38314,1./(1.38314-1.))))
        P.ylabel('$\mu=m_B^*-M_0-\mu_{\Lambda CDM}-\sum_i^{n=3}a_i q_i$',fontsize=16)
        P.xlabel('$A_V$',fontsize=16)
        P.legend()
        
    

if __name__=='__main__':


    SP=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl'))
    SN_name=N.loadtxt('/sps/snovae/user/leget/ace_sne.list',dtype='str')

    Filtre=N.array([False]*len(SN_name))

    for i in range(len(SN_name)):
        if SN_name[i] in SP.keys():
            Filtre[i]=True

    SN_name=SN_name[Filtre]

    #LSD=load_SUGAR_data('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl',SP.keys(),'/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_v1.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP_binning_speed_without_MFR_issue/hyperparameters.dat',Filter=0)
    LSD=load_SUGAR_data('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl',SN_name,'/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_v1.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP_binning_speed_without_MFR_issue/hyperparameters.dat',Filter=0)
    #LSD.compute_photometry()
    #LSD.compute_photometry_error()
    #LSD.compute_pkl_photometry('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_validation_photometry_B_filter.pkl')
    LSD.load_data('/sps/snovae/user/leget/CABALLO/UBVRI.pkl',PHOT='/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_validation_photometry_B_filter.pkl')

    HD=Hubble_diagram(LSD.Photometry,LSD.Photometry_err,LSD.data,LSD.Cov_data,LSD.z,LSD.z_err) 
    HD.Make_hubble_diagram()
    HD.plot_result_after(SALT=False)
    P.savefig('../Post_doc_plot/decembre_2016/HD_SUGAR_CABALLO_inter_ACE.pdf')
    #HD.plot_mass_step(LSD.sn_name,'/sps/snovae/user/leget/BEDELL/Host.pkl')
    #HD.plot_slopes()
    #P.savefig('../Quality_plot/HD_SALT2_ALLEG.pdf')
    #HD.plot_result_before(SALT=True)
    #HD.plot_slopes()

    #LSDC=load_SALT2_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',SP.keys())                                                                      
    LSDC=load_SALT2_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',SN_name)                                                                      
    LSDC.load_photometry_and_data(Filter='B')
    HDC=Hubble_diagram(LSDC.Photometry,LSDC.Photometry_err,LSDC.data,LSDC.Cov_data,LSDC.z,LSDC.z_err)
    HDC.Make_hubble_diagram()
    #HDC.plot_result_before(SALT=True)
    HDC.plot_result_after(SALT=True)
    P.savefig('../Post_doc_plot/decembre_2016/HD_SALT2_CABALLO_inter_ACE.pdf')
    #HDC.plot_mass_step(LSDC.sn_name,'/sps/snovae/user/leget/BEDELL/Host.pkl')




    ### POUR LE CHAPITRE 3 A PARTIR DE LA 


    ##list_sn_bizarre=['SN2005bl','SN2005dm','SNF20070803-005','SNF20080522-000','SNF20080723-012',
    ##                 'SN2005M','SNF20060512-001','SNF20070528-003','SNF20070912-000','SNF20080909-030',
    ##                 'SNF20080919-001','SN2005cc','SN2006bk','SN2009hs','SNF20070825-001','SNF20080905-005',
    ##                 'SNF20080927-001','LSQ12gdj','PTF11mty','PTF11mkx','LSQ12cyz','SN2007cq',
    ##                 'SNF20071108-021','SNF20061022-005','PTF11bju']#,'LSQ12fhe']

    
    ##SP=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training.pkl'))
    ##SPC=cPickle.load(open('/sps/snovae/user/leget/CABALLO/META_JLA.pkl'))
    ##SPC=cPickle.load(open('/sps/snovae/user/leget/META.pkl'))
    ##SPA=cPickle.load(open('/sps/snovae/user/leget/ACE/META.pkl'))
    ##dic_CAB = cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl'))
    #SP=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_ACE_selected.pkl'))
    #ACE_sn=N.loadtxt('/sps/snovae/user/leget/ace_sne.list',dtype='str')

    ##sn_name=SPA.keys()
    ##for i in range(len(sn_name)):
    ##    if sn_name[i]=='DATASET':
    ##        del sn_name[i]
    ##        break


    ##LSDC=load_SALT2_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',dic_CAB.keys())
    ##LSDC.load_photometry_and_data(Filter='B')
    #LSD1.cut_salt2(snid_pkl=None,sn_list=list_sn_bizarre,low_z=True,Phot=False)
    #LSDC.cut_salt2(snid_pkl=None,sn_list=list_sn_bizarre,low_z=True,Phot=True)
    ##LSDC.cut_salt2(snid_pkl=None,sn_list=None,low_z=False,Phot=False)

    #z_correct=[0.054726414660625403,0.071158914539179829]

    #for i in range(len(LSDC.sn_name)):
    #    if LSDC.sn_name[i]=='PTF10yux':
    #        LSDC.redshift[i]=

    ##NP=N.zeros(len(LSDC.sn_name))
    ##Photo=N.zeros(len(LSDC.sn_name))
    ##NPhoto=N.zeros(len(LSDC.sn_name))

    ##dic=cPickle.load(open('/sps/snovae/user/leget/CABALLO/META.pkl'))

    ##for i in range(len(LSDC.sn_name)):
    ##   print i
    ##    DATE=1000000
    ##    for j,key in enumerate(dic[LSDC.sn_name[i]]['spectra'].keys()):
    ##        NP[i]+=1
    ##        if dic[LSDC.sn_name[i]]['spectra'][key]['obs.photo']==1:
    ##            Photo[i]+=1
    ##        else:
    ##            NPhoto[i]+=1


    ##LSDA=load_SALT2_data('/sps/snovae/user/leget/ACE/META.pkl',sn_name)
    ##LSDA.load_photometry_and_data(Filter='B')
    ##LSDA.cut_salt2(snid_pkl=None,sn_list=list_sn_bizarre,low_z=True,Phot=True)
   
    ##SN_in_ACE=[]
    ##SN_not_in_ACE=[]

    ##FILTRE=N.array([True]*len(LSDC.sn_name))

    ##for i,sn in enumerate(LSDC.sn_name):
    ##    if sn not in LSDA.sn_name:
    ##        SN_not_in_ACE.append(sn)
    ##        FILTRE[i]=False
    ##    else:
    ##        SN_in_ACE.append(sn)
            

    ##LSD1=load_SALT2_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',SN_in_ACE)
    ##LSD1.load_photometry_and_data(Filter='B')

    ##LSD2=load_SALT2_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',SN_not_in_ACE)
    ##LSD2.load_photometry_and_data(Filter='B')

    # To do : - Number of points 
    # Number of photometric night
    # Number of Non photometric night
    # date of first data

    ##NP1=N.zeros(len(SN_in_ACE))
    ##NP2=N.zeros(len(SN_not_in_ACE))

    ##Photo1=N.zeros(len(SN_in_ACE))
    ##Photo2=N.zeros(len(SN_not_in_ACE))

    ##NPhoto1=N.zeros(len(SN_in_ACE))
    ##NPhoto2=N.zeros(len(SN_not_in_ACE))

    ##Date_first1=N.zeros(len(SN_in_ACE))
    ##Date_first2=N.zeros(len(SN_not_in_ACE))




    ##for i in range(len(LSD1.sn_name)):
    ##    print i
    ##    DATE=1000000
    ##    for j,key in enumerate(dic[LSD1.sn_name[i]]['spectra'].keys()):
    ##        NP1[i]+=1
    ##        if dic[LSD1.sn_name[i]]['spectra'][key]['obs.photo']==1:
    ##            Photo1[i]+=1
    ##        else:
    ##            NPhoto1[i]+=1

    ##        if dic[LSD1.sn_name[i]]['spectra'][key]['salt2.phase']<DATE:
    ##            DATE=dic[LSD1.sn_name[i]]['spectra'][key]['salt2.phase']
    ##            Date_first1[i]=DATE

    ##for i in range(len(LSD2.sn_name)):
    ##    print i
    ##    DATE=1000000
    ##    for j,key in enumerate(dic[LSD2.sn_name[i]]['spectra'].keys()):
    ##        NP2[i]+=1
    ##        if dic[LSD2.sn_name[i]]['spectra'][key]['obs.photo']==1:
    ##            Photo2[i]+=1
    ##        else:
    ##           NPhoto2[i]+=1

    ##        if dic[LSD2.sn_name[i]]['spectra'][key]['salt2.phase']<DATE:
    ##           DATE=dic[LSD2.sn_name[i]]['spectra'][key]['salt2.phase']
    ##            Date_first2[i]=DATE

    #P.figure()
    #toto, tata, patches1=P.hist(NP1,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(NP2,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.xlabel('Number of points per supernovae')
    #P.ylabel('Number of supernovae')
    #P.ylim(0,35)
    #P.legend()


    #P.figure()
    #toto, tata, patches1=P.hist(Date_first1,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(Date_first2,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.ylabel('Number of supernovae')
    #P.xlabel('SALT2 phase of first spectrum (days)')
    #P.ylim(0,30)
    #P.legend()

    #P.figure()
    #toto, tata, patches1=P.hist(Photo1,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(Photo2,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.ylabel('Number of supernovae')
    #P.xlabel('Number of photometric night per supernovae')
    #P.ylim(0,40)
    #P.legend()

    #P.figure()
    #toto, tata, patches1=P.hist(NPhoto1,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(NPhoto2,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.ylabel('Number of supernovae')
    #P.xlabel('Number of non photometric night per supernovae')
    #P.ylim(0,45)
    #P.legend()


    #P.figure()
    #toto, tata, patches1=P.hist(NPhoto1/NP1,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(NPhoto2/NP2,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.ylabel('Number of supernovae')
    #P.xlabel('Ratio of non photometric night')
    #P.ylim(0,45)
    #P.legend()


    #P.figure()
    #toto, tata, patches1=P.hist(LSD1.data[:,0],histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(LSD2.data[:,0],histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.xlabel('$x_1$')
    #P.ylabel('Number of supernovae')
    #P.ylim(0,30)
    #P.legend()

    #P.figure()
    #toto, tata, patches1=P.hist(LSD1.data[:,1],histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(LSD2.data[:,1],histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.xlabel('$c$')
    #P.ylabel('Number of supernovae')
    #P.legend()

    #P.figure()
    #toto, tata, patches1=P.hist(LSD1.z,histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(LSD2.z,histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.xlabel('$redshift$')
    #P.ylabel('Number of supernovae')
    #P.ylim(0,35)
    #P.legend()


    #FILTRE=['B','V','R']
    #for i in range(3):
    #i=0
    #LSD=load_SUGAR_data('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training.pkl',SP.keys(),'/sps/snovae/user/leget/CABALLO/SUGAR_model.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP/hyperparameters.dat',Filter=i)
    #LSD.compute_photometry()
    #LSD.compute_photometry_error()
    #LSD.compute_pkl_photometry('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training_photometry_%s.pkl'%(FILTRE[i]))
    #LSD.load_data('/sps/snovae/user/leget/CABALLO/UBVRI.pkl',PHOT='/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training_photometry_%s.pkl'%(FILTRE[i]))    

    #LSD=load_SUGAR_data('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training_validation.pkl',LSD1.sn_name,'/sps/snovae/user/leget/CABALLO/SUGAR_model.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP/hyperparameters.dat',Filter=i)
    #    LSD.compute_photometry()
    #    LSD.compute_photometry_error()
    #    LSD.compute_pkl_photometry('/sps/snovae/user/leget/CABALLO/SUGAR_parameters_ACE_sample_photometry_%s.pkl'%(FILTRE[i]))
    #LSD.load_data('/sps/snovae/user/leget/CABALLO/META_JLA.pkl',PHOT='/sps/snovae/user/leget/CABALLO/SUGAR_parameters_training_validation_photometry_%s.pkl'%(FILTRE[i]))    

    ##HD=Hubble_diagram(LSDC.Photometry,LSDC.Photometry_err,LSDC.data,LSDC.Cov_data,LSDC.z,LSDC.z_err) 
    ##HD1=Hubble_diagram(LSD1.Photometry,LSD1.Photometry_err,LSD1.data,LSD1.Cov_data,LSD1.z,LSD1.z_err)
    ##HD2=Hubble_diagram(LSD2.Photometry,LSD2.Photometry_err,LSD2.data,LSD2.Cov_data,LSD2.z,LSD2.z_err)
    ##HD.Make_hubble_diagram()
    ##HD1.Make_hubble_diagram()
    ##HD2.Make_hubble_diagram()
    #HD1.plot_result_after(SALT=True)
    #HD2.plot_result_after(SALT=True)
    #HD.plot_result_after(SALT=True)
    ##HD.plot_mass_step(LSDC.sn_name,'/sps/snovae/user/leget/BEDELL/Host.pkl')
    #P.savefig('../Quality_plot/HD_SALT2_ALLEG.pdf')
    #HD.plot_result_before(SALT=True)
    #HD.plot_slopes()

    #P.figure()
    #toto, tata, patches1=P.hist(HD1.HF.residu/N.sqrt(HD1.HF.y_error_corrected**2+HD1.Y_err**2),histtype='stepfilled',label='In CABALLO and in ACE')
    #P.setp(patches1, 'facecolor', 'b', 'alpha', 0.7)
    #toto, tata, patches2=P.hist(HD2.HF.residu/N.sqrt(HD2.HF.y_error_corrected**2+HD2.Y_err**2),histtype='stepfilled',label='In CABALLO and not in ACE')
    #P.setp(patches2, 'facecolor', 'r', 'alpha', 0.7)
    #P.xlabel('pull')
    #P.ylabel('Number of supernovae')
    #P.ylim(0,45)
    #P.legend()
    #P.savefig('../Quality_plot/distribution_pulse.pdf')


    ##P.figure()

    #FILTRE=((NPhoto/NP)<0.8)
    #FILTRE=(LSDC.DayMax<55250)

    ##P.errorbar(LSDC.DayMax[FILTRE],HD.HF.residu[FILTRE], xerr=None ,
    ##           yerr=N.sqrt(HD.HF.y_error_corrected[FILTRE]**2+HD.Y_err[FILTRE]**2),linestyle='',ecolor='b',alpha=0.5,marker='.',zorder=0)
    ##P.errorbar(LSDC.DayMax[~FILTRE],HD.HF.residu[~FILTRE], xerr=None ,
    ##           yerr=N.sqrt(HD.HF.y_error_corrected[~FILTRE]**2+HD.Y_err[~FILTRE]**2),linestyle='',ecolor='r',alpha=0.5,marker='.',zorder=0)


    ##P.scatter(LSDC.DayMax[FILTRE],HD.HF.residu[FILTRE],c='b',label='In CABALLO and in ACE') 
    ##P.scatter(LSDC.DayMax[~FILTRE],HD.HF.residu[~FILTRE],c='r',label='In CABALLO and not in ACE')
    
    ##from ToolBox import Hubblefit
    ##filtre=(LSDC.DayMax<55250)
    ##print len(filtre)
    ##print N.sum(filtre)
    #filtre=FILTRE
    ##wrms_young,wrms_young_err=Hubblefit.comp_rms(HD.HF.residu[filtre], 10, err=True, variance=HD.HF.y_error_corrected[filtre]**2+HD.Y_err[filtre]**2)
    ##wrms_old,wrms_old_err=Hubblefit.comp_rms(HD.HF.residu[~filtre], 10, err=True, variance=HD.HF.y_error_corrected[~filtre]**2+HD.Y_err[~filtre]**2)

    ##JJJ=N.linspace(min(LSDC.DayMax)-10,max(LSDC.DayMax)+10,100)
    ##P.plot(JJJ,N.zeros(len(JJJ)),'k')
    ##P.plot([55250,55250],[-1,1],'r',linewidth=3)
    ##P.xlabel('Modified Julian Date')
    ##P.ylim(-1,1)
    ##P.ylabel('residual')
    ##P.xlim(min(LSDC.DayMax)-10,max(LSDC.DayMax)+10)
    ##P.text(53330., 0.9, 'wRMS(days<55250)=$%.3f\pm%.3f$'%((wrms_young,wrms_young_err)),fontsize=10) 
    ##P.text(55270., 0.9, 'wRMS(days>55250)=$%.3f\pm%.3f$'%((wrms_old,wrms_old_err)),fontsize=10) 
    ##P.legend(loc=3)
    #P.savefig('../Quality_plot/toto.pdf')
    #scat=P.scatter(LSDC.DayMax,HD.HF.residu,s=1)
    

    #from ToolBox import MPL
    #browser=MPL.PointBrowser(LSDC.DayMax, HD.HF.residu,N.array(LSDC.sn_name),scat)
    #P.show()
