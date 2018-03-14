"""salt2 hubble diagram."""

import sugar
import numpy as N
import os 
import copy
import cPickle
import iminuit as minuit
from scipy import optimize,integrate


def comp_rms(residuals, dof, err=True, variance=None):
    """
    Compute the RMS or WRMS of a given distribution.

    :param 1D-array residuals: the residuals of the fit.
    :param int dof: the number of degree of freedom of the fit.
    :param bool err: return the error on the RMS (WRMS) if set to True.
    :param 1D-aray variance: variance of each point. If given,
                             return the weighted RMS (WRMS).

    :return: rms or rms, rms_err
    """

    if variance is None:                # RMS
        rms = float(N.sqrt(N.sum(residuals**2)/dof))
        rms_err = float(rms / N.sqrt(2*dof))
    else:                               # Weighted RMS
        assert len(residuals) == len(variance)
        rms = float(N.sqrt(N.sum((residuals**2)/variance) / N.sum(1./variance)))
        #rms_err = float(N.sqrt(1./N.sum(1./variance)))
        rms_err = N.sqrt(2.*len(residuals)) / (2*N.sum(1./variance)*rms)

    if err:
        return rms, rms_err
    else:
        return rms


def R200_to_M200(r200):
    
    G = 6.67*10**(-11)

    MS = 1.989*10**30
    h = 70000./(3.085677*10**22)
    
    m200 = ((r200*3.086*10**22)**3)*100*((h*h)/G)
    m200 /= MS

    return m200
    
    
def distance_metric(z,Omega_M=0.3):

    return 1./N.sqrt(Omega_M*(1+z)**3+(1.-Omega_M))

    
def luminosity_distance(zh,zcmb,H0=70.):

    c=299792.458
    if type(zcmb)==N.ndarray:
        integr=N.zeros_like(zcmb)
        for i in range(len(zcmb)):
            integr[i]=integrate.quad(distance_metric,0,zcmb[i])[0]
    else:
        integr=integrate.quad(distance_metric,0,zcmb)[0]

    return (1+zh)*(c/H0)*integr*10**6



def distance_modulus(zhel,zcmb,cst=0):
    
    return 5.*N.log(luminosity_distance(zhel,zcmb))/N.log(10.)-5.+cst

                    

class Hubble_diagram:

    def __init__(self,Phot,Data,Data_cov,zhelio,zcmb,z_err):

        self.Y=copy.deepcopy(Phot)
        self.dmz = N.zeros(len(z_err))
        self.dmz = 5/N.log(10) * N.sqrt(z_err**2 + 0.001**2) / zcmb

        self.data = Data
        self.data_cov = Data_cov
        self.zhelio = copy.deepcopy(zhelio)
        self.zcmb = copy.deepcopy(zcmb)
        self.z_err = z_err

        self.disp_intrinseque=0.
        self.disp=0.
        self.dof=len(self.Y)-3.

        
    def comp_chi2(self,Alpha,Beta,MB):

                    
        chi2=N.zeros(len(self.Y))
        self.residu=N.zeros(len(self.Y))
        self.VAR=N.zeros(len(self.Y))
        A=N.array([1.,Alpha,Beta])
        for sn in range(len(self.Y)):
            numerateur= (self.Y[sn] - MB + Alpha*self.data[sn,0] - Beta*self.data[sn,1]) - distance_modulus(self.zhelio[sn],self.zcmb[sn])
            denominateur=(A.dot(N.dot(self.data_cov[sn],A.reshape(1,len(A)).T)))+self.disp_intrinseque**2+self.dmz[sn]**2
            chi2[sn]=numerateur**2/denominateur
            self.residu[sn]=numerateur
            self.VAR[sn]=denominateur
        self.chi2=N.sum(chi2)

    def Minuit_chi2(self):

        def _compute_chi2(alpha,beta,mb):


            self.comp_chi2(alpha,beta,mb)
            print 'alpha :', alpha, 'beta :', beta ,' mb:', mb , ' chi2:',self.chi2 

            return self.chi2

        Find_param=minuit.Minuit(_compute_chi2, alpha=-0.15,beta=2.9,mb=-19.1)
            
        Find_param.migrad()
        self.Params=Find_param.values
        self.Params_Covariance=Find_param.covariance
        self.comp_chi2(self.Params['alpha'],self.Params['beta'],self.Params['mb'])

        calls=0
        if abs((self.chi2/(self.dof))-1.)>0.1:
            while abs((self.chi2/(self.dof))-1.)>0.001:
        
                if calls<100:
                    print 'je cherche de la dispersion pour la %i eme fois'%(calls+1)
                    self._compute_dispertion()
                    self.dispertion_intrinseque=copy.deepcopy(self.disp)
                    Find_param=minuit.Minuit(_compute_chi2, alpha=-0.13,beta=2.81,mb=-19.1)
                
                    Find_param.migrad()
                    self.Params=Find_param.values
                    self.Params_Covariance=Find_param.covariance
                    self.comp_chi2(self.Params['alpha'],self.Params['beta'],self.Params['mb'])
                    calls+=1

                else:
                    print 'error : calls limit are exceeded'
                break
        self.y_corrected=self.residu+self.Params['mb']
        self.y_error_corrected=N.sqrt(self.VAR-self.disp_intrinseque**2)
        self.WRMS,self.WRMS_err=comp_rms(self.residu, self.dof, err=True, variance=self.VAR)
            
    def _compute_dispertion(self):
        self.disp=copy.deepcopy(self.disp_intrinseque)
        self.disp=optimize.fmin(self._disp_function,self.disp)[0]
             
    def _disp_function(self,d):
        self.disp_intrinseque=d
        self.comp_chi2(self.Params['alpha'],self.Params['beta'],self.Params['mb'])
        return abs((self.chi2/self.dof)-1.)

    def Make_hubble_diagram(self):

        self.Minuit_chi2()
        self.LUM=distance_modulus(self.zhelio,self.zcmb)


class hubble_salt2(Hubble_diagram):

    def __init__(self,sn_name = None):

        lds = sugar.load_data_sugar()
        lds.load_salt2_data()

        if sn_name is None:
            Filtre = N.array([True]*len(lds.sn_name))
        else:
            Filtre = N.array([True]*len(lds.sn_name))
            for sn in range(len(lds.sn_name)):
                if lds.sn_name[sn] not in sn_name:
                    Filtre[sn] = False
        
        cov = N.zeros((N.sum(Filtre),3,3))
        data = N.zeros((N.sum(Filtre),2))
        data[:,0] = lds.X1[Filtre]
        data[:,1] = lds.C[Filtre]

        cov[:,0,0] = (lds.mb_err**2)[Filtre]
        cov[:,1,1] = (lds.X1_err**2)[Filtre]
        cov[:,2,2] = (lds.C_err**2)[Filtre]
        cov[:,1,2] = lds.X1_C_cov[Filtre]
        cov[:,2,1] = lds.X1_C_cov[Filtre]
        cov[:,0,1] = lds.X1_mb_cov[Filtre]
        cov[:,1,0] = lds.X1_mb_cov[Filtre]
        cov[:,0,2] = lds.C_mb_cov[Filtre]
        cov[:,2,0] = lds.C_mb_cov[Filtre]
        
        Hubble_diagram.__init__(self,lds.mb[Filtre],data,cov,lds.zhelio[Filtre],lds.zcmb[Filtre],lds.zerr[Filtre])
        self.Make_hubble_diagram()

class Hubble_diagram_sugar:

    def __init__(self,Phot,Data,Data_cov,z_err):

        self.Y=copy.deepcopy(Phot)
        self.dmz = N.zeros(len(z_err))
        #self.dmz = 5/N.log(10) * N.sqrt(z_err**2 + 0.001**2) / zcmb

        self.data = Data
        self.data_cov = Data_cov
        self.z_err = z_err

        self.disp_intrinseque=0.
        self.disp=0.
        self.dof=len(self.Y)-3.

        
    def comp_chi2(self,alpha1,alpha2,alpha3,beta,MB):
                    
        chi2=N.zeros(len(self.Y))
        self.residu=N.zeros(len(self.Y))
        self.VAR=N.zeros(len(self.Y))
        A=N.array([1.,beta,alpha1,alpha2,alpha3])
        for sn in range(len(self.Y)):
            numerateur= (self.Y[sn] - MB -beta*self.data[sn,0] - alpha1*self.data[sn,1] - alpha2*self.data[sn,2]- alpha3*self.data[sn,3])
            denominateur=(A.dot(N.dot(self.data_cov[sn],A.reshape(1,len(A)).T)))+self.disp_intrinseque**2+self.dmz[sn]**2
            chi2[sn]=numerateur**2/denominateur
            self.residu[sn]=numerateur
            self.VAR[sn]=denominateur
        self.chi2=N.sum(chi2)

    def Minuit_chi2(self):

        def _compute_chi2(alpha1,alpha2,alpha3,beta,mb):
            self.comp_chi2(alpha1,alpha2,alpha3,beta,mb)
            return self.chi2

        Find_param=minuit.Minuit(_compute_chi2, alpha1=0,alpha2=0,alpha3=0,beta=0,mb=0)
            
        Find_param.migrad()
        self.Params=Find_param.values
        self.Params_Covariance=Find_param.covariance
        self.comp_chi2(self.Params['alpha1'],self.Params['alpha2'],self.Params['alpha3'],self.Params['beta'],self.Params['mb'])

        calls=0
        if abs((self.chi2/(self.dof))-1.)>0.1:
            while abs((self.chi2/(self.dof))-1.)>0.001:
        
                if calls<100:
                    print 'je cherche de la dispersion pour la %i eme fois'%(calls+1)
                    self._compute_dispertion()
                    self.dispertion_intrinseque=copy.deepcopy(self.disp)
                    Find_param=minuit.Minuit(_compute_chi2, alpha1=0,alpha2=0,alpha3=0,beta=0,mb=0)
                
                    Find_param.migrad()
                    self.Params=Find_param.values
                    self.Params_Covariance=Find_param.covariance
                    self.comp_chi2(self.Params['alpha1'],self.Params['alpha2'],self.Params['alpha3'],self.Params['beta'],self.Params['mb'])
                    calls+=1

                else:
                    print 'error : calls limit are exceeded'
                break
        self.y_corrected=self.residu+self.Params['mb']
        self.y_error_corrected=N.sqrt(self.VAR-self.disp_intrinseque**2)
        self.WRMS,self.WRMS_err=comp_rms(self.residu, self.dof, err=True, variance=self.VAR)
            
    def _compute_dispertion(self):
        self.disp=copy.deepcopy(self.disp_intrinseque)
        self.disp=optimize.fmin(self._disp_function,self.disp)[0]
             
    def _disp_function(self,d):
        self.disp_intrinseque=d
        self.comp_chi2(self.Params['alpha1'],self.Params['alpha2'],self.Params['alpha3'],self.Params['beta'],self.Params['mb'])
        return abs((self.chi2/self.dof)-1.)

    def Make_hubble_diagram(self):
        self.Minuit_chi2()

class hubble_sugar(Hubble_diagram_sugar):

    def __init__(self):

        dic = cPickle.load(open('data_output/sugar_parameters.pkl'))
        
        cov = N.zeros((len(dic.keys()),5,5))
        data = N.zeros((len(dic.keys()),4))
        sn_name = dic.keys()
        mb = N.zeros(len(dic.keys()))

        for i in range(len(dic.keys())):
            data[i,0] = dic[sn_name[i]]['Av']
            data[i,1] = dic[sn_name[i]]['q1']
            data[i,2] = dic[sn_name[i]]['q2']
            data[i,3] = dic[sn_name[i]]['q3']
            cov[i] = dic[sn_name[i]]['cov_q']
            mb[i] = dic[sn_name[i]]['grey']
        
        Hubble_diagram_sugar.__init__(self,mb,data,cov,N.zeros(len(dic.keys())))
        self.Make_hubble_diagram()


        
if __name__=="__main__":

    dic = cPickle.load(open('data_output/emfa_output.pkl'))
    sn_name = N.array(dic['sn_name'])[dic['filter']]

    hss = hubble_sugar()
    hs = hubble_salt2(sn_name=sn_name)
    dic = cPickle.load(open('data_output/sugar_parameters.pkl'))
    grey = hss.residu#N.array([dic[sn]['grey'] for sn in dic.keys()])

    print len(grey)
    print len(hs.residu)
    
    binning = N.linspace(-0.55,0.40,10)

    import pylab as P
    from matplotlib import rc, rcParams
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Ellipse

    P.figure()
    P.hist(hs.residu,bins=binning,color='r',histtype='step',lw=5, label = 'SALT2 Hubble Residual (STD = %.2f mag)'%(N.std(hs.residu)),alpha=0.7)
    P.hist(grey,bins=binning,color='b',histtype='step',lw=5,label = 'SUGAR $\Delta M_{Grey}$ (STD = %.2f mag)'%(N.std(grey)),alpha=0.7)
    P.ylim(0,50)
    P.ylabel('# of SNIa',fontsize=20)
    P.xlabel('residuals (mag)',fontsize=20)
    P.legend()

            
        

    

    


                                
    
