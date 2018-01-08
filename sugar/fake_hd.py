import numpy as np
import sugar
import sncosmo
import lmfit
import copy
import pylab as plt
import cPickle
from scipy.integrate import quad
from scipy.stats import f as FF
from ToolBox import Statistics

def intfun(z,om,ol,w):
    return 1. / np.sqrt( (om * ((1.+z)**3)) + (ol * ((1.+z)**(3.*(1.+w)))) )

def dl(z,om,ol,w):
    clight = 299792458.
    H = 0.70
    d_l = (1+z) * (clight/H) * (np.array([quad(intfun, 0, Z, args=(om,ol,w))[0] for Z in z]))
    return d_l

def mu(z,om,ol,w):
    return 5. * np.log10(dl(z,om,ol,w)) - 5.

def prob_func(ndata, nparas, new_chi, best_chi, nfix=1.):

    nparas = nparas + nfix
    nfree = ndata - nparas
    nfix = 1.0*nfix
    dchi = new_chi / best_chi - 1.0
    return FF.cdf(dchi * nfree / nfix, nfix, nfree)

class fake_hd:

    def __init__(self,wrms,omega_m=0.3,w=-1.,MB=-19.2,simple=True):

        np.random.seed(1)

        self.wrms = wrms
        self.om = omega_m
        self.w = w
        self.noise = np.sqrt(0.15**2 - 0.10**2)
        self.sigma_int = np.sqrt(self.wrms**2 - self.noise**2)
        
        if simple:
            JLA = np.loadtxt('../../Desktop/post_doc/janvier_2017/JLA/jla_lcparams.txt',dtype ='str')
            self.zcmb = np.array(map(float,list(JLA[:,1])))

            self.nsn = len(self.zcmb)
        
            self.Mb = np.random.normal(loc=MB, scale=self.sigma_int, size=self.nsn)
            self.y_err = np.random.normal(loc=0., scale = self.noise)
        else:
            simu = cPickle.load(open('../../Desktop/post_doc/janvier_2017/JLA/simu.pkl'))
        
            self.zcmb = np.array(simu['z'])
            self.nsn = len(self.zcmb)
            z = np.linspace(0,np.max(self.zcmb),11)
            self.noise_bin = []
            self.noise = np.array(simu['error'])
            #for i in range(10):
            #    A = []
            #    for j in range(len(self.zcmb)):
            #        if self.zcmb[j]>z[i] and self.zcmb[j]<z[i+1]:
            #            A.append(self.noise[j])
            #    self.noise_bin.append(np.mean(A))
                
            self.Mb = np.random.normal(loc=MB, scale=self.sigma_int, size=self.nsn)
            #self.noise = np.zeros(len(self.Mb))
            #for i in range(10):
            for j in range(len(self.zcmb)):
                self.noise_bin.append(np.random.normal(scale = self.noise[j]))
    
            self.y_err = np.array(self.noise_bin)

        self.Mb += self.y_err
        self.y_err = np.sqrt(self.y_err**2)

        self.mu_truth = mu(self.zcmb,self.om,(1.-self.om),self.w)
        self.mb = self.mu_truth + self.Mb
        self.mb_err = self.y_err

    def hubble_fit(self):

        def chi2(param):
            residual = self.mb - param['MB'].value - mu(self.zcmb, param['om'].value,(1.-param['om'].value),param['w'].value)
            chi2 = np.sum((residual**2/(self.mb_err**2+self.sigma_int**2)))
            print chi2, chi2/(len(self.mb) - 3.)
            return chi2
        
        params = lmfit.Parameters()
        params.add('om',value=0.3,min=-2.,max=2.)
        params.add('MB',value=-19.2,min=-21.,max=-17.)
        params.add('w',value=-1.,min=0.,max=-2.)

        self.params=params
        minner = lmfit.Minimizer(chi2, params)
        self.minner=minner
        result = minner.minimize(method='nelder')
        self.result=result
        self.cosmo_SNIa_parameters={}
        for key in params.keys():
            self.cosmo_SNIa_parameters.update({key:result.params[key].value})


    def conf_interval(self,param1,param2,Number_grid):
        
        dic_lim = {'om':[0.0,0.4],
                   'MB':[-18.,-20.],
                   'w':[-1.4,-0.5]}
        
        x = np.linspace(dic_lim[param1][0],dic_lim[param1][1],Number_grid)
        y = np.linspace(dic_lim[param2][0],dic_lim[param2][1],Number_grid)

        X, Y = np.meshgrid(x,y)

        F = np.zeros_like(X)


        def _local_chi2(param):

            residual = self.mb - param['MB'] - mu(self.zcmb, param['om'],(1.-param['om']),param['w'])
            chi2 = np.sum((residual**2/(self.mb_err**2+self.sigma_int**2)))
            print chi2, chi2/(len(self.mb) - 3.)
            return chi2


        PARAMS = copy.deepcopy(self.cosmo_SNIa_parameters)
        chi2_zeros = copy.deepcopy(self.result.residual[0])

        t = 0
        for i in range(Number_grid):
            for j in range(Number_grid):
                print t,'/',Number_grid**2
                PARAMS[param1] = X[i,j]
                PARAMS[param2] = Y[i,j]
                chi2_F = _local_chi2(PARAMS)
                F[i,j] = prob_func(len(self.zcmb), len(self.zcmb)-(len(self.zcmb)-len(PARAMS)), chi2_F,chi2_zeros, nfix=2.)
                t += 1
        return X,Y,F
                                                
    def plot_contour(self):
        plt.figure()
        self.cx,self.cy,self.grid=self.conf_interval('om','w',30)
        plt.contourf(self.cx, self.cy,Statistics.pvalue2sigma(1-self.grid), np.array([np.min(Statistics.pvalue2sigma(1-self.grid)),1.,2.]),cmap=plt.cm.Blues_r,vmin=1)
        plt.contour(self.cx, self.cy, Statistics.pvalue2sigma(1-self.grid), np.array([np.min(Statistics.pvalue2sigma(1-self.grid)),1.,2.]),colors=('k',),linewidths=(3,))
        plt.ylim(-2,-0.4)
        plt.xlim(0.12,0.45)
        plt.ylabel('w',fontsize=18)
        plt.xlabel('$\Omega_m$',fontsize=18)

    
if __name__=='__main__':

    fhh = fake_hd(0.15,omega_m=0.3,w=-1.,MB=-19.2,simple=False)
    fhh.hubble_fit()
    cx, cy, cz = fhh.conf_interval('om','w', 60)
    #fhh.plot_contour()

    fh = fake_hd(0.13,omega_m=0.3,w=-1.,MB=-19.2,simple=False)
    fh.hubble_fit()
    cxx, cyy, czz = fh.conf_interval('om','w', 60)

    #plt.style.use('dark_background')
    #plt.figure(frameon=False)
    ylim = plt.ylim(-2.0,-0.41)
    xlim = plt.xlim(0.,0.45)
    plt.plot([0.3,0.3],ylim,'k--',lw=3,zorder=0)
    plt.plot(xlim,[-1,-1],'k--',lw=3,zorder=0)
    plt.contourf(cx, cy, Statistics.pvalue2sigma(1-cz), np.array([np.min(Statistics.pvalue2sigma(1-cz)),1.,2.]),cmap=plt.cm.Reds_r,vmin=1,alpha=0.8)
    plt.contourf(cxx, cyy, Statistics.pvalue2sigma(1-czz), np.array([np.min(Statistics.pvalue2sigma(1-czz)),1.,2.]),cmap=plt.cm.Blues_r,vmin=1,alpha=0.8)
    plt.ylim(-1.45,-0.55)
    plt.xlim(0.11,0.45)
    plt.ylabel('w',fontsize=18)
    plt.xlabel('$\Omega_m$',fontsize=18)
    p1 = plt.Rectangle((0, 0), 1, 1, fc="red")
    p2 = plt.Rectangle((0, 0), 1, 1, fc="blue")
    plt.legend([p1, p2], ['using SALT2', 'using SUGAR'])
                                        

    
