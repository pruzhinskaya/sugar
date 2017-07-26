import pylab as P
import numpy as N
import cPickle 
import copy 
from SUGAR_photometry_fitting import build_SUGAR_ligh_curves
from scipy.stats import norm as NORMAL_LAW
from scipy.stats import lognorm as LOGNORMAL_LAW


# TO DO :
# - lancer pour 1000


class SUGAR_MC(build_SUGAR_ligh_curves):

    def __init__(self,SUGAR_asci,hyperparameters_dat,SUGAR_parameter):

        build_SUGAR_ligh_curves.__init__(self,SUGAR_asci,hyperparameters_dat)
        self.dic_parameter=cPickle.load(open(SUGAR_parameter))


    def generate_one_SED(self,Q,Av,Grey,Time=N.linspace(-12,42,55)):
        
        self.comp_SUGAR_in_mag(Q,Av,Grey)
        self.align_time(Time)
        self.go_to_flux()
        
        self.dic={'SUGAR_param':{'q1':Q[0],
                                 'q2':Q[1],
                                 'q3':Q[2],
                                 'Av':Av,
                                 'Grey':Grey},
                  'SED':{}}


        for time in range(len(Time)): 
            self.dic['SED'].update({'%i'%(Time[time]):{'Flux':self.Flux[time*190:(time+1)*190],
                                                       'wavelength':self.wavelength}})
        


    def fit_distribution_parameter(self):

        SN=self.dic_parameter.keys()
        Q=N.zeros((len(SN),3))
        Av=N.zeros(len(SN))
        Grey=N.zeros(len(SN))
        
        for i,sn in enumerate(SN):
            for j in range(3):
                Q[i,j]=self.dic_parameter[sn]['x%i'%(j+1)]
            Av[i]=self.dic_parameter[sn]['Av']
            Grey[i]=self.dic_parameter[sn]['Grey']
        
        self.observed_param=[Q,Av,Grey]
        self.Average=[]
        self.Sigma=[]

        for i in range(3):
            if i==0:
                for j in range(3):

                    Moyenne,ecart_type=NORMAL_LAW.fit(self.observed_param[i][:,j])
                    self.Average.append(Moyenne)
                    self.Sigma.append(ecart_type)

                    #P.figure()
                    #P.hist(self.observed_param[i][:,j],normed=True)
                    #xmin, xmax = P.xlim()
                    #X = N.linspace(xmin, xmax, 100)
                    #PDF = NORMAL_LAW.pdf(X, Moyenne, ecart_type)
                    #P.plot(X, PDF, 'r', linewidth=3)

                    #P.savefig('../test%i.pdf'%(3+j))
            else:
                #P.figure()
                #P.hist(self.observed_param[i],normed=True)
                
                Moyenne,ecart_type=NORMAL_LAW.fit(self.observed_param[i])
                self.Average.append(Moyenne)
                self.Sigma.append(ecart_type)

                #self.Average.append(Moyenne)
                #self.Sigma.append(ecart_type)
                #xmin, xmax = P.xlim()
                #X = N.linspace(xmin, xmax, 100)
                #PDF = NORMAL_LAW.pdf(X, Moyenne, ecart_type)
                #P.plot(X, PDF, 'r', linewidth=3)
                #P.savefig('../test%i.pdf'%(5+i))


    def generate_MC(self,NN=1000):
        
        self.Q_MC=N.zeros((NN,3))
        self.Av_MC=N.zeros(NN)
        self.Grey_MC=N.zeros(NN)

        for i in range(NN):
            print i 
            for j in range(3):
                self.Q_MC[i,j]=N.random.normal(scale=self.Sigma[j])
            Av=-5.
            while Av<-0.2:
                Av=N.random.normal(scale=self.Sigma[3])
            self.Av_MC[i]=copy.deepcopy(Av)
            self.Grey_MC[i]=N.random.normal(scale=self.Sigma[4])

            
    def build_sample_SNIa(self,pkl_file,NNN=100):

        self.fit_distribution_parameter()
        self.generate_MC(NN=NNN)

        self.dic_MC={}

        for i in range(NNN):
            print i
            self.generate_one_SED(self.Q_MC[i],self.Av_MC[i],self.Grey_MC[i])
            self.dic_MC.update({'SNIa_%i'%(i):copy.deepcopy(self.dic)})


        File=open(pkl_file,'w')
        cPickle.dump(self.dic_MC,File)
        File.close()
            
if __name__=='__main__':

    SMC=SUGAR_MC('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_v1.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP_binning_speed_without_MFR_issue/hyperparameters.dat','/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl')
    SMC.build_sample_SNIa('/sps/snovae/user/leget/SUGAR_Monte_Carlo_SED.pkl',NNN=1000)
