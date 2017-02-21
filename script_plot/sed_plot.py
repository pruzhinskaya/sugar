import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as P
import numpy as N
import cPickle
import scipy.interpolate as inter
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics 
import copy
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from pca_plot import plot_matrix
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons
import sys,os,optparse     
from Load_time_spectra import d_l
from scipy.stats import norm as NORMAL_LAW

rep_ACE='/sps/snovae/user/leget/ACE/'
rep_BEDELL='/sps/snovae/user/leget/BEDELL/'

# control plot for the Hubble fit in spectro

def read_option():

    usage = "usage: [%prog] -s sn_name -f fig_name [otheroptions]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--sn","-s",dest="sn_name",help="sn name",default=None)
    parser.add_option("--fig","-f",dest="fig_name",help="fig name",default=None)
    parser.add_option("--figr","-r",dest="fig_name_res",help="fig name res",default=None)
    option,args = parser.parse_args()



    return option



def genrerate_SED_plot(Phase=5):

    GS=[]
    f = P.figure(figsize=(14,20))
    SIZE=N.linspace(0.04,0.96,Phase+1)
    P.suptitle("GirdSpec w/ different subplotpars")
    print Phase
    for i in range(Phase):
        GS.append(GridSpec(2,1,height_ratios=[2,1]))
        GS[i].update(left=0.05, right=0.48,bottom=SIZE[i]+0.01,top=SIZE[i+1]-0.01, hspace=0.01)

    GS.append(GridSpec(1, 1))
    GS[Phase].update(left=0.55, right=0.98, hspace=0.05,bottom=0.05,top=0.95)

    return f,GS





class SUGAR_plot:

    def __init__(self,dico_hubble_fit,dic_max,rep_spectrum,SUGAR_parameter=None,SALT2_pkl=None):
                
        if SUGAR_parameter is not None:
            self.dicS=cPickle.load(open(SUGAR_parameter))
        else:
            self.dicS=None

        if SALT2_pkl is not None:
            self.dicSA=cPickle.load(open(SALT2_pkl))
        else:
            self.dicS=None

        # Load the hubble fit data
        self.dic=cPickle.load(open(rep_spectrum))

        dico = cPickle.load(open(dico_hubble_fit))
        #self.key=dico['key']
        self.Y_err=dico['Mag_all_sn_err']
        self.Mag_dered=dico['Mag_no_corrected']
        self.Mag_red=N.zeros(N.shape(self.Mag_dered))
        self.alpha=dico['alpha']
        self.M0=dico['M0']
        self.number_correction=dico['number_correction']
        self.xplus=dico['xplus']
        self.data=dico['data']
        self.Cov_error=dico['Cov_error']
        self.X=dico['X']
        self.sn_name=dico['sn_name']
        self.CHI2=dico['CHI2']
        self.trans=dico['delta_M_grey']
        self.corr_matrix=dico['corr_matrix']

        del dico

        dico=cPickle.load(open(dic_max))
        self.Rv=dico['RV']
        self.Av_cardel=dico['Av_cardelli']
        self.sn_name_cardell=dico['sn_name']

        for i,sn in enumerate(self.sn_name):
            for j,SN in enumerate(self.sn_name_cardell):
                if sn==SN:
                    self.Mag_red[i]=copy.deepcopy(self.Mag_dered[i])+self.Av_cardel[j]*Astro.Extinction.extinctionLaw(self.X,Rv=self.Rv,law='CCM89')

        floor_filter=N.array([False]*len(self.X))

        for Bin in range(len(self.X)):
            if self.X[Bin]>6360. and self.X[Bin]<6600.:
                floor_filter[Bin]=True
            else:
                continue
        self.floor_filter=floor_filter


    def plot_chi2(self):
        P.figure()
        itera=N.linspace(1,len(self.CHI2[0]),len(self.CHI2[0]))
        P.plot(itera,self.CHI2[0],'b')
        P.xlabel('iteration')
        P.ylabel(r'$\chi^2$')
        P.legend()
        P.show()


    def plot_Factor_vs_redshift(self,dic_meta,hi=False):

        if hi:
            data=self.xplus
        else:
            data=self.data



        Z=N.zeros(len(data[:,0]))
        dico=cPickle.load(open(dic_meta))

        for i in range(len(self.sn_name)):
            Z[i]=dico[self.sn_name[i]]['host.zcmb']

            
        P.figure(figsize=(10,12))
        P.subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.98,hspace=0.001)
        for i in range(3):
            P.subplot(3,1,i+1)
            RHO=N.corrcoef(Z,data[:,i])[0,1]
            SIG=Statistics.correlation_significance(RHO,len(Z),sigma=True)
            P.scatter(Z,data[:,i],label=r'$\rho=%.2f$ ($\sigma=%.2f$)'%((RHO,abs(SIG))))
            if i !=2:
                P.xticks([-1,12],['',''])
            P.xlim(0,N.max(Z)+0.01)
            if hi:
                P.ylabel('$h_{%i}$'%(i+1),fontsize=16)
            else:
                P.ylabel('$q_{%i}$'%(i+1),fontsize=16)
            P.legend(loc=4)
        P.xlabel('redshift',fontsize=16)

    def plot_hist_grey_grey(self,dic_SUGAR):

        dic =cPickle.load(open(dic_SUGAR))
        grey=[]
        for i in range(len(self.sn_name)):
            grey.append(dic[self.sn_name[i]]['Grey'])

        P.figure(figsize=(12,8))
        P.subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        P.subplot(1,2,1)
        P.hist(grey,bins=N.linspace(-0.8,0.8,33),label='RMS=%.2f'%(N.std(grey)))
        P.ylabel('Number of supernovae',fontsize=16)
        P.xlabel('$\Delta M_{grey}$',fontsize=16)
        P.xlim(-0.8,0.8)
        P.ylim(0,27)
        P.legend()

        P.subplot(1,2,2)
        RHO=N.corrcoef(self.trans,grey)[0,1]
        P.scatter(self.trans,grey,label=r'$\rho=%.2f$'%(RHO))
        P.plot(self.trans,self.trans,'r',linewidth=3)
        P.legend(loc=2)
        P.xlabel('$\Delta M_{grey}$ during SUGAR construction',fontsize=16)
        P.ylabel('$\Delta M_{grey}$ from SUGAR fitting',fontsize=16)





    def plot_hist_grey(self,Meta_pkl,Grey_salt=False):

        dic =cPickle.load(open(Meta_pkl))
        if Grey_salt:
            P.subplot(1,2,1)
        else:
            P.figure()
        P.hist(self.trans,bins=N.linspace(-0.5,0.5,21),label='RMS=%.2f'%(N.std(self.trans)))
        P.ylabel('Number of supernovae',fontsize=16)
        P.xlabel('$\Delta M_{grey}$',fontsize=16)
        P.xlim(-0.5,0.5)
        P.ylim(0,27)
        P.legend()

        X0=N.zeros(len(self.trans))
        X0_err=N.zeros(len(self.trans))

        if Grey_salt:
            toto=[]
            for i in range(len(self.trans)):
                X0[i]=-2.5*N.log(dic[self.sn_name[i]]['salt2.X0'])/N.log(10)
                toto.append(dic[self.sn_name[i]]['salt2.X0'])
                X0[i]+= -5.*N.log10(d_l(dic[self.sn_name[i]]['host.zcmb'],SNLS=True))+5.
                X0_err[i]=2.5*dic[self.sn_name[i]]['salt2.X0.err']/(dic[self.sn_name[i]]['salt2.X0']*N.log(10))
    


            P.subplot(1,2,2)
            P.hist(X0-N.mean(X0),bins=N.linspace(-0.8,0.8,33),color='r',label='RMS=%.2f'%(N.std(X0)))
            P.xlabel('$-2.5\log(X0)-\mu_{\Lambda CDM}-cst$',fontsize=16)
            P.xlim(-0.8,0.8)
            P.ylim(0,27)
            P.legend()



    def plot_Grey_vs_SALT2(self,Meta_pkl):

        dic =cPickle.load(open(Meta_pkl))
        Color=N.zeros(len(self.trans))
        Color_err=N.zeros(len(self.trans))
        X1=N.zeros(len(self.trans))
        X1_err=N.zeros(len(self.trans))
        X0=N.zeros(len(self.trans))
        X0_err=N.zeros(len(self.trans))

        

        for i in range(len(Color)):
            Color[i]=dic[self.sn_name[i]]['salt2.Color']
            Color_err[i]=dic[self.sn_name[i]]['salt2.Color.err']
            X1[i]=dic[self.sn_name[i]]['salt2.X1']
            X1_err[i]=dic[self.sn_name[i]]['salt2.X1.err']
            X0[i]=-2.5*N.log(dic[self.sn_name[i]]['salt2.X0'])/N.log(10)
            X0_err[i]=2.5*dic[self.sn_name[i]]['salt2.X0.err']/(dic[self.sn_name[i]]['salt2.X0']*N.log(10))
            self.trans[i]-= -5.*N.log10(d_l(dic[self.sn_name[i]]['host.zcmb'],SNLS=True))+5.


        P.figure(figsize=(16,8))
        P.subplot(1,3,1)
        P.subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        Rho=N.corrcoef(self.trans,X0)[0,1]
        P.scatter(self.trans,X0,label=r'$\rho=%.2f$'%(Rho))
        P.errorbar(self.trans,X0,linestyle='', yerr=X0_err,xerr=None,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.ylabel('SALT2 X0',fontsize=16)
        P.xlabel('SUGAR $\Delta M_{grey}$',fontsize=16)
        P.legend(loc=4)
        P.subplot(1,3,2)
        Rho=N.corrcoef(self.trans,X1)[0,1]
        P.scatter(self.trans,X1,label=r'$\rho=%.2f$'%(Rho))
        P.errorbar(self.trans,X1,linestyle='', yerr=X1_err,xerr=None,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.ylabel('SALT2 X1',fontsize=16)
        P.xlabel('$SUGAR \Delta M_{grey}$',fontsize=16)
        P.legend(loc=4)
        P.subplot(1,3,3)
        Rho=N.corrcoef(self.trans,Color)[0,1]
        P.scatter(self.trans,Color,label=r'$\rho=%.2f$'%(Rho))
        P.errorbar(self.trans,Color,linestyle='', yerr=Color_err,xerr=None,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.ylabel('SALT2 Color',fontsize=16)
        P.xlabel('SUGAR $\Delta M_{grey}$',fontsize=16)
        P.legend(loc=4)



    def plot_std_time(self):

        STD=N.std(self.Mag_red,axis=0)
        STD_dered=N.std(self.Mag_dered,axis=0)
        STD_shape=N.zeros(len(self.M0))
        Y_correc=copy.deepcopy(self.Mag_dered)
        
        #print 'be carrefull the last component is not used'
        for sn in range(len(self.sn_name)):
            for i in range(self.number_correction):
                Y_correc[sn]-=self.data[sn,i]*self.alpha[:,i]
            Y_correc[sn]-=self.trans[sn]

        STD_shape=N.std(Y_correc,axis=0)
        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        STD = STD[reorder]
        STD_dered = STD_dered[reorder]
        STD_shape = STD_shape[reorder]

        X=self.X[reorder]
        P.figure(figsize=(12,14))
        compt=0
        XLAB=[1,2,3,4]
        YLAB=[2,4,6]
        for i in range(19):
            if (-12+(3*i))%12==0 and compt<7:
                P.subplot(3,2,compt+1)
                print compt + 1
                print (-12+(3*i))
                print ''
                P.plot(X[i*190:(i+1)*190],STD[i*190:(i+1)*190],'r')
                P.plot(X[i*190:(i+1)*190],STD_dered[i*190:(i+1)*190],'b')
                P.plot(X[i*190:(i+1)*190],STD_shape[i*190:(i+1)*190],'k')
                if (-12+(3*i))!=0:
                    P.title('Phase = %i days'%((-12+(3*i))))
                else:
                    P.title('Phase = %i days'%((-12+(3*i))))
                if compt+1 in XLAB:
                    P.xticks([2500.,9500.],['toto','pouet'])
                else:
                    P.xlabel('wavelength $[\AA]$')
                if compt+1 in YLAB:
                    P.yticks([-0.5,15],['toto','pouet'])
                else:
                    P.ylabel('STD')
                P.xlim(3300,8700)
                P.ylim(0.,0.6)
                compt+=1
            if i==18:
                P.subplot(3,2,compt+1)
                P.plot(X[i*190:(i+1)*190],STD[i*190:(i+1)*190],'r')
                P.plot(X[i*190:(i+1)*190],STD_dered[i*190:(i+1)*190],'b')
                P.plot(X[i*190:(i+1)*190],STD_shape[i*190:(i+1)*190],'k')
                P.title('Phase = 42 days')
                P.xlabel('wavelength $[\AA]$')
                P.yticks([-0.5,15],['toto','pouet'])
                P.xlim(3300,8700)
                P.ylim(0.,0.6)

                compt+=1

                                
        #P.show()
        self.STD_shape=STD_shape
        
    def plot_M0_time(self):

        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        X=self.X[reorder]
        M0=self.M0[reorder]
        P.figure(figsize=(12,14))
        compt=0
        XLAB=[1,2,3,4]
        YLAB=[2,4,6]
        
        for i in range(19):
            if (-12+(3*i))%12==0 and compt<7:
                P.subplot(3,2,compt+1)
                print compt + 1
                print (-12+(3*i))
                print ''
                P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]+15,'b')
                if (-12+(3*i))!=0:
                    P.title('Phase = %i days'%((-12+(3*i))))
                else:
                    P.title('Phase = %i days'%((-12+(3*i))))
                if compt+1 in XLAB:
                    P.xticks([2500.,9500.],['toto','pouet'])
                else:
                    P.xlabel('wavelength $[\AA]$')
                if compt+1 in YLAB:
                    #P.yticks([-0.5,15],['toto','pouet'])
                    print 'prout'
                else:
                    P.ylabel('$M_0(\lambda)$ + cst')
                P.xlim(3300,8700)
                P.gca().invert_yaxis()
                compt+=1
            if i==18:
                P.subplot(3,2,compt+1)
                P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]+15,'b')
                P.title('Phase = 42 days')
                P.xlabel('wavelength $[\AA]$')
                #P.yticks([-0.5,15],['toto','pouet'])
                P.xlim(3300,8700)
                P.gca().invert_yaxis()
                compt+=1

                                
        P.show()


#    def plot_spectrophtometric_effec_time(self,comp=0):
#
#        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
#        X=self.X[reorder]
#        M0=self.M0[reorder]
#        ALPHA=self.alpha[:,comp][reorder]
#        compt=0
#        XLAB=[1,2,3,4]
#        YLAB=[2,4,6]
#        f,GSS=genrerate_SED_plot(Phase=6)
#        for i in range(19):
#            if (-12+(3*i))%12==0 and compt<7:
#                P.subplot(GSS[5-compt][0])
#                print compt + 1
#                print (-12+(3*i))
#                print ''
#                P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42],'b')

    
                
                #y_moins=self.M0-self.alpha[:,correction]*(mean(self.data[:,correction])+sqrt(var(self.data[:,correction])))
                #y_plus=self.M0+self.alpha[:,correction]*(mean(self.data[:,correction])+sqrt(var(self.data[:,correction])))

                #P.fill_between(self.X,self.M0+19.2,y_plus,color='m',alpha=0.7 )
                #P.fill_between(self.X,y_moins,self.M0+19.2,color='g',alpha=0.7)

#                if (-12+(3*i))!=0:
#                    P.title('Phase = %i days'%((-12+(3*i))))
#                else:
#                    P.title('Phase = %i day'%((-12+(3*i))))
#                P.xticks([2500.,9500.],['toto','pouet'])
#                P.ylim(min(M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42])-0.2,max(M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42])+0.2)
#                P.ylabel('$M_0(\lambda)$ + cst')
#                P.xlim(3300,8700)
#                P.gca().invert_yaxis()
#
#                P.subplot(GSS[5-compt][1])
#                P.plot(X[i*190:(i+1)*190],ALPHA[i*190:(i+1)*190],'b')
#                P.xticks([2500.,9500.],['toto','pouet'])
#                P.ylim(min(ALPHA),max(ALPHA))
#                P.xlim(3300,8700)
#                compt+=1
#
#            if i==18:
#                P.subplot(GSS[5-compt][0])
#                P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42],'b')
#                P.title('Phase = 42 days')
#                P.ylabel('$M_0(\lambda)$ + cst')
#                P.ylim(min(M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42])-0.2,max(M0[i*190:(i+1)*190]-M0[i*190:(i+1)*190][42])+0.2)
#                P.xlim(3300,8700)
#                P.gca().invert_yaxis()
#
#                P.subplot(GSS[5-compt][1])
#                P.plot(X[i*190:(i+1)*190],ALPHA[i*190:(i+1)*190],'b')
#                P.ylim(min(ALPHA),max(ALPHA))
#                P.xlim(3300,8700)
#                compt+=1

#        P.subplot(GSS[6][:, :-1])
     

    def plot_spectrophtometric_effec_time(self,comp=0):

        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        X=self.X[reorder]
        M0=self.M0[reorder]
        ALPHA=self.alpha[:,comp][reorder]
     
        CST=N.mean(M0)
        fig,ax1=P.subplots(figsize=(14,20))
        P.subplots_adjust(left=0.05, right=0.93,bottom=0.05,top=0.97)
        Time=N.linspace(-12,42,19)
        Y2_label=[]
        Y2_pos=[]
        for i in range(19):
            
            if i%2==0:
                if (-12+(3*i))!=0:
                    Y2_label.append('%i days'%(Time[i]))
                else:
                    Y2_label.append('%i day'%(Time[i]))

                Y2_pos.append(M0[i*190:(i+1)*190][-1]-CST)

                ax1.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,'b')

                
                y_moins=M0[i*190:(i+1)*190]-ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))
                y_plus=M0[i*190:(i+1)*190]+ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))

                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_plus-CST,color='m',alpha=0.7 )
                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_moins-CST,color='g',alpha=0.7)

                if i==0:
                    CST-=3
                else:
                    CST-=2


        ax1.set_ylabel('$M_0(t,\lambda)$ + cst',fontsize=16)
        ax1.set_xlim(3300,8620)
        ax1.set_ylim(-1,23)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=16)
        p1 = P.Rectangle((0, 0), 1, 1, fc="magenta")
        p2 = P.Rectangle((0, 0), 1, 1, fc="green")
        ax1.legend([p1, p2], ['+1$\sigma$', '-1$\sigma$'],loc=4)
        P.title(r'Average spectrum with $\pm$1$\sigma$ variation ($\alpha^{%i}$)'%(comp+1),fontsize=16)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(-1,23)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])


        

        
            
            #P.plot(X[i*190:(i+1)*190],ALPHA[i*190:(i+1)*190],'b')
             



    def plot_alpha_time(self):

        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        X=self.X[reorder]
        for comp in range(3):
            M0=self.alpha[:,comp][reorder]
            P.figure(figsize=(12,14))
            compt=0
            XLAB=[1,2,3,4]
            YLAB=[2,4,6]
            for i in range(19):
                if (-12+(3*i))%12==0 and compt<7:
                    P.subplot(3,2,compt+1)
                    print compt + 1
                    print (-12+(3*i))
                    print ''
                    P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190],'b')
                    if (-12+(3*i))!=0:
                        P.title('Phase = %i days'%((-12+(3*i))))
                    else:
                        P.title('Phase = %i days'%((-12+(3*i))))
                    if compt+1 in XLAB:
                        P.xticks([2500.,9500.],['toto','pouet'])
                    else:
                        P.xlabel('wavelength $[\AA]$')
                    if compt+1 in YLAB:
                    #P.yticks([-0.5,15],['toto','pouet'])
                        print 'prout'
                    else:
                        P.ylabel(r'$\alpha_{%i}(\lambda)$'%(comp+1))
                    P.xlim(3300,8700)

                    compt+=1
                if i==18:
                    P.subplot(3,2,compt+1)
                    P.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190],'b')
                    P.title('Phase = 42 days')
                    P.xlabel('wavelength $[\AA]$')
                #P.yticks([-0.5,15],['toto','pouet'])
                    P.xlim(3300,8700)
                    compt+=1
            #P.savefig('../these_plot/plot_octobre_2015/Alpha%i_time.pdf'%(comp+1))
                                
    

class plot_reconstruct:

    def __init__(self,DIC_spectra,SUGAR_model,Dic_sugar_parameter,dic_at_max,dic_salt=None):

        self.dic=cPickle.load(open(DIC_spectra))
        self.sn_name=self.dic.keys()
        self.dicS=cPickle.load(open(Dic_sugar_parameter))
        if dic_salt is not None :
            self.dicSA=cPickle.load(open(dic_salt))
        else:
            self.dicSA=None
        dic_at_max=cPickle.load(open(dic_at_max))
        self.Rv=dic_at_max['RV']
        SUGAR=N.loadtxt(SUGAR_model)
        self.M0=SUGAR[:,2]
        self.alpha=SUGAR[:,3:6]


    def plot_spectra_reconstruct(self,sn,T_min=-12,T_max=42):
        
        fig,ax1=P.subplots(figsize=(10,8))
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.07,top=0.95)
        #P.figure(figsize=(10,8))
        #P.figure(figsize=(5,8))
        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(-12,42,19)
        DAYS=[-999]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    OFFSET.append(Off+0.2-N.mean(self.dic[sn]['0']['Y']))
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k',label='Observed spectra')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(d_l(self.dicSA[sn]['%i'%(j)]['z_cmb'],SNLS=True))+5.
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(d_l(self.dicSA[sn]['%i'%(j)]['z_cmb'],SNLS=True))+5.
                            ax1.plot(self.dicSA[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2)

                    moins=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    #P.text(self.dic[sn]['%i'%(j)]['X'][-10],self.dic[sn]['%i'%(j)]['Y'][-1]+OFFSET[Off], '%10.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    Y2_pos.append(self.dic[sn]['%i'%(j)]['Y'][-1]+OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1


        Phase=N.array(Phase)
    

        Reconstruction=N.zeros((len(Phase),190))
        for Bin in range(190):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*19:(Bin+1)*19])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*19:(Bin+1)*19])
            
                Reconstruction[:,Bin]+=self.dicS[sn]['x%i'%(i+1)]*SPLINE(Phase)

            Reconstruction[:,Bin]+=self.dicS[sn]['Grey']
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*Astro.Extinction.extinctionLaw(self.dic[sn]['0']['X'][Bin],Rv=self.Rv,law='CCM89')

        MIN=10**25
        MAX=-10**25

        #if self.dicSA is not None:

        #    for j in range(len(Phase)):
        #        if Phase[j]==self.dicSA[sn]['Time'][j]:
        #            FFF=(self.dicSA[sn]['x']>self.dic[sn]['%i'%(j)]['X'][0])
        #            FFF=FFF & (self.dicSA[sn]['x']<self.dic[sn]['%i'%(j)]['X'][-1])
        #            if j==0:
        #                ax1.plot(self.dicSA[sn]['x'][FFF],self.dicSA[sn]['Y_cosmo'][j][FFF]+OFFSET[j],'r',label='SALT2.4 model',lw=2)
        #            else:
        #                ax1.plot(self.dicSA[sn]['x'][FFF],self.dicSA[sn]['Y_cosmo'][j][FFF]+OFFSET[j],'r',lw=2)
        #        print Phase[j]-self.dicSA[sn]['Time'][j]


        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',lw=2)

            if N.min(Reconstruction[j]+OFFSET[j])<MIN:
                MIN=N.min(Reconstruction[j]+OFFSET[j])
                
            if N.max(Reconstruction[j]+OFFSET[j])>MAX:
                MAX=N.max(Reconstruction[j]+OFFSET[j])



        #if sn in self.sn_name_cardell:
        #    P.title(sn+ ' in training')
        #else:
        #    P.title(sn+ ' not in training')
        P.title(sn)
        ax1.set_xlim(3300,8600)
        #P.xlim(5500,6500)
        ax1.set_ylim(MIN-0.5,MAX+3)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ +cst',fontsize=12)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=12)
        ax1.legend(loc=4)
        P.gca().invert_yaxis()
        
        ax2=ax1.twinx()
        ax2.set_ylim(MIN-0.5,MAX+3)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])



    def plot_spectra_reconstruct_residuals(self,sn,T_min=-12,T_max=42):

        #P.figure(figsize=(10,8))
        fig,ax1=P.subplots(figsize=(10,8))
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.07,top=0.95)

        RESIDUAL_SUGAR=[]
        RESIDUAL_SALT2=[]
        ERROR_SPECTRA=[]

        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(-12,42,19)
        DAYS=[-999]
        JJ=[]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    JJ.append(j)
                    OFFSET.append(Off+0.2)
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],'k',label='Observed spectra')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(d_l(self.dicSA[sn]['%i'%(j)]['z_cmb'],SNLS=True))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS))]
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],'k')
                        if self.dicSA:


                            
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(d_l(self.dicSA[sn]['%i'%(j)]['z_cmb'],SNLS=True))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS)])
                            
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2)
                            
                    moins=OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    #ax1.text(self.dic[sn]['%i'%(j)]['X'][-10],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))[-1]+OFFSET[Off], '%10.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    Y2_pos.append(OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1
                    

                    ERROR_SPECTRA.append(N.sqrt(self.dic[sn]['%i'%(j)]['V']))

        Phase=N.array(Phase)

        Reconstruction=N.zeros((len(Phase),190))
        for Bin in range(190):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*19:(Bin+1)*19])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*19:(Bin+1)*19])
                Reconstruction[:,Bin]+=self.dicS[sn]['x%i'%(i+1)]*SPLINE(Phase)

            Reconstruction[:,Bin]+=self.dicS[sn]['Grey']
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*Astro.Extinction.extinctionLaw(self.dic[sn]['0']['X'][Bin],Rv=self.Rv,law='CCM89')

        MIN=10**25
        MAX=-10**25

        #if self.dicSA is not None:
        # 
        #    for j in range(len(Phase)):
        #        FFF=(self.dicSA[sn]['x']>self.dic[sn]['%i'%(j)]['X'][0])
        #        FFF=FFF & (self.dicSA[sn]['x']<self.dic[sn]['%i'%(j)]['X'][-1])
        #        SPLINE=inter.InterpolatedUnivariateSpline(self.dic[sn]['%i'%(JJ[j])]['X'],self.dic[sn]['%i'%(JJ[j])]['Y'])
        #        SPLINE_SALT=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['x'][FFF],self.dicSA[sn]['Y_cosmo'][j][FFF])
        #        Y=SPLINE(self.dicSA[sn]['x'][FFF])
        #        Y_SALT2=SPLINE_SALT(self.dic[sn]['%i'%(JJ[j])]['X'])
        #        
        #        if j==0:
        #            ax1.plot(self.dicSA[sn]['x'][FFF],self.dicSA[sn]['Y_cosmo'][j][FFF]+OFFSET[j]-Y,'r',label='SALT2.4 model',lw=2)
        #        else:
        #            ax1.plot(self.dicSA[sn]['x'][FFF],self.dicSA[sn]['Y_cosmo'][j][FFF]+OFFSET[j]-Y,'r',lw=2)

        #        RESIDUAL_SALT2.append(Y_SALT2-self.dic[sn]['%i'%(JJ[j])]['Y'])
                
        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',lw=2)

            RESIDUAL_SUGAR.append(Reconstruction[j]-self.dic[sn]['%i'%(JJ[j])]['Y'])
                            
        
        MIN=N.min(OFFSET[0])
        MAX=N.max(OFFSET[-1])



        P.title(sn)
        ax1.set_xlim(3300,8600)
        ax1.set_ylim(MIN-2,MAX+3)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ +cst',fontsize=12)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=12)
        ax1.legend(loc=4)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(MIN-2,MAX+3)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])

        return RESIDUAL_SUGAR, RESIDUAL_SALT2, ERROR_SPECTRA

def plot_wRMS_time(wlength,matrix,ylabel,title,cmap=P.cm.jet,plotpoints=True):
    
    values = [diag(matrix,k=i) for i in range(len(matrix))]
    
    means=map(mean,values)
    stds=map(std,values)
    med,nmad=N.array([median_stats(x) for x in values]).T
    

    #Plot the matrix
    wlength=[wlength[0],wlength[-1],wlength[-1],wlength[0]]
    fig = P.figure(dpi=150,figsize=(8,8))
    ax = fig.add_axes([0.08,0.09,0.88,0.88]) #title=title
    im = ax.imshow(matrix,cmap=cmap,extent=wlength,interpolation='nearest')
    cb = fig.colorbar(im)
    cb.set_label(ylabel,size='x-large')
    ax.set_xlabel(r'Wavelength [$\AA$]',size='large')
    ax.set_ylabel(r'Wavelength [$\AA$]',size='large')
    


def plot_residual(sn_name,rep,bin=42,Plot=False):

    
#    dico={option.sn_name:{'SUGAR':sucre,
#                          'SALT2':sel,
#                          'ERROR':error}}

    residual_SUGAR=[]
    error_SPECTRA=[]
    residual_SALT2=[]
    for i,sn in enumerate(sn_name):
        dic=cPickle.load(open(rep+sn+'.pkl'))
        res_SUGAR=N.array(dic[sn]['SUGAR'])
        res_SALT2=N.array(dic[sn]['SALT2'])
        error=N.array(dic[sn]['ERROR'])
        for T in range(len(res_SUGAR[:,bin])):
            residual_SUGAR.append(res_SUGAR[T,bin])
            error_SPECTRA.append(error[T,bin])
            residual_SALT2.append(res_SALT2[T,bin])

    residual_SUGAR=N.array(residual_SUGAR)
    residual_SALT2=N.array(residual_SALT2)
    error_SPECTRA=N.array(error_SPECTRA)
    pull_SUGAR=residual_SUGAR/error_SPECTRA
    pull_SALT2=residual_SALT2/error_SPECTRA
    #P.hist(pull_SUGAR,bins=100)

    Moyenne_pull,ecart_type_pull=NORMAL_LAW.fit(pull_SUGAR)
    Moyenne_pull_SALT2,ecart_type_pull_SALT2=NORMAL_LAW.fit(pull_SALT2)
    if Plot:
        P.hist(pull_SUGAR,bins=100,normed=True)
        xmin, xmax = P.xlim()
        X = N.linspace(xmin, xmax, 100)
        PDF = NORMAL_LAW.pdf(X, Moyenne_pull, ecart_type_pull)
        P.plot(X, PDF, 'r', linewidth=3)
        title = "Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (Moyenne_pull, ecart_type_pull)
        P.title(title)
        P.show()

    #return ecart_type_pull,ecart_type_pull_SALT2
    return residual_SUGAR,residual_SALT2

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################




if __name__=='__main__':

    option = read_option()

    sps='/sps/snovae/user/leget'
    ## training
    ##SED_PKL=sps+'/CABALLO/model_training_SUGAR/SUGAR_sed_with_grey_GP_clermont_CABALLO.pkl'
    ##SED_MAX=sps+'/CABALLO/model_training_SUGAR/model_at_max_3_eigenvector_without_grey_save_before_PCA.pkl'
    ##SPECTRA_rep=sps+'/CABALLO/all_CABALLO_data_binning_training.pkl'
    ##SED=SUGAR_plot(SED_PKL,SED_MAX,SPECTRA_rep,SUGAR_parameter=sps+'/CABALLO/SUGAR_parameters.pkl',SALT2_pkl=sps+'/CABALLO/SED_SALT2_JLA_CABALLO_with_cosmology.pkl')

    ## Validation
    SED_PKL=sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/SUGAR_validation_3_eigenvector_CABALLO_test_RV.pkl'
    #SED_PKL=sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/SUGAR_validation_3_eigenvector_CABALLO_test_RV_test_AV.pkl'
    SED_MAX=sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/model_at_max_3_eigenvector_without_grey_without_MFR_problem_test_RV.pkl'
    SPECTRA_rep=sps+'/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl' 


    SED=SUGAR_plot(SED_PKL,SED_MAX,SPECTRA_rep)
    #SED.plot_spectrophtometric_effec_time(comp=0)
    #P.savefig('../These_plot/plot_phd/Chapitre9/Comment_Astier/Alpha1_spectra_effect.pdf')
    #SED.plot_spectrophtometric_effec_time(comp=1)
    #P.savefig('../These_plot/plot_phd/Chapitre9/Comment_Astier/Alpha2_spectra_effect.pdf')
    #SED.plot_spectrophtometric_effec_time(comp=2)
    #P.savefig('../These_plot/plot_phd/Chapitre9/Comment_Astier/Alpha3_spectra_effect.pdf')
    #SED.plot_hist_grey(sps+'/CABALLO/META_JLA.pkl')
    #SED.plot_Factor_vs_redshift(sps+'/CABALLO/META_JLA.pkl',hi=False)
    #SED.plot_Factor_vs_redshift(sps+'/CABALLO/META_JLA.pkl',hi=True)
        
    #SED.plot_hist_grey_grey(sps+'/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl')
    #SED.plot_Grey_vs_SALT2(sps+'/CABALLO/META_JLA.pkl')
    ##SED=plot_reconstruct(SPECTRA_rep,sps+'/CABALLO/SUGAR_validation/SUGAR_model_v1.asci',sps+'/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl',SED_MAX,dic_salt=sps+'/File_for_PF.pkl')
    ##sucre,sel,error=SED.plot_spectra_reconstruct_residuals(option.sn_name,T_min=-12,T_max=42)
    ##P.savefig(option.fig_name_res)
    ##P.close()
    ##SED.plot_spectra_reconstruct(option.sn_name,T_min=-12,T_max=42)
    ##P.savefig(option.fig_name)
    ##P.close()

    ##dico={option.sn_name:{'SUGAR':sucre,
    ##                      'SALT2':sel,
    ##                   'ERROR':error}}

    #File=open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_residual_Masha_SALT2/'+option.sn_name+'.pkl','w')
    #cPickle.dump(dico,File)
    #File.close()

    ###dic=cPickle.load(open(sps+'/CABALLO/SUGAR_validation/SUGAR_model_for_phd/SUGAR_validation_3_eigenvector_CABALLO_test_RV.pkl'))
    
    ###residu=[]
    ###for i,sn in enumerate(dic['sn_name']):
    ###    dic=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_residual_for_phd/'+sn+'.pkl'))
    ###    residu.append(dic[sn]['SALT2'])

    dic=cPickle.load(open(SED_MAX))
    SN_name=dic['sn_name']

    STD=[]
    STD_salt=[]
    for i in range(190):
        print i
        std,std_salt=plot_residual(SN_name,'/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_residual_Masha_SALT2/',bin=i)
        STD.append(N.std(std))
        STD_salt.append(N.std(std_salt))

    P.plot(dic['X'],STD_salt,'r',linewidth=3,label='SALT2.4')
    P.plot(dic['X'],STD,'b',linewidth=3,label='SUGAR')
    P.xlabel('wavelength $[\AA]$')
    P.ylabel('STD (mag)')
    P.ylim(0,0.5)
    P.xlim(3300,8700)
    P.legend(loc=3)
    P.savefig('../Post_doc_plot/decembre_2016/test_Masha_SALT2.pdf')
