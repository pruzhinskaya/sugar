from pylab import * 
import pylab as P
import numpy as N
import cPickle
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics 
import SNobject
import ResampleSpec
import copy
import EWAnalysis
from pca_analysis import passage_error,passage_error_error_sn
from diagramme_hubble import d_l
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from multilinearfit import *
from pca_plot import plot_matrix
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons


rep_ACE='/sps/snovae/user/leget/ACE/'
rep_BEDELL='/sps/snovae/user/leget/BEDELL/'

# control plot for the Hubble fit in spectro


class SUGAR_plot:

    def __init__(self,dico_hubble_fit):
                

        # Load the hubble fit data

        dico = cPickle.load(open(dico_hubble_fit))
        #self.key=dico['key']
        self.Y_err=dico['Mag_all_sn_err']
        self.Mag_no_corrected=dico['Mag_no_corrected']
        self.Mag_corrected=dico['Mag_corrected']
        self.alpha=dico['alpha']
        self.M0=dico['M0']
        self.number_correction=dico['number_correction']
        self.xplus=dico['xplus']
        #self.WRMS=dico['WRMS']
        self.Y_build=dico['Y_build']
        self.Y_build_error=dico['Y_build_error']
        self.data=dico['data']
        self.Cov_error=dico['Cov_error']
        self.X=dico['X']
        self.sn_name=dico['sn_name']
        self.Rv=dico['RV']
        self.Av=dico['Av']
        self.Av_cardel=dico['Av_cardelli']
        self.disp_matrix=dico['disp_matrix']
        self.slope=dico['reddening_law']
        self.CHI2=dico['CHI2']
        self.trans=dico['delta_M_grey']
        self.corr_matrix=dico['corr_matrix']
        self.alpha_err_Jackknife = dico['alpha_err_Jackknife']
        self.M0_err_Jackknife = dico['M0_err_Jackknife']
        self.reddening_law_err_Jackknife = dico['reddening_law_err_Jackknife']
        self.RV_err_Jackknife = dico['RV_err_Jackknife']
        self.dico=dico


        floor_filter=array([False]*len(self.X))

        for Bin in range(len(self.X)):
            if self.X[Bin]>6360. and self.X[Bin]<6600.:
                floor_filter[Bin]=True
            else:
                continue
        self.floor_filter=floor_filter


    def plot_chi2(self):
        figure()
        itera=linspace(1,len(self.CHI2[0]),len(self.CHI2[0]))
        plot(itera,self.CHI2[0],'b')
        xlabel('iteration')
        ylabel(r'$\chi^2$')
        legend()
        show()


    def plot_Av_vs_color(self,Meta_pkl):

        dic =cPickle.load(open(Meta_pkl))
        Color=N.zeros(len(self.Av))
        Color_err=N.zeros(len(self.Av))
        X1=N.zeros(len(self.Av))
        X1_err=N.zeros(len(self.Av))

        for i in range(len(Color)):
            Color[i]=dic[self.sn_name[i]]['salt2.Color']
            Color_err[i]=dic[self.sn_name[i]]['salt2.Color.err']
            X1[i]=dic[self.sn_name[i]]['salt2.X1']
            X1_err[i]=dic[self.sn_name[i]]['salt2.X1.err']




        P.figure(figsize=(12,8))
        P.subplot(1,2,1)
        subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        Rho=N.corrcoef(self.Av,X1)[0,1]
        P.scatter(self.Av,X1,label=r'$\rho=%.2f$'%(Rho))
        P.errorbar(self.Av,X1,linestyle='', yerr=X1_err,xerr=None,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.ylabel('SALT2 X1',fontsize=16)
        P.xlabel('$A_{\lambda_0}$',fontsize=16)
        P.legend(loc=4)
        P.subplot(1,2,2)
        Rho=N.corrcoef(self.Av,Color)[0,1]
        P.scatter(self.Av,Color,label=r'$\rho=%.2f$'%(Rho))
        P.errorbar(self.Av,Color,linestyle='', yerr=Color_err,xerr=None,ecolor='grey',alpha=0.9,marker='.',zorder=0)
        P.ylabel('SALT2 Color',fontsize=16)
        P.xlabel('$A_{\lambda_0}$',fontsize=16)
        P.legend(loc=4)

        #Corr=N.corrcoef(self.Av,Color)
        #self.eigval,self.eigvec=N.linalg.eig(Corr)
        #data=passage_error(N.array([self.Av,Color]).T,N.array([N.ones(len(self.Av)),Color_err]).T,self.eigvec,sub_space=2)
        #P.figure(figsize=(16,12))
        #for i in range(3):
        #    P.subplot(2,3,i+1)
        #    P.scatter(self.data[:,i],data[:,0])
        #for i in range(3):
        #    P.subplot(2,3,i+4)
        #    P.scatter(self.data[:,i],data[:,1])


    def plot_Av_distri(self):

        P.figure()
        #subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        P.hist(self.Av,bins=25,color='r')
        P.ylabel('number of supernovae',fontsize=16)
        P.xlabel('$A_{\lambda_0}$',fontsize=16)
        P.ylim(0,20)
        P.legend(loc=4)

    def subplot_mass_step(self,HOST_pkl,LOCAL_HOST_pkl,Meta_pkl):

        P.figure(figsize=(16,12))
        subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.98,wspace = 0.01)
        P.subplot(1,3,1)
        self.plot_mass_step(HOST_pkl,LOCAL_HOST_pkl,Meta_pkl,Not_Fig=True)
        P.ylabel('$A_{\lambda_0}$',fontsize=16)
        P.ylim(-0.4,1.1)
        P.subplot(1,3,2)
        self.plot_mass_step(HOST_pkl,LOCAL_HOST_pkl,Meta_pkl,LOCAL=1,Not_Fig=True)
        P.yticks([-0.5,10],['',''])
        P.ylim(-0.4,1.1)
        P.subplot(1,3,3)
        self.plot_mass_step(HOST_pkl,LOCAL_HOST_pkl,Meta_pkl,LOCAL=2,Not_Fig=True)
        P.yticks([-0.5,10],['',''])
        P.ylim(-0.4,1.1)

    def plot_mass_step(self,HOST_pkl,LOCAL_HOST_pkl,Meta_pkl,dic_NaID=None,LOCAL=None,Not_Fig=False):

        HOST=cPickle.load(open(HOST_pkl))
        META=cPickle.load(open(Meta_pkl))
        LOCAL_HOST=cPickle.load(open(LOCAL_HOST_pkl))
        data= self.Av
        error= N.ones(len(self.Av))
        sn_name=self.sn_name

        Masse=N.zeros(len(data))
        Masse_minus=N.zeros(len(data))
        Masse_plus=N.zeros(len(data))
        Filtre_M=N.array([True]*len(data))


        Local_Masse=N.zeros(len(data))
        Local_Masse_minus=N.zeros(len(data))
        Local_Masse_plus=N.zeros(len(data))

        H_alpha=N.zeros(len(data))
        H_alpha_minus=N.zeros(len(data))
        H_alpha_plus=N.zeros(len(data))

        Filtre_H=N.array([True]*len(data))
        if dic_NaID is not None:
            NAID=cPickle.load(open(dic_NaID))
        Filtre_NaID=N.array([False]*len(data))

        for i in range(len(sn_name)):
            sn=sn_name[i]
            if dic_NaID is not None:
                Filtre_NaID[i]=NAID[sn]

            if sn in HOST.keys() :
                Masse[i]=HOST[sn]['mchost.mass']
                Masse_minus[i]=HOST[sn]['mchost.mass_m.err']
                Masse_plus[i]=HOST[sn]['mchost.mass_p.err']

                if HOST[sn]['mchost.mass']==0:
                    Filtre_M[i]=False
            else :
                Filtre_M[i]=False
            if sn in LOCAL_HOST.keys():
                Pc=3.086*10**18

                H_alpha[i]=LOCAL_HOST[sn]['HA']*4*N.pi*(d_l(META[sn]['host.zcmb'],SNLS=True)*Pc)**2
                H_alpha_minus[i]=LOCAL_HOST[sn]['HA.err']*4*N.pi*(d_l(META[sn]['host.zcmb'],SNLS=True)*Pc)**2
                H_alpha_plus[i]=LOCAL_HOST[sn]['HA.err']*4*N.pi*(d_l(META[sn]['host.zcmb'],SNLS=True)*Pc)**2

                if LOCAL==2:
                    if 'localmass' in LOCAL_HOST[sn].keys():
                        Local_Masse[i]=LOCAL_HOST[sn]['localmass']
                        Local_Masse_minus[i]=LOCAL_HOST[sn]['localmass.err']
                        Local_Masse_plus[i]=LOCAL_HOST[sn]['localmass.err']
                    else:
                        print sn
                        Filtre_H[i]=False
                    if not N.isfinite(Local_Masse[i]):
                        Filtre_H[i]=False

                if 0.023>META[sn]['host.zcmb'] or META[sn]['host.zcmb']>0.08:
                    Filtre_H[i]=False
            else:
               Filtre_H[i]=False

        if LOCAL==0:
            cst_sfr=1.
        else:
            cst_sfr=5.5e-42


        if LOCAL==2:
            Local_mass=Local_Masse[Filtre_H]
            Local_mass_err=Local_Masse_minus[Filtre_H]
        else:
            Local_mass=0
            Local_mass_err=0

        H_copy=copy.deepcopy(H_alpha)

        H_alpha_minus[Filtre_H]=N.log10(H_alpha[Filtre_H]*cst_sfr)-N.log10(H_copy[Filtre_H]*cst_sfr-H_alpha_minus[Filtre_H]*cst_sfr)+Local_mass_err
        H_alpha_plus[Filtre_H]=-N.log10(H_alpha[Filtre_H]*cst_sfr)+N.log10(H_copy[Filtre_H]*cst_sfr+H_alpha_plus[Filtre_H]*cst_sfr)+Local_mass_err
        H_alpha[Filtre_H]=N.log10(H_alpha[Filtre_H]*cst_sfr)-Local_mass

        H_alpha_minus[~N.isfinite(H_alpha_minus)]=999.

        if LOCAL==2:
            SFR=H_alpha[Filtre_H]+Local_mass
            SFR[(SFR<-6)]=-6
            H_alpha_minus[Filtre_H][(SFR<-6)]=999.
            H_alpha[Filtre_H]=SFR-Local_mass

        if LOCAL is not None:
            if LOCAL==2:
                cst=0.5
            else:
                cst=1
            H_wRMS=H.comp_rms(H_alpha[Filtre_H], 10, err=False, variance=H_alpha_minus[Filtre_H]**2)
            XH_mean=N.median(H_alpha[Filtre_H])
            XH_min=XH_mean-cst*H_wRMS
            XH_max=XH_mean+cst*H_wRMS
            Masse=H_alpha
            Masse_minus=H_alpha_minus
            Masse_plus=H_alpha_plus
            Filtre_M=Filtre_H
            Filtre_NaID=((Filtre_NaID)&Filtre_M)
            
        label_local=[r'$\log\left(\Sigma_{H_{\alpha}} [\mathrm{\mathsf{erg}} \ \mathrm{\mathsf{s}}^{-1} \ \mathrm{\mathsf{kpc}}^{-2}]  \right)$',
                     r'$\log\left(\mathrm{\mathsf{SFR}} [\mathrm{\mathsf{M}}_{\odot} \ \mathrm{\mathsf{yr}}^{-1} \ \mathrm{\mathsf{kpc}}^{-2}]  \right)$',
                     r'$\log\left(\mathrm{\mathsf{sSFR}} [\mathrm{\mathsf{yr}}^{-1}]  \right)$']

        if not Not_Fig:
            P.figure()


        Rho=N.corrcoef(Masse[Filtre_M],data[Filtre_M])[0,1]
        SIG=Statistics.correlation_significance(Rho,len(data[Filtre_M]),sigma=True)
        P.scatter(Masse[Filtre_M],data[Filtre_M],c='b',label=r'$\rho=%.2f$, $\sigma=%.2f$'%((Rho,SIG)))
        P.errorbar(Masse[Filtre_M],data[Filtre_M],linestyle='', yerr=None,xerr=[Masse_minus[Filtre_M],Masse_plus[Filtre_M]],ecolor='grey',alpha=0.9,marker='.',zorder=0)
        if N.sum(Filtre_NaID)!=0:
            P.scatter(Masse[Filtre_NaID],data[Filtre_NaID],c='r',s=50)
        self.Filtre_NaID=Filtre_NaID

        P.ylim(min(data)-N.std(data),max(data)+2*N.std(data))
        if LOCAL is None:
            P.xlabel('$\log(M/M_{\odot})$',fontsize=16)
            P.xlim(7,12)
        else:
            P.xlabel(label_local[LOCAL],fontsize=16)
            P.xlim(XH_min,XH_max)
        P.legend(loc=2)



    def plot_projection_ortho(self):

        error=N.zeros(N.shape(self.data))
        for i in range(len(self.data[:,0])):
            error[i]=N.sqrt(N.diag(self.Cov_error[i]))

        P.figure(figsize=(14,6))
        P.subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        for i in range(3):
            P.subplot(1,3,i+1)
            corr=N.corrcoef(self.data[:,i],self.xplus[:,i])[0,1]
            P.scatter(self.data[:,i],self.xplus[:,i],label=r'$\rho=%.2f$'%(corr))
            P.errorbar(self.data[:,i],self.xplus[:,i],linestyle='',xerr=error[:,i],yerr=None,ecolor='blue',alpha=0.7,marker='.',zorder=0)
            P.xlabel('$q_{%i}$'%(i+1),fontsize=16)
            P.ylabel('$h_{%i}$'%(i+1),fontsize=16)
            P.legend(loc=4)
            

    def plot_spectrum_corrected(self,No_corrected=True):

        Mag_all_sn=copy.deepcopy(self.Mag_no_corrected)

        for Bin in range(len(self.X)):
            if self.Rv>0:
                Mag_all_sn[:,Bin]-=(dot(self.alpha[Bin],self.data.T))+self.trans+(self.Av*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
            else:
                Mag_all_sn[:,Bin]-=(dot(self.alpha[Bin],self.data.T))+self.trans
        
        self.MAG_CORRECTED= Mag_all_sn
        figure(figsize=(12,12))
        subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.98,hspace=0.001)
        for sn in range(len(self.sn_name)):
            subplot(2,1,1)
            if No_corrected:
                if sn==0:
                    plot(self.X,self.Mag_no_corrected[sn]+16.2,'r',alpha=0.5,label='Observed spectra')
                else:
                    plot(self.X,self.Mag_no_corrected[sn]+16.2,'r',alpha=0.5)

            if sn==0:
                plot(self.X,Mag_all_sn[sn]+20.5,'b',alpha=0.5,label='Corrected spectra ($q_1$, $q_2$, $q_3$, $A_{\lambda_0}$)')
            else:
                plot(self.X,Mag_all_sn[sn]+20.5,'b',alpha=0.5)

        #title('%i eigenvector corrected'%(len(self.alpha[0])))
        #title('spectrum corrected')

        ylabel('Mag AB + cst')
        ylim(-5,7)
        xticks([2500.,9500.],['toto','pouet'])
        xlim(self.X[0]-60,self.X[-1]+60)
        legend(loc=4)
        gca().invert_yaxis()
        STD=N.std(Mag_all_sn,axis=0)
        STD_no_correct=N.std(self.Mag_no_corrected,axis=0)
        subplot(2,1,2)
        if No_corrected:
            plot(self.X,STD_no_correct,'r',label=r'Observed STD average floor $[6360\AA,6600\AA]$ = %.2f mag' %(mean(STD_no_correct[self.floor_filter])))

        plot(self.X,STD,'b',label=r'Corrected STD average floor $[6360\AA,6600\AA]$ = %.2f mag' %(mean(STD[self.floor_filter])))

        plot(self.X,zeros(len(self.X)),'k')
        ylabel('STD')
        xlabel('wavelength [$\AA$]')
        xlim(self.X[0]-60,self.X[-1]+60)
        ylim(0.0,0.62)
        legend()

    def look_pca_residuals(self,i_th_eigenvector):
        
        Lam=self.dico['Lambda'][:,i_th_eigenvector]
        
        figure()
        plot(self.X,Lam)
        
    def spectra_corrected_pca_residuals(self,i_th_eigenvector,Name_fig=None):

        Mag_all_sn=copy.deepcopy(self.Mag_no_corrected)
        
        Lam=self.dico['Lambda']
        Z=self.dico['Z']
        STD=N.zeros(len(self.X))
        
        for Bin in range(len(self.X)):
            if self.Rv>0:
                Mag_all_sn[:,Bin]-=(dot(self.alpha[Bin],self.data.T))+self.trans+(self.Av*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
            else:
                Mag_all_sn[:,Bin]-=(dot(self.alpha[Bin],self.data.T))+self.trans
                
            for corr in range(i_th_eigenvector):
                Mag_all_sn[:,Bin]-=Z[:,corr-1]*Lam[Bin,corr-1]

            STD[Bin]=N.std(Mag_all_sn[:,Bin])
            
        figure(figsize=(9,9))
        subplots_adjust(hspace=0.001)
        for sn in range(len(self.sn_name)):
            subplot(2,1,1)
            plot(self.X,Mag_all_sn[sn]+19.2)

        title('spectrum corrected')

        ylabel('all spectrum (Mag AB + cst)')
        ylim(-2.8,5.2)
        xticks([2500.,9500.],['toto','pouet'])
        xlim(self.X[0]-60,self.X[-1]+60)
        gca().invert_yaxis()

        subplot(2,1,2)
        plot(self.X,STD,'b',label=r' STD mean floor $[6360\AA,6600\AA]$ = %.2f mag' %(mean(STD[self.floor_filter])))
        plot(self.X,zeros(len(self.X)),'k')
        ylabel('STD')
        xlabel('wavelength [$\AA$]')
        xlim(self.X[0]-60,self.X[-1]+60)
        ylim(0.0,0.35)
        legend()
        if Name_fig!=None:
            savefig(Name_fig[0])
        

        figure(figsize=(8,10))
        correction=i_th_eigenvector
        gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
        subplots_adjust(hspace=0.001)
            
            #ax1=subplot(211)                                                                                                                               
        subplot(gs[0])
        MIN=[-0.21,-0.21,-0.21]
        MAX=[0.24,0.24,0.24]
            
        y_moins=zeros(len(self.X))
        y_plus=zeros(len(self.X))

        for Bin in range(len(self.X)):
            y_moins[Bin]=self.M0[Bin]+19.2-Lam[Bin,correction-1]*(mean(Z[:,correction-1])+sqrt(var(Z[:,correction-1])))
            y_plus[Bin]=self.M0[Bin]+19.2+Lam[Bin,correction-1]*(mean(Z[:,correction-1])+sqrt(var(Z[:,correction-1])))

        fill_between(self.X,self.M0+19.2,y_plus,color='m',alpha=0.7 )
        fill_between(self.X,y_moins,self.M0+19.2,color='g',alpha=0.7)
        p3 = plot(self.X,self.M0+19.2,'b',label='toto')
        ylim(-0.3,1.5)
        gca().invert_yaxis()
        ylabel('$M_0(t=0,\lambda) + cst$')
        title('Mean spectrum with $\pm$1$\sigma$ variation (eigenvector %i)'%(correction))
        p1 = Rectangle((0, 0), 1, 1, fc="magenta")
        p2 = Rectangle((0, 0), 1, 1, fc="green")
        legend([p1, p2], ['+1$\sigma$', '-1$\sigma$'])
        xticks([2500.,9500.],['toto','pouet'])
        xlim(self.X[0]-60,self.X[-1]+60)

            #ax2=subplot(212,sharex=ax1)                                                                                                                    
        subplot(gs[1])

#            mean_effect=(1./len(self.X))*sqrt(N.sum(self.alpha[:,correction]*self.alpha[:,correction]))                                                    
        mean_effect=std(Z[:,correction-1])*mean(abs(Lam[:,correction-1]))
         
        plot(self.X,Lam[:,correction],'b',label=r'Mean effect (mag)=%.3f'%((mean_effect)))
        ylabel(r'$\Lambda_{%i}(\lambda)$'%(correction))
        xlabel('wavelength [$\AA$]')
        #ylim(MIN[correction],MAX[correction])
        xlim(self.X[0]-60,self.X[-1]+60)
        if correction==2:
            legend()
        else:
            legend(loc=4)

        if Name_fig is not None:
            savefig(Name_fig[1])



        #figure()

        #DATA=zeros((len(self.sn_name),len(self.data[0])+i_th_eigenvector))
        #COVX=zeros((len(self.sn_name),len(self.data[0])+i_th_eigenvector,len(self.data[0])+i_th_eigenvector))

        #DATA[:,i_th_eigenvector:]=self.data
        #DATA[:,:i_th_eigenvector]=self.Z
        #COVX[:,i_th_eigenvector:,i_th_eigenvector:]=self.Cov_error
        #for sn in range(len(self.sn_name)):
        #    diag(CovX[sn,:i_th_eigenvector,:i_th_eigenvector])=1
            
        #GF=global_fit(self.Mag_no_corrected,self.Y_err,self.X,data=DATA,CovX=COVX,
        #              dm_z=???,alpha0=???,reddening_law=???,M00=self.M0,B_V=self.Av,H0=self.xplus,Delta_M0=self.trans,Disp_matrix_Init=None,
        #              Color=True,delta_M_grey=True,CCM=True)
        #GF.Compute_traditional_chi2(disp_added=None)
        #disp_matrix=measured_dispersion_matrix()







    def plot_spectral_variability(self,name_fig=None,ERROR=False):
       
        c=['r','b','g','k']
        NUMBER=['first','second','third']

        for correction in range(len(self.alpha[0])):

            figure(figsize=(12,10))
            gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
            subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.95,hspace=0.001)

            #ax1=subplot(211)
            subplot(gs[0])
            #MIN=[-0.21,-0.21,-0.21]
            #MAX=[0.24,0.24,0.24]

            y_moins=zeros(len(self.X))
            y_plus=zeros(len(self.X))
            
            for Bin in range(len(self.X)):
                y_moins[Bin]=self.M0[Bin]+19.2-self.alpha[Bin,correction]*(mean(self.data[:,correction])+sqrt(var(self.data[:,correction])))
                y_plus[Bin]=self.M0[Bin]+19.2+self.alpha[Bin,correction]*(mean(self.data[:,correction])+sqrt(var(self.data[:,correction])))

            fill_between(self.X,self.M0+19.2,y_plus,color='m',alpha=0.7 )
            fill_between(self.X,y_moins,self.M0+19.2,color='g',alpha=0.7)
            p3 = plot(self.X,self.M0+19.2,'b',label='toto')
            ylim(-0.5,1.3)
            gca().invert_yaxis()
            ylabel('$M_0(t=0,\lambda) + cst$')
            title(r'Average spectrum with $\pm$1$\sigma$ variation ($\alpha_{%i}(t=0,\lambda)$)'%(correction+1))
            p1 = Rectangle((0, 0), 1, 1, fc="magenta")
            p2 = Rectangle((0, 0), 1, 1, fc="green")
            legend([p1, p2], ['+1$\sigma$', '-1$\sigma$'])
            xticks([2500.,9500.],['toto','pouet'])
            xlim(self.X[0]-60,self.X[-1]+60)

            #ax2=subplot(212,sharex=ax1)
            subplot(gs[1])
            if ERROR:
                fill_between(self.X,self.alpha[:,correction]-self.alpha_err_Jackknife[:,correction],self.alpha[:,correction]+self.alpha_err_Jackknife[:,correction],color='b',alpha=0.7 )

#            mean_effect=(1./len(self.X))*sqrt(N.sum(self.alpha[:,correction]*self.alpha[:,correction]))
            mean_effect=std(self.data[:,correction])*mean(abs(self.alpha[:,correction]))
            plot(self.X,self.alpha[:,correction],'b',label=r'Average effect (mag)=%.3f'%((mean_effect)))
            ylabel(r'$\alpha_{%i}(t=0,\lambda)$'%(correction+1))
            xlabel('wavelength [$\AA$]')
            #ylim(MIN[correction],MAX[correction])
            xlim(self.X[0]-60,self.X[-1]+60)
            #if correction==0:
            legend(loc=4)
            #else:
            #    legend()
            #xticklabels = ax2.get_xticklabels()
            #setp(xticklabels, visible=True)
            if name_fig!=None:
                savefig('../These_plot/plot_septembre_2015/'+name_fig[correction])
           #legend()

    
    def plot_corr_matrix(self):

        plot_matrix(self.X,self.corr_matrix,r'$\rho$','correlation matrix',cmap=matplotlib.cm.jet,plotpoints=True)
    
           

    def plot_sn_residual(self,sn_name):
        
        
        for sn in range(len(self.sn_name)):
            if self.sn_name[sn]==sn_name:
                figure()
                y_moins= self.Y_build[sn]-self.Mag_no_corrected[sn]-sqrt(self.Y_build_error[sn])
                y_plus=self.Y_build[sn]-self.Mag_no_corrected[sn]+sqrt(self.Y_build_error[sn])
                fill_between(self.X,y_moins,y_plus,color='b',alpha=0.3)
                
                plot(self.X,self.Y_build[sn]-self.Mag_no_corrected[sn],'b',label='residual')
                plot(self.X,zeros(len(self.X)),'k')
                gca().invert_yaxis()
                ylabel('$\Delta \mu(\lambda)$')
                xlabel('wavelength [$\AA$]')
                ylim(-0.4,0.4)
                title(self.sn_name[sn])#+' '+self.key)
                legend()


        

    def plot_sn_rebuild(self,sn_name,derrening=False):  


        for sn in range(len(self.sn_name)):
            if self.sn_name[sn]==sn_name:
                
                if derrening:
                    
                    for Bin in range(len(self.X)):
                        self.Mag_no_corrected[sn,Bin]-=(self.Av[sn]*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
                        self.Y_build[sn,Bin]-=(self.Av[sn]*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
                
 

                y_moins= self.Y_build[sn]+19.2-sqrt(self.Y_build_error[sn])
                y_plus=self.Y_build[sn]+19.2+sqrt(self.Y_build_error[sn])
                fill_between(self.X,y_moins,y_plus,color='b',alpha=0.3)
                fill_between(self.X,self.Mag_no_corrected[sn]-self.Y_err[sn]+19.2,self.Mag_no_corrected[sn]+self.Y_err[sn]+19.2,color='r',alpha=0.3)
                plot(self.X,self.Y_build[sn]+19.2,'b',label='reconstructed')
                plot(self.X,self.Mag_no_corrected[sn]+19.2,'r',label='real')
                #ylim(-1,4.8)
                gca().invert_yaxis()
                ylabel('Mag AB + cst')
                xlabel('wavelength [$\AA$]')
                title(self.sn_name[sn])#+' '+self.key)
                legend()



                        
    def plot_all_sn_rebuild(self,Save=False,TeX=False,Mag_lim=[-2.5,2.5]):     

        for sn in range(len(self.sn_name)):
            chi2=dot(matrix(self.Mag_no_corrected[sn]-self.Y_build[sn]),dot(inv(diag(self.Y_err[sn]**2+self.Y_build_error[sn])),matrix(self.Mag_no_corrected[sn]-self.Y_build[sn]).T))
            #DeltaM=average(self.Mag_no_corrected[sn]-self.Y_ODR_build[sn],weights=1./(self.Y_err[sn]**2+self.Y_ODR_build_error[sn]))
            figure()
            y_moins= self.Y_build[sn]+19.2-sqrt(self.Y_build_error[sn])
            y_plus=self.Y_build[sn]+19.2+sqrt(self.Y_build_error[sn])
            fill_between(self.X,y_moins,y_plus,color='b',alpha=0.3)
            fill_between(self.X,self.Mag_no_corrected[sn]-self.Y_err[sn]+19.2,self.Mag_no_corrected[sn]+self.Y_err[sn]+19.2,color='r',alpha=0.3)
            plot(self.X,self.Y_build[sn]+19.2,'b',label=r'reconstructed   $\chi^2$/dof=%f'%(chi2/len(self.X)))
            plot(self.X,self.Mag_no_corrected[sn]+19.2,'r',label='real')#   $\Delta M$=%f'%(DeltaM))
            #ylim(Mag_lim[0],Mag_lim[1])
            gca().invert_yaxis()
            ylabel('Mag AB + cst')
            xlabel('wavelength [$\AA$]')
            title(self.sn_name[sn]+' %i eigenvector corrected'%(len(self.alpha[0])))
            legend()
            
    def plot_bin_Av_slope(self,Bin):

        if type(Bin) == list:
            for bb in Bin:
                print bb
                self.plot_bin_Av_slope(bb)
                
            return
                

        CCM=zeros(len(self.X))
        slopes_star=zeros(len(self.X))
        slopes_star_err=zeros(len(self.X))

        Mag_all_sn=self.Mag_no_corrected
        Mag_all_sn_err=self.Y_err
        
       
        slopes=self.slope
        Av=self.Av
        M0=self.M0
        alpha=self.alpha
        data=self.data
        trans=self.trans
        reddening_law_err_Jackknife=self.reddening_law_err_Jackknife
        
        toto= 100.
        BIN=0
    
        #for X_Bin in range(len(self.X)):
        #    if self.Rv==0:
        #        self.Rv=3.1
        #    CCM[X_Bin]=Astro.Extinction.extinctionLaw(self.X[X_Bin],Rv=self.Rv)
        #    if abs(1.-CCM[X_Bin])<toto:
        #        toto=abs(1.-CCM[X_Bin])
        #        BIN=X_Bin
        
        CCM31=Astro.Extinction.extinctionLaw(self.X,Rv=3.1,law='CCM89')
        CCM26=Astro.Extinction.extinctionLaw(self.X,Rv=2.6,law='CCM89')
        CCM14=Astro.Extinction.extinctionLaw(self.X,Rv=1.4,law='CCM89')
        Ind_med=(len(self.X)/2)-1
        CCM31/=CCM31[Ind_med]
        CCM26/=CCM26[Ind_med]
        CCM14/=CCM14[Ind_med]
        AVV=N.linspace(-0.5,1,20)       
        #for X_Bin in range(len(self.X)):
        #    slopes_star[X_Bin]=(slopes[X_Bin]/slopes[BIN])
        

        figure(figsize=(11,11))
        subplots_adjust(left=0.12, bottom=0.07, right=0.99, top=0.99)
        MAG=copy.deepcopy(Mag_all_sn)
        
        for sn in range(len(MAG[:,0])):
            for X_Bin in range(len(self.X)):
                for correction in range(len(self.alpha[0])):
                    MAG[sn,X_Bin]-=alpha[X_Bin,correction]*data[sn,correction]

                MAG[sn,X_Bin]-=trans[sn]+M0[X_Bin]

     
        self.MAG=MAG

        subplot(2,1,1)

        for i,sn in enumerate(self.sn_name):
            if sn =='SN2007le':
                scatter(Av[i],MAG[i,Bin],zorder=100,c='r',marker='^',s=100,label='SN2007le')
            if sn =='SN2005cf':
                scatter(Av[i],MAG[i,Bin],zorder=100,c='r',marker='o',s=100,label='SN2005cf')
  

        errorbar(Av,MAG[:,Bin],linestyle='',xerr=None,yerr=sqrt(self.Y_build_error[:,Bin]),ecolor='blue',alpha=0.7,marker='.',zorder=0)
        scat=scatter(Av,MAG[:,Bin],zorder=100,s=50,c='b')
            
        plot(AVV,slopes[Bin]*AVV,'r',label='$\gamma_{%i\AA}$'%(self.X[Bin]))
        plot(AVV,CCM31[Bin]*AVV,'k--',linewidth=3,label='CCM $(R_V=3.1)$')
        plot(AVV,CCM26[Bin]*AVV,'r--',linewidth=3,label='CCM $(R_V=2.6)$')
        plot(AVV,CCM14[Bin]*AVV,'k-.',linewidth=3,label='CCM $(R_V=1.4)$')
        ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) - \sum_{i=1}^{i=3} \\alpha_i(0,\lambda) q_i$',fontsize=14)
        xlabel('$A_{\lambda_0}$',fontsize=16)
        title(r'$\lambda=%i \AA$'%(self.X[Bin]))
        ylim(min(MAG[:,Bin])-0.3,max(MAG[:,Bin])+0.3)
        #gca().invert_yaxis()
        legend(loc=4)

        subplot(2,1,2)
        scatter(self.X[Bin],slopes[Bin],c='r',marker='o',s=100)
        #fill_between(self.X,slopes_star-slopes_star_err,slopes_star+slopes_star_err,color='r',alpha=0.3)
        plot(self.X,slopes,'r',label= '$\gamma_{\lambda}$')
        plot(self.X,CCM31,'k-.',linewidth=3,label= 'CCM $(R_V=3.1)$')
        plot(self.X,CCM26,'r--',linewidth=3,label= 'CCM $(R_V=2.6)$')
        plot(self.X,CCM14,'k--',linewidth=3,label= 'CCM $(R_V=1.4)$')
        ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$',fontsize=16)
        xlabel('wavelength [$\AA$]')        
        #yticks([-1.,11.],['toto','pouet'])
        #ylim(0.4,2.4)
        ylim(0.4,2.1)
        xlim(self.X[0]-60,self.X[-1]+60)
        legend()
        #browser=MPL.PointBrowser(Av, Mag_all_sn[:,Bin],self.sn_name,scat)
        #show()
        

            


    def plot_bin(self,Bin):

        Mag_all_sn=self.Mag_no_corrected
        Mag_all_sn_err=self.Y_err
        new_err=zeros(shape(self.data))
        for sn in range(len(self.sn_name)):
            if self.Cov_error!= None:
                new_err[sn]=sqrt(diag(self.Cov_error[sn]))
            else:
                new_err[sn]=self.err[sn]

        for i in range(self.number_correction):

            figure()
            errorbar(self.data[:,i],Mag_all_sn[:,Bin]+19.2,linestyle='',xerr=new_err[:,i],yerr=sqrt(Mag_all_sn_err[:,Bin]**2+self.disp_matrix[Bin,Bin]),ecolor='grey',alpha=0.4,marker='.',zorder=0)
            scat=scatter(self.data[:,i],Mag_all_sn[:,Bin]+19.2,zorder=100,c='b')#,label='real')
            
            plot(self.data[:,i],self.alpha[Bin,i]*self.data[:,i]+self.M0[Bin]+19.2,label='%f * x%i + %f'%((self.alpha[Bin,i],i+1,self.M0[Bin]+19.2)))
            ylabel('Mag AB + cst')
            xlabel('projection on vec %i'%(i+1))
            title('bin %i vec %i'%((Bin,i+1)))
            ylim(min(Mag_all_sn[:,Bin])-1+19.2,max(Mag_all_sn[:,Bin])+1+19.2)
            gca().invert_yaxis()
            legend()
            browser=MPL.PointBrowser(self.data[:,i], Mag_all_sn[:,Bin],self.sn_name,scat)
            show()



    def plot_bin_debug(self,Bin):

        Mag_all_sn=self.Mag_corrected
        Mag_all_sn_err=self.Y_build_error
        new_err=zeros(shape(self.data))
        for sn in range(len(self.sn_name)):
            if self.Cov_error!= None:
                new_err[sn]=sqrt(diag(self.Cov_error[sn]))
            else:
                new_err[sn]=self.err[sn]

        for i in range(self.number_correction):

            figure()
            errorbar(self.data[:,i],Mag_all_sn[:,Bin]+19.2,linestyle='',xerr=new_err[:,i],yerr=sqrt(Mag_all_sn_err[:,Bin]**2+self.disp_matrix[Bin,Bin]),ecolor='grey',alpha=0.4,marker='.',zorder=0)
            scat=scatter(self.data[:,i],Mag_all_sn[:,Bin]+19.2,zorder=100,c='b')#,label='real')
            
            plot(self.data[:,i],ones(len(self.data[:,i]))*self.M0[Bin]+19.2)
            ylabel('Mag AB + cst')
            xlabel('projection on vec %i'%(i+1))
            title('bin %i vec %i'%((Bin,i+1)))
            ylim(min(Mag_all_sn[:,Bin])-1+19.2,max(Mag_all_sn[:,Bin])+1+19.2)
            gca().invert_yaxis()
            legend()
            browser=MPL.PointBrowser(self.data[:,i], Mag_all_sn[:,Bin],self.sn_name,scat)
            show()

 

    def plot_color_law(self,Error=False):

        
        slopes=self.slope
        slopes_err=self.reddening_law_err_Jackknife
        figure(figsize=(7,3))
        subplots_adjust(bottom=0.20)
        CCM=zeros(len(self.X))
        CCM_star=zeros(len(self.X))
        slopes_star=zeros(len(self.X))
        slopes_star_err=zeros(len(self.X))
        toto= 10000.
        BIN=0
    
        #for Bin in range(len(self.X)):
        #    if self.Rv==0:
        #        self.Rv=3.1
        #    CCM[Bin]=Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv)
        #    if abs(5550.-self.X[Bin])<toto:
        #        toto=abs(5550-self.X[Bin])
        #        BIN=Bin
      
        CCM=Astro.Extinction.extinctionLaw(self.X,law='CCM89',Rv=self.Rv)
        Ind_med=(len(self.X)/2)-1
        CCM/=CCM[Ind_med]
        
        for Bin in range(len(self.X)):
            slopes_star[Bin]=(slopes[Bin]/slopes[BIN])
            if Error:
                slopes_star_err[Bin]=(slopes_err[Bin]/slopes[BIN])
            CCM_star[Bin]=(CCM[Bin]/CCM[BIN])
            
        if Error:
            fill_between(self.X,slopes_star-slopes_star_err,slopes_star+slopes_star_err,color='r',alpha=0.3)
        plot(self.X,slopes,'r',label='$\gamma_{\lambda}$')
        plot(self.X,CCM,'r--',label='CCM law ($R_V=%.2f$)'%(self.Rv),linewidth=4)
        ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
        xlabel('wavelength [$\AA$]')
        title('extinction law')
        
        yticks([-1.,11.],['toto','pouet'])
        ylim(0.1,2.2)
        xlim(self.X[0]-60,self.X[-1]+60)
        legend()


class SUGAR_vs_agno(SUGAR_plot):

    def __init__(self,dico_hubble_fit,dico_agno):
        
        SUGAR_plot.__init__(self,dico_hubble_fit)

        dic=cPickle.load(open(dico_agno))

        intrinsic=dic['intrinsic']

        self.Rv_agno=dic['Rv']
        self.Av_cardel_agno=dic['Av_cardel']
        self.Av_agno=dic['Av']
        self.disp_added_agno=dic['disp_added']
        self.disp_matrix_agno=dic['disp_matrix']
        self.slopes_agno=dic['slopes']
        self.slopes_err_agno=dic['slopes_err']
        self.residuals_agno=dic['residuals']
        self.COVY_agno=dic['COVY']
        self.Y_agno=dic['Y']
        self.Y_err_agno=dic['Y_err']
        self.X_agno=dic['X']
        self.data_agno=dic['data']
        self.err_agno=dic['err']
        self.sn_name_agno=dic['sn_name']
        if intrinsic:
            self.alpha_agno=dic['alpha']
            self.M0_agno=dic['M0']

    def compare_slopes(self):
        
        figure()
        toto= 10000.
        BIN=0

        plot(self.X,self.slope,'r',label= 'Global fit')
        plot(self.X,self.slopes_agno,'r',label='agnostic _fit',linewidth=2)
        ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
        xlabel('wavelength [$\AA$]')
        title('extinction law')
        
        #yticks([-1.,11.],['toto','pouet'])
        xlim(self.X[0]-60,self.X[-1]+60)
        ylim(0,2.5)
        legend()


    def compare_CCM(self):
        
        if len(self.X)%2==1:
            Ind_med=list(self.X).index(N.median(self.X))
        else:
            Ind_med=(len(self.X)/2)-1       
        
        
        CCM_global=Astro.Extinction.extinctionLaw(self.X,Rv=self.Rv)
        CCM_agno=Astro.Extinction.extinctionLaw(self.X,Rv=self.Rv_agno)
        
        CCM_global/=CCM_global[Ind_med]
        CCM_agno/=CCM_agno[Ind_med]
        
        figure()
        toto= 10000.
        BIN=0

        plot(self.X,CCM_global,'r',label= 'CCM Global fit')
        plot(self.X,CCM_agno,'r',label='CCM agnostic fit',linewidth=2)
        ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
        xlabel('wavelength [$\AA$]')
        title('extinction law')
        
        #yticks([-1.,11.],['toto','pouet'])
        xlim(self.X[0]-60,self.X[-1]+60)
        ylim(0,2.5)
        legend()

       
    def compare_Av(self):

        figure()
        toto= 10000.
        BIN=0

        plot(self.Av,'r',label= 'Av Global fit')
        plot(self.Av_agno,'r',label='Av agnostic fit',linewidth=2)
        ylabel(r'$A_V$')
        xlabel('supernovae')
        legend()


    def compare_Av_cardell(self):
        
        figure()
        toto= 10000.
        BIN=0
        plot(self.Av_cardel,'r',label= 'Av Global fit')
        plot(self.Av_cardel_agno,'r',label='Av agnostic fit',linewidth=2)
        ylabel(r'$A_V$ cardelli')
        xlabel('supernovae')
        legend()


    def compare_disp_matrix(self):
        
        agno=(self.disp_added_agno**2)*(self.disp_matrix_agno)
        global_fit=self.disp_matrix
        MATRIX=(agno-global_fit)
        
        plot_matrix(self.X,MATRIX,r'difference','agno-global',cmap=matplotlib.cm.jet,plotpoints=True,VM=[None,None])
        


class SUGAR_Doctor_plot(SUGAR_plot):

    def __init__(self,dico_hubble_fit):

        SUGAR_plot.__init__(self,dico_hubble_fit)

        self.chi2_Alpha_variation=self.dico['chi2_Alpha_variation']
        self.Alpha_variation=self.dico['Alpha_variation']

        self.chi2_xplus_variation=self.dico['chi2_xplus_variation']
        self.xplus_variation=self.dico['xplus_variation']

        self.chi2_red_law_variation=self.dico['chi2_red_law_variation']
        self.red_law_variation=self.dico['red_law_variation']
            
        self.M0_variation=self.dico['M0_variation']
        self.chi2_M0_variation=self.dico['chi2_M0_variation']
            
        self.Av_variation=self.dico['Av_variation']
        self.chi2_Av_variation=self.dico['chi2_Av_variation']
    
        self.grey_variation=self.dico['grey_variation']
        self.chi2_grey_variation=self.dico['chi2_grey_variation']
    
        self.inv_Rv_variation=self.dico['inv_Rv_variation']
        self.chi2_inv_Rv_variation=self.dico['chi2_inv_Rv_variation']

        if N.sum(self.chi2_red_law_variation) == 0.:
            self.COLOR=False
            corr=len(self.alpha[0])
        else:
            self.COLOR=True
            corr=len(self.alpha[0])+1

        if N.sum(self.chi2_grey_variation) ==0. :
            self.GREY=False
        else:
            self.GREY=True

        self.residuals=zeros((len(self.Mag_no_corrected[0]),corr,len(self.Mag_no_corrected[:,0])))

        for BIN in range(len(self.X)):

            if self.COLOR:
                self.residuals[BIN,0]=copy.deepcopy(self.Mag_no_corrected[:,BIN])
                for i in range(corr-1):
                    self.residuals[BIN,0]-=self.data[:,i]*self.alpha[BIN,i]
                if self.GREY:
                    self.residuals[BIN,0]-=self.trans

                self.residuals[BIN,0]-=self.M0[BIN]
                        
                for i in range(corr-1):
                    self.residuals[BIN,i+1]=copy.deepcopy(self.Mag_no_corrected[:,BIN])
                    self.residuals[BIN,i+1]-=self.Av*self.slope[BIN]
                    for j in range(corr-1):
                        if i!=j:
                            self.residuals[BIN,i+1]-=self.data[:,j]*self.alpha[BIN,j]
                    if self.GREY:
                        self.residuals[BIN,i+1]-=self.trans

                    self.residuals[BIN,i+1]-=self.M0[BIN]

            else:                   
                for i in range(corr):
                    self.residuals[BIN,i]=copy.deepcopy(self.Mag_no_corrected[:,BIN])
                    for j in range(corr):
                        if i!=j:
                            self.residuals[BIN,i]-=self.data[:,j]*self.alpha[BIN,j]
                    if self.GREY:
                        self.residuals[Bin,i]-=self.trans

                    self.residuals[BIN,i]-=self.M0[BIN]

    def plot_interact_red_law(self):

        figure(figsize=(10,10))
        subplot(221)
        subplots_adjust(bottom=0.25)
        l1, = plot(self.Av,self.residuals[0,0], lw=2, linestyle='',marker='o',color='red')
        L1, = plot(self.Av,self.slope[0]*self.Av,'b')
        if self.GREY:
            ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) - \sum_i \\alpha_i(0,\lambda) q_i - \Delta M_{grey}$')
        else:
            ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) - \sum_i \\alpha_i(0,\lambda) q_i$')

        xlabel('$A_v$')

        subplot(223)
        l2, = plot(self.X[0],self.slope[0],color='r',marker='o',markersize=7)
        plot(self.X,self.slope,'r',label= 'extinction law')
        ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
        xlabel('wavelength [$\AA$]')  
        
        subplot(122)
        L3, = plot(self.slope[0]*ones(len(self.chi2_red_law_variation[0])),self.chi2_red_law_variation[0],lw=2,color='b')
        l3, = plot(self.red_law_variation[0],self.chi2_red_law_variation[0], lw=2,color='b')
        xlabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
        ylabel(r'$\chi^2$') 
        xlim(min(self.slope)-3.*std(self.slope),max(self.slope)+3.*std(self.slope))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)
       

        slambda = Slider(axlam, 'lambda', 0, len(self.X)-1, valinit=0)
       

        def update(val):

            Bin=slambda.val
            Bin=int(Bin)
            l1.set_ydata(self.residuals[Bin,0])
            L1.set_ydata(self.slope[Bin]*self.Av)
            l2.set_xdata(self.X[Bin])
            l2.set_ydata(self.slope[Bin])
            L3.set_xdata(self.slope[Bin]*ones(len(self.chi2_red_law_variation[Bin])))
            L3.set_ydata(self.chi2_red_law_variation[Bin])
            l3.set_xdata(self.red_law_variation[Bin])
            l3.set_ydata(self.chi2_red_law_variation[Bin])


            draw()

        slambda.on_changed(update)



        resetax = axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            #sfreq.reset()
            #samp.reset()
            slambda.reset()
            
        button.on_clicked(reset)

        show()


    def plot_interact_M0(self):

        figure(figsize=(10,12))
 
        P1=subplot(211)
        l2, = plot(self.X[0],self.M0[0],color='B',marker='o',markersize=7)
        plot(self.X,self.M0,'b',label= 'mean spectrum')
        gca().invert_yaxis()
        P1.set_ylabel(r'$M_0$',fontsize=12)
        P1.set_xlabel('wavelength [$\AA$]',fontsize=12)  
        
        P2=subplot(212)
        L3, = plot(self.M0[0]*ones(len(self.chi2_M0_variation[0])),self.chi2_M0_variation[0],lw=2,color='b')
        l3, = plot(self.M0_variation[0],self.chi2_M0_variation[0], lw=2,color='b')
        P2.set_xlabel(r'$M_0(\lambda=%i \AA)$'%(self.X[0]),fontsize=12)
        P2.set_ylabel(r'$\chi^2$',fontsize=12) 
        xlim(min(self.M0),max(self.M0))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.025, 0.8, 0.03], axisbg=axcolor)
       
        slambda = Slider(axlam, 'lambda', 0, len(self.X)-1, valinit=0)
       
        def update(val):

            Bin=slambda.val
            Bin=int(Bin)
            
            l2.set_xdata(self.X[Bin])
            l2.set_ydata(self.M0[Bin])

            L3.set_xdata(self.M0[Bin]*ones(len(self.chi2_M0_variation[Bin])))
            L3.set_ydata(self.chi2_M0_variation[Bin])
            l3.set_xdata(self.M0_variation[Bin])
            l3.set_ydata(self.chi2_M0_variation[Bin])
            P2.set_xlabel(r'$M_0(\lambda=%i \AA)$'%(self.X[Bin]),fontsize=12)
            draw()

        slambda.on_changed(update)
        show()

    def plot_interact_alpha(self):

        if self.COLOR:
            cst=1
        else:
            cst=0

            
        if self.COLOR:
            if self.GREY:
                add_label='$- A_V S(\lambda) - \Delta M_{grey}$'
            else:
                add_label='$- A_V S(\lambda)$'
        else:
            if self.GREY:
                add_label='$- \Delta M_{grey}$'

            else:
                add_label=''
    
        figure(figsize=(10,10))
        P1=subplot(221)
        subplots_adjust(bottom=0.25)
        l1, = plot(self.data[:,0],self.residuals[0,0+cst], lw=2, linestyle='',marker='o',color='blue')
        L1, = plot(self.data[:,0],self.alpha[0,0]*self.data[:,0],'b')
        P1.set_ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) -  \sum_{i\\ne%i} \\alpha_i(0,\lambda) q_i$'%(1) + add_label,fontsize=12)
        P1.set_xlabel('$x_1$',fontsize=16)
        ylim(N.min(self.residuals),N.max(self.residuals))
        xlim(N.min(self.data),N.max(self.data))

        P2=subplot(223)
        l2, = plot(self.X[0],self.alpha[0,0],color='blue',marker='o',markersize=7)
        L2, = plot(self.X,self.alpha[:,0],'b')
        ylim(N.min(self.alpha),N.max(self.alpha))
        P2.set_ylabel(r'$\alpha_1(\lambda=%i \AA)$'%(self.X[0]),fontsize=16)
        P2.set_xlabel('wavelength [$\AA$]')  
        
        P3=subplot(122)
        L3, = plot(self.alpha[0,0]*ones(len(self.chi2_Alpha_variation[0,0])),self.chi2_Alpha_variation[0,0],lw=2,color='b')
        l3, = plot(self.Alpha_variation[0,0],self.chi2_Alpha_variation[0,0], lw=2,color='b')
        P3.set_xlabel(r'$\alpha_1(\lambda=%i \AA)$'%(self.X[0]),fontsize=16)
        P3.set_ylabel(r'$\chi^2$',fontsize=16) 
        xlim(N.min(self.alpha),N.max(self.alpha))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)
        axcomp = axes([0.1, 0.05, 0.2, 0.03], axisbg=axcolor)

        slambda = Slider(axlam, 'lambda', 0, len(self.X)-1, valinit=0)
        scomp = Slider(axcomp, 'composante', 0, len(self.alpha[0])-0.5, valinit=0)

        def update(val):

            Bin=slambda.val
            comp=scomp.val
            comp=int(comp)
            Bin=int(Bin)

            l1.set_ydata(self.residuals[Bin,comp+cst])
            l1.set_xdata(self.data[:,comp])
            L1.set_ydata(self.alpha[Bin,comp]*self.data[:,comp])
            L1.set_xdata(self.data[:,comp])
            P1.set_xlabel('$x_{%i}$'%(comp+1),fontsize=16)
            P1.set_ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) -  \sum_{i\\ne%i} \\alpha_i(0,\lambda) q_i$'%(comp+1) + add_label,fontsize=12)

            l2.set_xdata(self.X[Bin])
            l2.set_ydata(self.alpha[Bin,comp])
            L2.set_ydata(self.alpha[:,comp])
            P2.set_ylabel(r'$\alpha_{%i}(\lambda=%i \AA)$'%((comp+1,self.X[Bin])),fontsize=16)

            L3.set_xdata(self.alpha[Bin,comp]*ones(len(self.chi2_Alpha_variation[Bin,comp])))
            L3.set_ydata(self.chi2_Alpha_variation[Bin,comp])
            l3.set_xdata(self.Alpha_variation[Bin,comp])
            l3.set_ydata(self.chi2_Alpha_variation[Bin,comp])
            P3.set_xlabel(r'$\alpha_{%i}(\lambda=%i \AA)$'%((comp+1,self.X[Bin])),fontsize=16)

            draw()

        slambda.on_changed(update)
        scomp.on_changed(update)


        resetax = axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            slambda.reset()
        button.on_clicked(reset)

        show()

    def plot_interact_xplus(self):
          
        figure(figsize=(10,10))
        P1=subplot(211)
        subplots_adjust(bottom=0.25)       
        l1, = plot(self.xplus[0,0],color='blue',marker='o',markersize=7)
        L1, = plot(self.xplus[:,0],'b',label='True value')
        LL1, = plot(self.data[:,0],'k',label='Measured value') 
        ylim(N.min(self.xplus),N.max(self.xplus))
        xlim(-1,len(self.sn_name))
        P1.set_ylabel(r'$x_1$',fontsize=16)
        P1.set_xlabel(self.sn_name[0],fontsize=16)  
        legend(loc=2)
        
        P2=subplot(212)
        L2, = plot(self.xplus[0,0]*ones(len(self.chi2_xplus_variation[0,0])),self.chi2_xplus_variation[0,0],lw=2,color='b')
        l2, = plot(self.xplus_variation[0,0],self.chi2_xplus_variation[0,0], lw=2,color='b')
        P2.set_xlabel(r'$x_1$ ',fontsize=16)
        P2.set_ylabel(r'$\chi^2$',fontsize=16) 
        xlim(N.min(self.xplus),N.max(self.xplus))
        ylim(N.min(self.chi2_xplus_variation),N.max(self.chi2_xplus_variation))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)
        axcomp = axes([0.1, 0.05, 0.2, 0.03], axisbg=axcolor)

        ssupernovae = Slider(axlam, 'supernova', 0, len(self.sn_name)-1, valinit=0)
        scomp = Slider(axcomp, 'composante', 0, len(self.alpha[0])-0.5, valinit=0)

        def update(val):

            SN=ssupernovae.val
            comp=scomp.val
            comp=int(comp)
            SN=int(SN)

            l1.set_ydata(self.xplus[SN,comp])
            l1.set_xdata(SN)
            L1.set_ydata(self.xplus[:,comp])
            LL1.set_ydata(self.data[:,comp])
            P1.set_xlabel(self.sn_name[SN],fontsize=16)
            P1.set_ylabel(r'$x_{%i}$'%(comp+1),fontsize=16)            

            L2.set_xdata(self.xplus[SN,comp]*ones(len(self.chi2_xplus_variation[SN,comp])))
            L2.set_ydata(self.chi2_xplus_variation[SN,comp])
            l2.set_xdata(self.xplus_variation[SN,comp])
            l2.set_ydata(self.chi2_xplus_variation[SN,comp])
            P2.set_xlabel(r'$x_{%i}$'%(comp+1),fontsize=16)

            draw()

        ssupernovae.on_changed(update)
        scomp.on_changed(update)


        resetax = axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            ssupernovae.reset()
        button.on_clicked(reset)

        show()

    def plot_interact_Av(self):
          
        figure(figsize=(10,10))
        P1=subplot(211)
        subplots_adjust(bottom=0.25)       
        l1, = plot(self.Av[0],'r',marker='o',markersize=7)
        L1, = plot(self.Av,'r')
        ylim(N.min(self.Av),N.max(self.Av))
        xlim(-1,len(self.sn_name))
        P1.set_ylabel(r'$A_V$',fontsize=16)
        P1.set_xlabel(self.sn_name[0],fontsize=16)  
        
        
        P2=subplot(212)
        L2, = plot(self.Av[0]*ones(len(self.chi2_Av_variation[0])),self.chi2_Av_variation[0],lw=2,color='b')
        l2, = plot(self.Av_variation[0],self.chi2_Av_variation[0], lw=2,color='b')
        P2.set_xlabel(r'$A_V$ ',fontsize=16)
        P2.set_ylabel(r'$\chi^2$',fontsize=16) 
        xlim(N.min(self.Av),N.max(self.Av))
        ylim(N.min(self.chi2_Av_variation),N.max(self.chi2_Av_variation))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)

        ssupernovae = Slider(axlam, 'supernova', 0, len(self.sn_name)-1, valinit=0)

        def update(val):

            SN=ssupernovae.val
            SN=int(SN)

            l1.set_ydata(self.Av[SN])
            l1.set_xdata(SN)
            
            P1.set_xlabel(self.sn_name[SN],fontsize=16)

            L2.set_xdata(self.Av[SN]*ones(len(self.chi2_Av_variation[SN])))
            L2.set_ydata(self.chi2_Av_variation[SN])
            l2.set_xdata(self.Av_variation[SN])
            l2.set_ydata(self.chi2_Av_variation[SN])
           
            draw()

        ssupernovae.on_changed(update)

        resetax = axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            ssupernovae.reset()
        button.on_clicked(reset)

        show()


    def plot_interact_grey(self):
          
        figure(figsize=(10,10))
        P1=subplot(211)
        subplots_adjust(bottom=0.25)       
        l1, = plot(self.trans[0],'k',marker='o',markersize=7)
        L1, = plot(self.trans,'k')
        ylim(N.min(self.trans),N.max(self.trans))
        xlim(-1,len(self.sn_name))
        P1.set_ylabel(r'$\Delta M_{grey}$',fontsize=16)
        P1.set_xlabel(self.sn_name[0],fontsize=16)  
        
        
        P2=subplot(212)
        L2, = plot(self.trans[0]*ones(len(self.chi2_grey_variation[0])),self.chi2_grey_variation[0],lw=2,color='b')
        l2, = plot(self.grey_variation[0],self.chi2_grey_variation[0], lw=2,color='b')
        P2.set_xlabel(r'$\Delta M_{grey}$ ',fontsize=16)
        P2.set_ylabel(r'$\chi^2$',fontsize=16) 
        xlim(N.min(self.trans),N.max(self.trans))
        ylim(N.min(self.chi2_grey_variation),N.max(self.chi2_grey_variation))

        axcolor = 'lightgoldenrodyellow'
        axlam = axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)

        ssupernovae = Slider(axlam, 'supernova', 0, len(self.sn_name)-1, valinit=0)

        def update(val):

            SN=ssupernovae.val
            SN=int(SN)

            l1.set_ydata(self.trans[SN])
            l1.set_xdata(SN)
            
            P1.set_xlabel(self.sn_name[SN],fontsize=16)

            L2.set_xdata(self.trans[SN]*ones(len(self.chi2_grey_variation[SN])))
            L2.set_ydata(self.chi2_grey_variation[SN])
            l2.set_xdata(self.grey_variation[SN])
            l2.set_ydata(self.chi2_grey_variation[SN])
           

            draw()

        ssupernovae.on_changed(update)

        resetax = axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            ssupernovae.reset()
        button.on_clicked(reset)


 #       rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
        
 #       radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
 #       def colorfunc(label):
 #           l1.set_color(label)
 #           draw()
 #       radio.on_clicked(colorfunc)

        show()



        
   
def plot_light_curve(list_model):


    Phase=array([-5.,0.,5.,10.,15.,20.,25.,30.])
    color=['b','c','g','r','k']
    Filtre=['USNf','BSNf','VSNf','RSNf','ISNf']
    M0=zeros((5,len(Phase)))
    alpha1=zeros((5,len(Phase)))
    alpha2=zeros((5,len(Phase)))
    alpha3=zeros((5,len(Phase)))

    N_point=200
   

    for i in range(len(list_model)):
        dico_model = cPickle.load(open(list_model[i]))
        M0[:,i]=dico_model['M0']
        alpha1[:,i]=dico_model['alpha'][:,0]
        alpha2[:,i]=dico_model['alpha'][:,1]
        alpha3[:,i]=dico_model['alpha'][:,2]
        data=dico_model['data']

    M0P=zeros((5,N_point))

    M0P_vec1=zeros((5,N_point))
    M0P_vec2=zeros((5,N_point))
    M0P_vec3=zeros((5,N_point))

    for i in range(5):
        phase,M0P[i]=Interpolation_spline(Phase,M0[i],N_point)
        phase,M0P_vec1[i]=Interpolation_spline(Phase,alpha1[i]*(mean(data[:,0])+sqrt(var(data[:,0]))),N_point)
        phase,M0P_vec2[i]=Interpolation_spline(Phase,alpha2[i]*(mean(data[:,1])+sqrt(var(data[:,1]))),N_point)
        phase,M0P_vec3[i]=Interpolation_spline(Phase,alpha3[i]*(mean(data[:,2])+sqrt(var(data[:,2]))),N_point)

    
    X_fig=6.5
    Y_fig=10
    
    figure(figsize=(X_fig,Y_fig))

    for i in range(5):
        fill_between(phase,M0P[i]+19.2-M0P_vec1[i],M0P[i]+19.2+M0P_vec1[i],color=color[i],alpha=0.3)
        plot(phase,M0P[i]+19.2,color[i],label=Filtre[i])

    title('Effect on the mean light curves ($\pm$1$\sigma$)\n with the First eigenvector')
    ylabel('$M_0(t) + cst$',fontsize=20)
    xlabel('days')
    ylim(-0.25,4.)
    xlim(-5,30)
    gca().invert_yaxis()
    legend(loc=3)


    figure(figsize=(X_fig,Y_fig))
    for i in range(5):
        fill_between(phase,M0P[i]+19.2-M0P_vec2[i],M0P[i]+19.2+M0P_vec2[i],color=color[i],alpha=0.3)
        plot(phase,M0P[i]+19.2,color[i],label=Filtre[i])

    title('Effect on the mean light curves ($\pm$1$\sigma$)\n with the Second eigenvector')
    xlabel('days')
    ylabel('$M_0(t) + cst$',fontsize=20)
    ylim(-0.25,4.)
    xlim(-5,30)
    gca().invert_yaxis()
    legend(loc=3)
 


    figure(figsize=(X_fig,Y_fig))
    for i in range(5):
        fill_between(phase,M0P[i]+19.2-M0P_vec3[i],M0P[i]+19.2+M0P_vec3[i],color=color[i],alpha=0.3)
        plot(phase,M0P[i]+19.2,color[i],label=Filtre[i])

    title('Effect on the mean light curves ($\pm$1$\sigma$)\n with the Third eigenvector')
    xlabel('days')
    ylabel('$M_0(t) + cst$',fontsize=20)
    ylim(-0.25,4.)
    xlim(-5,30)
    gca().invert_yaxis()
    legend(loc=3)  

    


def plot_sn_multiphase(sn_name,list_spectra,list_model):
 
    cst=[19.2-1.25,19.2,19.2+1,19.2+2,19.2+3,19.2+4.2,19.2+5.5,19.2+7.]
    #cst=[19.2-1.25,19.2,19.2+1.2,19.2+2,19.2+3,19.2+4.2,19.2+5.5,19.2+6.]
   
    phase=['days','day','days','days','days','days','days','days']
    indic=0
    figure(figsize=(10,8))
    for i in range(len(list_model)):
        
        dico_spectra = cPickle.load(open(list_spectra[i]))
        dico_model = cPickle.load(open(list_model[i]))
        phase_sn=dico_spectra['phase_bis_sn']
        X=dico_model['X']
        Y_cosmo_corrected=dico_model['Mag_no_corrected']
        Y_ODR_build=dico_model['Y_build']
        Y_ODR_build_error=dico_model['Y_build_error']
        name=dico_model['sn_name']
        Y_err=dico_spectra['training']['Y_err']
        
        
        
        for sn in range(len(name)):
            if name[sn]==sn_name:
                if sum(Y_err[sn])>10**15:
                    print 'pouet'
                else:
                    if i==1:
                        redshift=dico_spectra['training']['SALT2'][sn,3]
                    indic+=1
                    y_moins= Y_ODR_build[sn]+cst[i]-sqrt(Y_ODR_build_error[sn])
                    y_plus= Y_ODR_build[sn]+cst[i]+sqrt(Y_ODR_build_error[sn])
                    fill_between(X,y_moins,y_plus,color='b',alpha=0.3)
                    fill_between(X,Y_cosmo_corrected[sn]-Y_err[sn]+cst[i],Y_cosmo_corrected[sn]+Y_err[sn]+cst[i],color='r',alpha=0.3)
                    if indic ==1:
                        plot(X,Y_ODR_build[sn]+cst[i],'b',label=r'Spectral indicator model')
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'r',label='Observed spectrum')
                    else:
                        plot(X,Y_ODR_build[sn]+cst[i],'b')
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'r')
                    ylabel(r'$M(t,\lambda) + cst$')
                    xlabel('wavelength [$\AA$]')
                    print average(abs(Y_ODR_build[sn]-Y_cosmo_corrected[sn]),weights=1./sqrt(Y_ODR_build_error[sn]+Y_err[sn]**2))
                    for SN in range(len(phase_sn[0])):
                        if phase_sn[0][SN]==sn_name:
                            PH=SN
                    text(X[-10],Y_cosmo_corrected[sn,-1] + cst[i], '%10.1f '%(phase_sn[1][PH])+phase[i])
    ylim(-2.1,11.8)
    xlim(3200,9700)
    gca().invert_yaxis()
    title(sn_name+' $z=%.3f$'%(redshift))
    legend(loc=4)



def plot_sn_twins_multiphase(sn_name1,sn_name2,list_spectra):
 
    cst=[19.2-1.25,19.2,19.2+1,19.2+2,19.2+3,19.2+4.2,19.2+5.5,19.2+7.]
    RANGE=['[-7.5,-2.5]','[-2.5,2.5]','[2.5,7.5]','[7.5,12.5]','[12.5,17.5]','[17.5,22.5]','[22.5,27.5]','[27.5,32.5]']

    phase=['days','day','days','days','days','days','days','days']
    indic1=0
    indic2=0
    figure(figsize=(10,8))
    
    for i in range(len(list_spectra)):
        print RANGE[i]
        dico_spectra = cPickle.load(open(list_spectra[i]))
        phase_sn=dico_spectra['phase_bis_sn']
        X=dico_spectra['X']
        Y_cosmo_corrected=dico_spectra['training']['Y_cosmo_corrected']
        name=dico_spectra['training']['sn_name']
        Y_err=dico_spectra['training']['Y_err']

 
        for sn in range(len(name)):
            if name[sn]==sn_name1 and sn_name2 in name:
                if sum(Y_err[sn])>10**15:
                    print 'pouet'
                else:
                    indic1+=1
                    fill_between(X,Y_cosmo_corrected[sn]-Y_err[sn]+cst[i],Y_cosmo_corrected[sn]+Y_err[sn]+cst[i],color='r',alpha=0.3)
                    if indic1 ==1:
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'r',label=name[sn])
                    else:
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'r')
                    ylabel(r'$M(t,\lambda) + cst$')
                    xlabel('wavelength [$\AA$]')
                     
                   
                    

            if name[sn]==sn_name2 and sn_name1 in name:
                if sum(Y_err[sn])>10**15:
                    print 'pouet'
                else:
                    indic2+=1
                    fill_between(X,Y_cosmo_corrected[sn]-Y_err[sn]+cst[i],Y_cosmo_corrected[sn]+Y_err[sn]+cst[i],color='b',alpha=0.3)
                    if indic2 ==1:
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'b',label=name[sn])
                    else:
                        plot(X,Y_cosmo_corrected[sn]+cst[i],'b')
                    ylabel(r'$M(t,\lambda) + cst$')
                    xlabel('wavelength [$\AA$]')
                    
                   
                    text(X[-1]+100,Y_cosmo_corrected[sn,-1] + cst[i], RANGE[i])

                

    ylim(-2.1,11.8)
    xlim(3200,9700)
    gca().invert_yaxis()
    title(sn_name1+' & '+sn_name2)
    legend(loc=4)




def plot_wRMS_time(wlength,matrix,ylabel,title,cmap=matplotlib.cm.jet,plotpoints=True):
    
    values = [diag(matrix,k=i) for i in range(len(matrix))]
    
    means=map(mean,values)
    stds=map(std,values)
    med,nmad=array([median_stats(x) for x in values]).T
    

    #Plot the matrix
    wlength=[wlength[0],wlength[-1],wlength[-1],wlength[0]]
    fig = figure(dpi=150,figsize=(8,8))
    ax = fig.add_axes([0.08,0.09,0.88,0.88]) #title=title
    im = ax.imshow(matrix,cmap=cmap,extent=wlength,interpolation='nearest')
    cb = fig.colorbar(im)
    cb.set_label(ylabel,size='x-large')
    ax.set_xlabel(r'Wavelength [$\AA$]',size='large')
    ax.set_ylabel(r'Wavelength [$\AA$]',size='large')
    



def plot_grey_phase(sn_name,list_model,Cluster=False):
    
    phase=[-5,0,5,10,15,20,25,30]
    figure()
    DM=[]
    PH=[]
    
    for i in range(len(list_model)):
        dico_model = cPickle.load(open(list_model[i]))
        name=dico_model['sn_name']
        trans=dico_model['delta_M_grey']

        for sn in range(len(name)):
            if name[sn]==sn_name:
                DM.append(trans[sn])
                PH.append(phase[i])

    plot(phase,zeros(len(phase)),'k')

    ylim(-0.5,0.5)
    xlim(-5,30)
    plot(PH,DM)
    if Cluster:
        title(sn_name+' (in galaxy cluster)')
    else:
        title(sn_name)
    xlabel('days')
    ylabel('$\Delta M_{grey}(t)$')



    
#    def plot_sn_SALT_vs_model(self,sn_name,phase,META):  
#
#
#        for sn in range(len(self.sn_name)):
#            if self.sn_name[sn]==sn_name:
#                
#                Mag_no_corrected=copy.deepcopy(self.Mag_no_corrected[sn])
#                Y_ODR_build=copy.deepcopy(self.Y_ODR_build[sn])
#                    
#                #for Bin in range(len(self.X)):
#                #    Mag_no_corrected[Bin]-=(self.Av[sn]*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
#                #    Y_ODR_build[Bin]-=(self.Av[sn]*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
#                
#                figure()
#
#                from ToolBox.Astro import Templates as T
#                from ToolBox.Astro import Coords
#                meta = SnfMetaData.SnfMetaData(META)
#                SALT=T.Templates(template='Salt2',X0=meta[sn_name]['salt2.X0'], X1=meta[sn_name]['salt2.X1'], c=meta[sn_name]['salt2.Color'])
#                x,y=SALT.spec_at_given_phase(phase)
#                Y=Coords.flbda2ABmag(x,y)-5.*log10(d_l(meta[sn_name]['host.zcmb'],SNLS=True))+5.
#
#                y_moins= Y_ODR_build+19.2-sqrt(self.Y_ODR_build_error[sn])
#                y_plus= Y_ODR_build+19.2+sqrt(self.Y_ODR_build_error[sn])
#                fill_between(self.X,y_moins,y_plus,color='b',alpha=0.3)
#                fill_between(self.X,Mag_no_corrected-self.Y_err[sn]+19.2,Mag_no_corrected+self.Y_err[sn]+19.2,color='k',alpha=0.3)
#                plot(x,Y+19.2,'r',label='SALT2')
#                plot(self.X,Y_ODR_build+19.2,'b',label='SUGAR')
#                plot(self.X,Mag_no_corrected+19.2,'k',label='Observed')
#                ylim(-1,4.8)
#                xlim(3500,9000)
#                gca().invert_yaxis()
#                ylabel('Mag AB + cst')
#                xlabel('wavelength [$\AA$]')
#                title(self.sn_name[sn])#+' '+self.key)
#                legend()
#
   
        
#def compute_SALT24_model(sn_name,phase,META=rep_BEDELL+'META_JLA.pkl'):
#
#    from ToolBox.Astro import Templates as T
#    from ToolBox.Astro import Coords
#    meta = SnfMetaData.SnfMetaData(META)
#    SALT=T.Templates(template='Salt2',X0=meta[sn_name]['salt2.X0'], X1=meta[sn_name]['salt2.X1'], c=meta[sn_name]['salt2.Color'])
#    x,y=SALT.spec_at_given_phase(phase)
#    Y=Coords.flbda2ABmag(x,y)-5.*log10(d_l(meta[sn_name]['host.zcmb'],SNLS=True))+5.
#
#    return 


def plot_extinction_law_alpha1(dico_max):

    dic=cPickle.load(open(dico_max))
    Gamma=dic['GF'].alpha[:,0]
    Alpha1=dic['GF'].alpha[:,1]
    X=dic['X']

    P.figure()
    P.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.97)
    P.plot(X,Gamma,'r',linewidth=3,label='$\gamma_{\lambda}$ before $D$ computation')
    P.plot(X,8*Alpha1,'b',linewidth=3,label='$8 \\times \\alpha^1$ before $D$ computation')
    P.legend()
    P.xlabel('wavelength [$\AA$]')

def plot_disp_eig(LIST_dic):

    COLOR=['r','k','c','b','g','y']

    STD=[]
    color_law=[]
    X=[]
    for i in range(len(LIST_dic)):
        SP=SUGAR_plot(LIST_dic[i])
        SP.plot_spectrum_corrected()
        P.close()
        if i==0:
            STD.append(N.std(SP.Mag_no_corrected,axis=0))
            X.append(SP.X) 
        STD.append(N.std(SP.MAG_CORRECTED,axis=0))
        color_law.append(SP.slope)
    P.figure(figsize=(20,20))
    P.subplot(2,1,1)
    P.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99,hspace=0.001)
    P.xlabel('wavelength [$\AA$]')
    P.legend()

    for i in range(len(LIST_dic)+1):
        if i==0:
            LABEL='No correction'
        else:
            if i==1:
                LABEL='%i factor and extinction corrected'%(i)
            else:
                LABEL='%i factors and extinction corrected'%(i)

        P.plot(X[0],STD[i],COLOR[i],label=LABEL,linewidth=3)

    P.ylabel('STD (mag)')

    P.ylim(0,0.65)
    P.xticks([3100,8800],['',''])
    P.xlim(3200,8700)
    P.legend()
    P.subplot(2,1,2)
    for i in range(len(LIST_dic)):
        if i==0:
            LABEL='$\gamma_{\lambda}$ (with %i factor)'%(i+1)
        else:
            LABEL='$\gamma_{\lambda}$ (with %i factors)'%(i+1)
        P.plot(X[0],color_law[i],COLOR[i+1],linewidth=3,label=LABEL)

    P.ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$')
    P.xticks([4000,5000,6000,7000,8000],['4000','5000','6000','7000','8000'])
    P.xlim(3200,8700)
    P.xlabel('wavelength [$\AA$]')
    P.ylim(0,2.2)
    P.legend()


def plot_vec_emfa(LIST_dic,vec,ALIGN=[1,1,1,1,1]):

    COLOR=['k','c','b','g','y']

    STD=[]
    X=[]
    for i in range(len(LIST_dic)):
        SP=SUGAR_plot(LIST_dic[i])
        SP.plot_spectrum_corrected()
        P.close()
        if i==0:
            X.append(SP.X)
        STD.append([])
        for j in range(vec+1):
            STD[i].append(SP.dico['Lambda'][:,j]/(N.sqrt(N.sum(SP.dico['Lambda'][:,j]**2))))

    P.figure(figsize=(20,15))
    P.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99)

    for j in range(vec+1):
        P.subplot(3,2,j+1)
        for i in range(len(LIST_dic)):
            if i==0:
                LABEL='Vectors on residuals (%i factor corrected and extinction corrected)'%(i+1)
            else:
                LABEL='Vectors on residuals (%i factors corrected and extinction corrected)'%(i+1)
            
            P.plot(X[0],ALIGN[i]*STD[i][j],COLOR[i],label=LABEL,linewidth=3)

        P.ylabel('$\Lambda_{%i}$'%(j+1),fontsize=16)
        P.xticks([4000,5000,6000,7000,8000],['4000','5000','6000','7000','8000'])
        P.xlabel('wavelength [$\AA$]')
    #P.ylim(0,0.65)

        P.xlim(3200,8700)
    P.legend(bbox_to_anchor = (2.2, 0.7))


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################




if __name__=='__main__':


    #lst_dic=[]
    #for i in range(5):
    #    lst_dic.append('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_for_phd/model_at_max_%i_eigenvector_without_grey_without_MFR_problem.pkl'%(i+1))
    
    #plot_disp_eig(lst_dic)
    #P.savefig('../These_plot/plot_phd/Chapitre7/STD_choice_eigenvector.pdf')
    #plot_vec_emfa(lst_dic,4,ALIGN=[-1,1,1,-1,-1])


    SP=SUGAR_plot('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_for_phd/model_at_max_3_eigenvector_without_grey_without_MFR_problem_test_RV_save_before_PCA.pkl')
    #SP.plot_projection_ortho()
    #SP.plot_Av_distri()
    #SP.plot_mass_step('/sps/snovae/user/leget/BEDELL/Host.pkl','/sps/snovae/user/leget/CABALLO/localhost_idr.pkl','/sps/snovae/user/leget/CABALLO/META.pkl',LOCAL=None,dic_NaID='/sps/snovae/user/leget/CABALLO/SUGAR_validation/NaID_dico.pkl')
    #SP.subplot_mass_step('/sps/snovae/user/leget/BEDELL/Host.pkl','/sps/snovae/user/leget/CABALLO/localhost_idr.pkl','/sps/snovae/user/leget/CABALLO/META.pkl')
    #SP.plot_Av_vs_color('/sps/snovae/user/leget/CABALLO/META_JLA.pkl')
    #SP.plot_bin_Av_slope(42)
    #SP.plot_spectrum_corrected()
    #    P.savefig('../These_plot/plot_phd/Chapitre7/all_spectrum_corrected_without_grey_with_%i_eigenvector.pdf'%(i+1))
    #SP=SUGAR_plot('/sps/snovae/user/leget/CABALLO/model_training_SUGAR/model_at_max_3_eigenvector_save_before_PCA.pkl')
     #   FIG=[]
     #   for j in range(i+1):

            #FIG1='../These_plot/plot_septembre_2015/plot_at_max/FIG1_%i_eigenvectorSI_%i_eigenvector_on_residuals_without_grey.pdf'%((i+1,j+1))
            #FIG2='../These_plot/plot_septembre_2015/plot_at_max/FIG2_%i_eigenvectorSI_%i_eigenvector_on_residuals_without_grey.pdf'%((i+1,j+1))

      #      FIG.append('Fit_%i_eigenvectorSI_%i_eigenvector_without_grey.pdf'%((i+1,j+1)))
      #  SP.plot_spectral_variability(name_fig=FIG)
            #SP.spectra_corrected_pca_residuals(j,Name_fig=[FIG1,FIG2])
        #SP.plot_spectrum_corrected()
        #P.savefig('../These_plot/plot_septembre_2015/plot_at_max/spectra_corrected_%i_without_grey.pdf'%(i+1))
        #SP.plot_color_law()
        #P.savefig('../These_plot/plot_septembre_2015/plot_at_max/color_law_%i_without_grey.pdf'%(i+1))
        #SP.plot_corr_matrix()
        #P.savefig('../These_plot/plot_septembre_2015/plot_at_max/corr_matrix_%i_without_grey.pdf'%(i+1))
