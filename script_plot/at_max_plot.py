"""plot for extinction law."""

import pylab as P
import numpy as N
import cPickle
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics 
import copy
from sugar.multilinearfit import *
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons


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


        floor_filter=N.array([False]*len(self.X))

        for Bin in range(len(self.X)):
            if self.X[Bin]>6360. and self.X[Bin]<6600.:
                floor_filter[Bin]=True
            else:
                continue
        self.floor_filter=floor_filter



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


    def plot_Av_distri(self):

        P.figure()
        #subplots_adjust(left=0.08, bottom=0.10, right=0.97, top=0.98)
        P.hist(self.Av,bins=25,color='r')
        P.ylabel('number of supernovae',fontsize=16)
        P.xlabel('$A_{\lambda_0}$',fontsize=16)
        P.ylim(0,20)
        P.legend(loc=4)


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
                Mag_all_sn[:,Bin]-=(N.dot(self.alpha[Bin],self.data.T))+self.trans+(self.Av*Astro.Extinction.extinctionLaw(self.X[Bin],Rv=self.Rv))
            else:
                Mag_all_sn[:,Bin]-=(N.dot(self.alpha[Bin],self.data.T))+self.trans
        
        self.MAG_CORRECTED= Mag_all_sn
        P.figure(figsize=(12,12))
        P.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.995,hspace=0.001)
        for sn in range(len(self.sn_name)):
            P.subplot(2,1,1)
            if No_corrected:
                if sn==0:
                    P.plot(self.X,self.Mag_no_corrected[sn]+16.2,'r',linewidth=3,alpha=0.5,label='Observed spectra')
                else:
                    P.plot(self.X,self.Mag_no_corrected[sn]+16.2,'r',linewidth=3,alpha=0.5)

            if sn==0:
                P.plot(self.X,Mag_all_sn[sn]+20.5,'b',linewidth=3,alpha=0.5,label='Corrected spectra ($q_1$, $q_2$, $q_3$, $A_{\lambda_0}$)')
            else:
                P.plot(self.X,Mag_all_sn[sn]+20.5,'b',linewidth=3,alpha=0.5)


        P.ylabel('Mag AB + cst',fontsize=20)
        P.ylim(-5,7)
        P.xticks([2500.,9500.],['toto','pouet'])
        P.xlim(self.X[0]-60,self.X[-1]+60)
        P.legend(loc=4)
        P.gca().invert_yaxis()
        STD=N.std(Mag_all_sn,axis=0)
        STD_no_correct=N.std(self.Mag_no_corrected,axis=0)
        P.subplot(2,1,2)
        if No_corrected:
            P.plot(self.X,STD_no_correct,'r',linewidth=3,label=r'Observed RMS average between $[6360\AA,6600\AA]$ = %.2f mag' %(N.mean(STD_no_correct[self.floor_filter])))

        P.plot(self.X,STD,'b',linewidth=3,label=r'Corrected RMS average between $[6360\AA,6600\AA]$ = %.2f mag' %(N.mean(STD[self.floor_filter])))

        P.plot(self.X,N.zeros(len(self.X)),'k')
        P.ylabel('RMS (mag)',fontsize=20)
        P.xlabel('wavelength [$\AA$]',fontsize=20)
        P.xlim(self.X[0]-60,self.X[-1]+60)
        P.ylim(0.0,0.62)
        P.legend()

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



    def plot_spectral_variability(self,name_fig=None,ERROR=False):
       
        c=['r','b','g','k']
        NUMBER=['first','second','third']

        for correction in range(len(self.alpha[0])):

            P.figure(figsize=(7,6))
            gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
            P.subplots_adjust(left=0.13, bottom=0.11, right=0.99, top=0.95,hspace=0.001)

            #ax1=subplot(211)
            P.subplot(gs[0])
            #MIN=[-0.21,-0.21,-0.21]
            #MAX=[0.24,0.24,0.24]

            y_moins=N.zeros(len(self.X))
            y_plus=N.zeros(len(self.X))
            
            for Bin in range(len(self.X)):
                y_moins[Bin]=self.M0[Bin]+19.2-self.alpha[Bin,correction]*(N.mean(self.data[:,correction])+N.sqrt(N.var(self.data[:,correction])))
                y_plus[Bin]=self.M0[Bin]+19.2+self.alpha[Bin,correction]*(N.mean(self.data[:,correction])+N.sqrt(N.var(self.data[:,correction])))

            P.fill_between(self.X,self.M0+19.2,y_plus,color='m',alpha=0.7 )
            P.fill_between(self.X,y_moins,self.M0+19.2,color='g',alpha=0.7)
            p3 = P.plot(self.X,self.M0+19.2,'b',linewidth=2,label='toto')
            P.ylim(-0.6,1.4)
            P.gca().invert_yaxis()
            P.ylabel('$M_0(t=0,\lambda) +$ cst. (mag)',fontsize=16)
            P.title(r'Average spectrum with $\pm$1$\sigma$ variation ($\alpha_{%i}(t=0,\lambda)$)'%(correction+1))
            p1 = P.Rectangle((0, 0), 1, 1, fc="magenta")
            p2 = P.Rectangle((0, 0), 1, 1, fc="green")
            P.legend([p1, p2], ['+1$\sigma$', '-1$\sigma$'])
            P.xticks([2500.,9500.],['toto','pouet'])
            P.xlim(self.X[0]-60,self.X[-1]+60)


            P.subplot(gs[1])
            if ERROR:
                P.fill_between(self.X,self.alpha[:,correction]-self.alpha_err_Jackknife[:,correction],self.alpha[:,correction]+self.alpha_err_Jackknife[:,correction],color='b',alpha=0.7 )

            mean_effect=N.std(self.data[:,correction])*N.mean(abs(self.alpha[:,correction]))
            P.plot(self.X,self.alpha[:,correction],'b',linewidth=3)#,label=r'Average effect (mag)=%.3f'%((mean_effect)))
            P.ylabel(r'$\alpha_{%i}(t=0,\lambda)$'%(correction+1),fontsize=16)
            P.xlabel('wavelength [$\AA$]',fontsize=16)
            P.xlim(self.X[0]-60,self.X[-1]+60)

            P.legend(loc=4)

            if name_fig!=None:
                P.savefig('../These_plot/plot_septembre_2015/'+name_fig[correction])


    
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
                

        CCM=N.zeros(len(self.X))
        slopes_star=N.zeros(len(self.X))
        slopes_star_err=N.zeros(len(self.X))

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
    
        
        CCM31=Astro.Extinction.extinctionLaw(self.X,Rv=3.1,law='CCM89')
        CCM26=Astro.Extinction.extinctionLaw(self.X,Rv=2.7,law='CCM89')
        CCM14=Astro.Extinction.extinctionLaw(self.X,Rv=1.4,law='CCM89')
        Ind_med=(len(self.X)/2)-1
        CCM31/=CCM31[Ind_med]
        CCM26/=CCM26[Ind_med]
        CCM14/=CCM14[Ind_med]
        AVV=N.linspace(-0.5,1,20)       
        #for X_Bin in range(len(self.X)):
        #    slopes_star[X_Bin]=(slopes[X_Bin]/slopes[BIN])
        

        P.figure(figsize=(11,11))
        P.subplots_adjust(left=0.12, bottom=0.07, right=0.99, top=0.995)
        MAG=copy.deepcopy(Mag_all_sn)
        
        for sn in range(len(MAG[:,0])):
            for X_Bin in range(len(self.X)):
                for correction in range(len(self.alpha[0])):
                    MAG[sn,X_Bin]-=alpha[X_Bin,correction]*data[sn,correction]

                MAG[sn,X_Bin]-=trans[sn]+M0[X_Bin]

     
        self.MAG=MAG

        P.subplot(2,1,2)

        for i,sn in enumerate(self.sn_name):
            if sn =='SN2007le':
                P.scatter(Av[i],MAG[i,Bin],zorder=100,c='r',marker='^',s=100,label='SN2007le')
            if sn =='SN2005cf':
                P.scatter(Av[i],MAG[i,Bin],zorder=100,c='r',marker='o',s=100,label='SN2005cf')
  

        P.errorbar(Av,MAG[:,Bin],linestyle='',xerr=None,yerr=N.sqrt(self.Y_build_error[:,Bin]),ecolor='blue',alpha=0.7,marker='.',zorder=0)
        scat=P.scatter(Av,MAG[:,Bin],zorder=100,s=50,c='b')
            
        P.plot(AVV,slopes[Bin]*AVV,'r',label='$\gamma_{%i\AA}$'%(self.X[Bin]))
        P.plot(AVV,CCM31[Bin]*AVV,'k--',linewidth=3,label='CCM $(R_V=3.1)$')
        P.plot(AVV,CCM26[Bin]*AVV,'r--',linewidth=3,label='CCM $(R_V=2.7)$')
        P.plot(AVV,CCM14[Bin]*AVV,'k-.',linewidth=3,label='CCM $(R_V=1.4)$')
        P.ylabel('$M(t=0,\lambda)-M_0(t=0,\lambda) - \sum_{i=1}^{i=3} \\alpha_i(0,\lambda) q_i$',fontsize=18)
        P.xlabel('$A_{\lambda_0}$',fontsize=20)
        P.text(-0.55,1.55,r'$\lambda=%i \AA$'%(self.X[Bin]),fontsize=20)
        P.ylim(min(MAG[:,Bin])-0.3,max(MAG[:,Bin])+0.3)
        #gca().invert_yaxis()
        P.legend(loc=4)

        P.subplot(2,1,1)
        P.scatter(self.X[Bin],slopes[Bin],c='r',marker='o',s=100)
        P.plot(self.X,slopes,'r',label= '$\gamma_{\lambda}$')
        P.plot(self.X,CCM31,'k-.',linewidth=3,label= 'CCM $(R_V=3.1)$')
        P.plot(self.X,CCM26,'r--',linewidth=3,label= 'CCM $(R_V=2.7)$')
        P.plot(self.X,CCM14,'k--',linewidth=3,label= 'CCM $(R_V=1.4)$')
        P.ylabel(r'$(\partial A_{\lambda}$ / $\partial A_V)$',fontsize=20)
        P.xlabel('wavelength [$\AA$]',fontsize=20)
        P.ylim(0.4,2.1)
        P.xlim(self.X[0]-60,self.X[-1]+60)
        P.legend()
        

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
    P.xlabel('wavelength [$\AA$]',fontsize=20)
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

    P.ylabel('RMS (mag)',fontsize=20)

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

    P.ylabel(r'$\gamma(\lambda)$',fontsize=20)
    P.xticks([4000,5000,6000,7000,8000],['4000','5000','6000','7000','8000'])
    P.xlim(3200,8700)
    P.xlabel('wavelength [$\AA$]',fontsize=20)
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

        P.ylabel('$\Lambda_{%i}$'%(j+1),fontsize=20)
        P.xticks([4000,5000,6000,7000,8000],['4000','5000','6000','7000','8000'])
        P.xlabel('wavelength [$\AA$]',fontsize=20)
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

    import sugar
    import os
    path = os.path.dirname(sugar.__file__)
    
    lst_dic=[]
    for i in range(5):
        #lst_dic.append(path+'/data_output/SUGAR_model_for_phd/model_at_max_%i_eigenvector_without_grey_without_MFR_problem.pkl'%(i+1))
        lst_dic.append('../../Desktop/sugar_paper_output/model_at_max_%i_eigenvector_without_grey_with_sigma_clipping.pkl'%(i+1))
    
    plot_disp_eig(lst_dic)
    P.savefig('plot_paper/residual_emfa_vectors_at_max.pdf')
    plot_vec_emfa(lst_dic,4,ALIGN=[-1,1,1,-1,-1])
    P.savefig('plot_paper/STD_choice_eigenvector.pdf')
    
    #SP=SUGAR_plot('../../Desktop/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl')
    #SP.plot_bin_Av_slope(42)
    #P.savefig('plot_paper/CCM_law_bin42.pdf')#,transparent=True)
    #SP.plot_spectrum_corrected()
    #P.savefig('plot_paper/all_spectrum_corrected_without_grey_with_3_eigenvector.pdf')
    #SP.plot_spectral_variability(name_fig=None)

