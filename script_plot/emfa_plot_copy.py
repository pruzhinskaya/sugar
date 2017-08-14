import pylab as P
import numpy as N
import scipy as S
import cPickle
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics
from ToolBox import Hubblefit as H
import copy
from Load_time_spectra import d_l
from Passage import passage_error,passage_error_error_sn
import multilinearfit as Multi
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def branch_filter(EWSi6000,EWSi5972,EWSi6355min,EWSi6355max,EWSi5972min,EWSi5972max):
        
    filt=N.array([True]*len(EWSi6000))
    return filt & (EWSi6355min<EWSi6000) & (EWSi6000<EWSi6355max) & (EWSi5972min<EWSi5972) & (EWSi5972<EWSi5972max) 
       
    
def wang_filter(VSi6000,VSi6355min,VSi6355max):
                
    filt=N.array([True]*len(VSi6000))
    return filt & (VSi6355min<((6355.-VSi6000)/6335.)*299792458*10.**(-3)) & (((6355.-VSi6000)/6335.)*299792458*10.**(-3)<VSi6355max)



def SP_set_kwargs(inkw, label, **kwargs):

    """Read dictionnary *label* from *inkw*, and set some default 
    values from *kwargs*."""

    outkw = inkw.get(label, {})
    for key in kwargs:
        outkw.setdefault(key, kwargs[key])

    return outkw


def PlotEllipse(x,y,xerr,yerr,covxy,ax,color='b',alpha=0.2):
    """                           
    Plot elliptic error using the covariance matrix  
    x and y are the data   
    xerr and yerr are the error (sigma) on x and y respetively   
    covxy is the covariance between x and y            
    ax is a matplotib axe    

    """

    V=N.array([N.array(((i,k),(k,j))) for i,j,k in zip(xerr**2,yerr**2,covxy)])
    Eingen=map(N.linalg.eig,V)
    cs=N.array([N.sqrt(float(Eingen[n][0][1])) for n in range(len(Eingen))])
    bs=N.array([N.sqrt(float(Eingen[n][0][0])) for n in range(len(Eingen))])
    alphas=N.array([N.arctan(float(Eingen[n][1][1][0])/float(Eingen[n][1][1][1])) for n in range(len(Eingen))])

    """    
    xy - center of ellipse   
    width - length of horizontal axis   
    height - length of vertical axis
    angle - rotation in degrees (anti-clockwise)                                                                                                                                     
    """

    ells = [Ellipse(xy=[i,j], width=2.*k, height=2.*l, angle=(m*180.)/N.pi) for i,j,k,l,m in zip(x,y,bs,cs,alphas)]
    for ell in ells:
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ell.set_facecolor(color)
        ell.set_edgecolor(color)

def median_stats(a, weights=None, axis=None, scale=1.4826, corrected=True):
    """Compute [weighted] median and :func:`nMAD` of array *a* along
        *axis*. Weighted computation is implemented for *axis* = None
        only. If *corrected*, apply finite-sample correction from Croux &
        Rousseeuw (1992)."""
    
    if weights is not None:
        if axis is not None:
            raise NotImplementedError("implemented on 1D-arrays only")
        else:
            med  = wpercentile(a, 50., weights=weights)
            nmad = wpercentile(N.abs(a - med), 50., weights=weights) * scale
    else:
        med = N.median(a, axis=axis)
        if axis is None:
            umed = med                      # Scalar
    if weights is not None:
        if axis is not None:
            raise NotImplementedError("implemented on 1D-arrays only")
        else:
            med  = wpercentile(a, 50., weights=weights)
            nmad = wpercentile(N.abs(a - med), 50., weights=weights) * scale
    else:
        med = N.median(a, axis=axis)
        if axis is None:
            umed = med                      # Scalar
        else:
            umed = N.expand_dims(med, axis) # Same ndim as a
        nmad = N.median(N.absolute(a - umed), axis=axis) * scale

    if corrected:
        # Finite-sample correction on nMAD (Croux & Rousseeuw, 1992)
        if axis is None:
            n = N.size(a)
        else:
            n = N.shape(a)[axis]
        if n<=9:
            c = [0,0,1.196,1.495,1.363,1.206,1.200,1.140,1.129,1.107][n]
        else:
            c = n/(n-0.8)
        nmad *= c
    
    return med,nmad


def plot_matrix(wlength,matrix,ylabel,title,cmap=P.cm.jet,plotpoints=True,VM=[-1,1]):
    
    values = [diag(matrix,k=i) for i in range(len(matrix))]
    
    means=map(mean,values)
    stds=map(std,values)
    med,nmad=array([median_stats(x) for x in values]).T
    
    #Plot the matrix
    wlength=[wlength[0],wlength[-1],wlength[-1],wlength[0]]
    fig = figure(dpi=150,figsize=(8,8))
    #fig = figure(figsize=(12,12))
    ax = fig.add_axes([0.08,0.09,0.88,0.88]) #title=title
    im = ax.imshow(matrix,vmin=VM[0],vmax=VM[1],cmap=cmap,extent=wlength,interpolation='nearest')
    cb = fig.colorbar(im)
    cb.set_label(ylabel,size='x-large')
    #cb.set_clim(-3,3)
    ax.set_xlabel(r'Wavelength [$\AA$]',size='large')
    ax.set_ylabel(r'Wavelength [$\AA$]',size='large')
 

class plot_emfa_spectral_indicator_space:

    def __init__(self,dic_emfa):
        
        self.dic=cPickle.load(open(dic_emfa))
        self.SI_latex =  ['pEW Ca II H&K', r'pEW Si II $\lambda$4131', 'pEW Mg II',
                          'pEW Fe $\lambda$4800', 'pEW S II W', 'pEW Si II $\lambda$5972',
                          'pEW Si II $\lambda$6355', 'pEW O I $\lambda$7773', 'pEW Ca II IR',
                          'V Si II $\lambda$4131','V WS II $\lambda$5454','V WS II $\lambda$5640',
                          'V Si II $\lambda$6355']
        self.SIsavefig=  ['pEWCaIIHK', 'pEWSiII4131', 'pEWMgII',
                          'pEWFe4800', 'pEWSIIW', 'pEWSiII5972',
                          'pEWSiII6355', 'pEWOI7773', 'pEWCaIIIR',
                          'VSiII4131','VWSII5454','VWSII5640',
                          'VSiII6355']
        


    def plot_BIC(self):

        BIC=N.zeros(13)

        for i in range(13):
            BIC[i]=-2.*self.dic['Log_L'][i]+(13*(i+2)-((i+1)*i)/2)*N.log(N.sum(self.dic['filter']))
        self.dic['BIC']=BIC
        PC=N.linspace(1,13,13)
        
        P.plot(PC,self.dic['BIC'],'-o')
        P.plot([PC[N.argmin(self.dic['BIC'])],PC[N.argmin(self.dic['BIC'])]],[min(self.dic['BIC']),max(self.dic['BIC'])],linewidth=3,label='BIC minimum position')
        XB=[]
        for xa in range(len(PC)):
            XB.append('%i'%(PC[xa]))
        P.xticks(PC,XB)

        P.xlabel('number of PC')
        P.ylabel('BIC value')
        P.legend()


    def plot_SI_space(self,Zcolor=None,Zname='Francis',rep_save=None,key_save='all',Plot_outlier=True,Outlier_emfa=False):
        
        data=self.dic['data']
        error=self.dic['error']

        if Plot_outlier:
            FILTRE=N.array([True]*len(data[:,0]))
        else:
            FILTRE=self.dic['filter']

        for j in range(len(data[0])):
        
            i=0
            k=1
            fig=P.figure(figsize=(25,12))
            P.subplots_adjust(hspace=0.1,wspace = 0.4)
            while i<=len(data[0])-1:
                    
                if i!=j:
                    P.subplot(3,4,k)
                    if Zcolor is not None:
                        cmap=P.cm.Blues
                        norm=(Zcolor-Zcolor.min())/(Zcolor-Zcolor.min()).max()
                        col=cmap(norm)

                    else:
                        D=N.linspace(-1,1,len(data[:,j][FILTRE]))
                        cmap=P.cm.jet
                        norm=(D-D.min())/(D-D.min()).max()
                        col=cmap(norm)
                        rho=N.corrcoef(data[:,j][FILTRE],data[:,i][FILTRE])[0,1]
                        SIG=Statistics.correlation_significance(rho,len(data[:,j][FILTRE]),sigma=True)
                        if i==2 or i ==9:
                            cstY=0.0
                            if i==9:
                                cstY=0.05
                        else:
                            cstY=1.
                        if j==2 or j==9:
                            cstX=0.0
                            if i==9:
                                cstX=0.05
                        else:
                            cstX=1.

                        if j==2 or j==9:
                            print cstX

                        P.text(min(data[:,j][FILTRE])-cstX*N.std(data[:,j][FILTRE]),min(data[:,i][FILTRE])-cstY*N.std(data[:,i][FILTRE]),r'$\rho=%.3f$, $\sigma=%.3f$'%((rho,SIG)))
                    P.errorbar(data[:,j][FILTRE],data[:,i][FILTRE], linestyle='', xerr=error[:,j][FILTRE],yerr=error[:,i][FILTRE],ecolor='grey',alpha=0.7,marker='.',zorder=0)
    
    
                    if Outlier_emfa:
                        if k==12:
                            P.scatter(data[:,j][~self.dic['filter']],data[:,i][~self.dic['filter']],marker='^',s=150,c='r',zorder=50,label='Outlier')
                        else:
                            P.scatter(data[:,j][~self.dic['filter']],data[:,i][~self.dic['filter']],marker='^',s=150,c='r',zorder=50)
                                   
                    if Zcolor is not None:
                        scat=P.scatter(data[:,j][FILTRE],data[:,i][FILTRE],c=Zcolor,edgecolor='none',cmap=(cmap),visible=True,zorder=100)
                        if k==4 or k==8 or k==12: 
                            cb=P.colorbar(scat,format='%.3f')
                            cb.set_label(Zname)                    
    
                    else:
                        D_bis=N.ones(len(data[:,j][FILTRE]))*N.corrcoef(data[:,j][FILTRE],data[:,i][FILTRE])[0,1]
                        scat=P.scatter(data[:,j][FILTRE],data[:,i][FILTRE],c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
            
                    if k in [1,2,3,4,5,6,7,8]:
                        P.xticks([min(data[:,j][FILTRE])-3.*N.std(data[:,j][FILTRE]),max(data[:,j][FILTRE])+3.*N.std(data[:,j][FILTRE])],['toto','pouet'])
                        
                    if k in [9,10,11,12]:
                        P.xlabel(self.SI_latex[j]+' ($\AA$)')
    
                    if j in [9,10,11,12]:
                        if k in [9,10,11,12]:
                            XA=N.linspace(min(data[:,j][FILTRE])-0.5*N.std(data[:,j][FILTRE]),max(data[:,j][FILTRE])+0.5*N.std(data[:,j][FILTRE]),3)
                            XB=[]
                            for xa in range(len(XA)):
                                XA[xa]=(int(XA[xa])/10)*10
                                XB.append('%i'%(XA[xa]))
                            P.xticks(XA,XB)
    
        
                    P.xlim(min(data[:,j][self.dic['filter']])-2*N.std(data[:,j][self.dic['filter']]),max(data[:,j][self.dic['filter']])+2*N.std(data[:,j][self.dic['filter']]))
                    P.ylim(min(data[:,i][self.dic['filter']])-3.5*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
                    k += 1
                    P.ylabel(self.SI_latex[i]+' ($\AA$)')     
                        
                i += 1 
    
            if Outlier_emfa:
                P.legend(loc=4)

            if Zcolor is None:
                fig.subplots_adjust(right=0.83)
                cbar_ax = fig.add_axes([0.85, 0.07, 0.02, 0.83])
                cb=fig.colorbar(scat,format='%.1f', cax=cbar_ax)
                cb.set_label(r'Pearson correlation coefficient')

                 
    
            P.suptitle('fonction de ' + self.SI_latex[j])
            if rep_save is not None:
                P.savefig(rep_save+key_save+'_12D_space_in_terms_of_'+self.SIsavefig[j]+'.pdf')

    def Compare_TO_SUGAR_parameter(self,SED_max='/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_for_phd/model_at_max_3_eigenvector_without_grey_without_MFR_problem_test_RV.pkl',SUGAR_parameter_pkl='/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_with_cosmology.pkl'):

        FILTRE=self.dic['filter']
        data=passage_error(self.dic['Norm_data'][FILTRE],self.dic['Norm_err'][FILTRE],self.dic['vec'],sub_space=3)
        error,covX=passage_error_error_sn(self.dic['Norm_err'][FILTRE],self.dic['vec'],3)


        dic_SUGAR=cPickle.load(open(SUGAR_parameter_pkl))
        dic_sed_max=cPickle.load(open(SED_max))
        Av_max=dic_sed_max['Av_cardelli']

        data_SUGAR=N.zeros(N.shape(data))
        error_SUGAR=N.zeros(N.shape(error))
        Av_SUGAR=N.zeros(len(Av_max))
        sn_name=N.array(self.dic['sn_name'])[self.dic['filter']]


        for i in range(len(sn_name)):
            data_SUGAR[i,0]=dic_SUGAR[sn_name[i]]['x1']
            data_SUGAR[i,1]=dic_SUGAR[sn_name[i]]['x2']
            data_SUGAR[i,2]=dic_SUGAR[sn_name[i]]['x3']
            Av_SUGAR[i]=dic_SUGAR[sn_name[i]]['Av']
            error_SUGAR[i]=N.sqrt(N.diag(dic_SUGAR[sn_name[i]]['cov_h'][2:,2:]))

        P.figure(figsize=(15,8))
        P.subplots_adjust(left=0.05, bottom=0.10, right=0.99, top=0.98)
        for i in range(3):
            RHO=N.corrcoef(data[:,i],data_SUGAR[:,i])[0,1]
            P.subplot(2,2,i+1)
            P.scatter(data[:,i],data_SUGAR[:,i],label=r'$\rho=%.2f$'%(RHO))
            P.plot(data[:,i],data[:,i],'r',linewidth=3)
            P.errorbar(data[:,i],data_SUGAR[:,i], linestyle='', xerr=error[:,i],yerr=error_SUGAR[:,i],ecolor='grey',alpha=0.7,marker='.',zorder=0)
            P.legend(loc=2)
            P.xlabel('$q_{%i}$ from spectral indicators @ max'%(i+1),fontsize=16)
            P.ylabel('$q_{%i}$ from SUGAR fitting'%(i+1),fontsize=16)

        RHO=N.corrcoef(Av_max,Av_SUGAR)[0,1]
        P.subplot(2,2,4)
        P.scatter(Av_max,Av_SUGAR,label=r'$\rho=%.2f$'%(RHO))
        P.plot(Av_max,Av_max,'r',linewidth=3)
        P.legend(loc=2)
        P.xlabel('$A_{V}$ from fit SED @ max',fontsize=16)
        P.ylabel('$A_{V}$ from SUGAR fitting',fontsize=16)


        P.figure()
        
        P.hist(Av_SUGAR,bins=20,color='r')
        P.ylabel('number of supernovae',fontsize=16)
        P.xlabel('$A_{V}$',fontsize=16)
        P.ylim(0,25)
	P.legend(loc=4)
  

    def plot_correlation_SALT2_param(self,META,EMFA=False,OTHER_vector=False,rep_save='../These_plot/plot_phd/',SAVE=True):

        if not EMFA:
            NSUBPLOT=7
            NSUBPLOT_WINDOW=[8,15,22]
            FILTRE=N.array([True]*len(self.dic['data']))
            #FILTRE=self.dic['filter']
            data=self.dic['data'][FILTRE]
            error=self.dic['error'][FILTRE]
        else:
            NSUBPLOT=5
            NSUBPLOT_WINDOW=[6,11,16]
            #FILTRE=N.array([True]*len(self.dic['data']))
            FILTRE=self.dic['filter']
            data=passage_error(self.dic['Norm_data'][FILTRE],self.dic['Norm_err'][FILTRE],self.dic['vec'],sub_space=5)
            error,covX=passage_error_error_sn(self.dic['Norm_err'][FILTRE],self.dic['vec'],5)
            if OTHER_vector:
                data=passage_error(self.dic['Norm_data'][FILTRE],self.dic['Norm_err'][FILTRE],self.dic['vec'],sub_space=10)
                error,covX=passage_error_error_sn(self.dic['Norm_err'][FILTRE],self.dic['vec'],10)
                data=data[:,5:]
                error=error[:,5:]


        META=cPickle.load(open(META))

        X1=N.zeros(len(data))
        C=N.zeros(len(data))
        X1_err=N.zeros(len(data))
        C_err=N.zeros(len(data))
        Mu=N.zeros(len(data))
        Mu_err=N.zeros(len(data))
        z=N.zeros(len(data))
        z_err=N.zeros(len(data))
        COV=N.zeros((len(data),2,2))
        data_salt2=N.zeros((len(data),2))

        for i in range(len(N.array(self.dic['sn_name'])[FILTRE])):
            sn=N.array(self.dic['sn_name'])[FILTRE][i]
            Mu[i]=META[sn]['salt2.RestFrameMag_0_B']-5.*N.log(d_l(META[sn]['host.zcmb'],SNLS=True))/N.log(10.)+5.
            Mu_err[i]=META[sn]['salt2.RestFrameMag_0_B.err']
            X1[i]=META[sn]['salt2.X1']
            C[i]=META[sn]['salt2.Color']
            X1_err[i]=META[sn]['salt2.X1.err']
            C_err[i]=META[sn]['salt2.Color.err']
            COV[i,0,0]=META[sn]['salt2.X1.err']**2
            COV[i,1,1]=META[sn]['salt2.Color.err']**2
            COV[i,0,1]=META[sn]['salt2.CovColorX1']
            COV[i,1,0]=META[sn]['salt2.CovColorX1']
            z_err[i]=META[sn]['host.zhelio.err']
            z[i]=META[sn]['host.zcmb']

        data_salt2[:,0]=X1
        data_salt2[:,1]=C
        dmz=5/N.log(10) * N.sqrt(z_err**2 + 0.001**2) / z

            
        HF=Multi.Multilinearfit(data_salt2,Mu,yerr=N.sqrt(Mu_err**2+dmz**2),xerr=None,covx=COV)
            
        HF.Multilinearfit(adddisp=True)
        HF.comp_stat()
        Residuals=HF.y_corrected-HF.M0

        Mu-=N.average(Mu, weights=1./Mu_err**2)

        Filtre=(Mu>3)
        print N.array(self.dic['sn_name'])[FILTRE][Filtre]
        
        fig=P.figure(figsize=(25,15))
        P.subplots_adjust(hspace=0.07,wspace = 0.05)

        D=N.linspace(-1,1,len(data[:,0]))
        for i in range(NSUBPLOT):
            
            P.subplot(4,NSUBPLOT,(i+1))

            cmap=P.cm.jet
            norm=(D-D.min())/(D-D.min()).max()
            col=cmap(norm)
            RHO=N.corrcoef(data[:,i],X1)[0,1]
            SIGMA=Statistics.correlation_significance(abs(RHO),len(X1),sigma=True)
            if i==2 and not EMFA:
                P.text(60,max(X1)+1,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
            else:
                P.text(N.min(data[:,i]),max(X1)+1,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
            
            D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],X1)[0,1])
            P.scatter(data[:,i],X1,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
            P.errorbar(data[:,i],X1, linestyle='', xerr=error[:,i],yerr=X1_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
            P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']):
                P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            else:
                P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  
            if i!=0:
                P.yticks([min(X1)-3.*N.std(X1),max(X1)+3.*N.std(X1)],['toto','pouet'])
            else:
                P.ylabel('$X_1$')

            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']):
                P.ylim(min(X1[self.dic['filter']])-2*N.std(X1[self.dic['filter']]),max(X1[self.dic['filter']])+2*N.std(X1[self.dic['filter']]))
            else:
                P.ylim(min(X1)-2*N.std(X1),max(X1)+2*N.std(X1))


            P.subplot(4,NSUBPLOT,(i+NSUBPLOT_WINDOW[0]))
            RHO=N.corrcoef(data[:,i],C)[0,1]
            SIGMA=Statistics.correlation_significance(RHO,len(C),sigma=True)
            if i==2 and not EMFA:
                P.text(60,0.51,r'$\rho=%.2f, \sigma=%.2f$'%(RHO,SIGMA))
            else:
                P.text(N.min(data[:,i]),0.51,r'$\rho=%.2f, \sigma=%.2f$'%(RHO,SIGMA))

            D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],C)[0,1])
            P.scatter(data[:,i],C,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
            P.errorbar(data[:,i],C, linestyle='', xerr=error[:,i],yerr=C_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
            P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']):
                P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            else:
                P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  

            if i!=0:
                P.yticks([min(C)-3.*N.std(C),max(C)+3.*N.std(C)],['toto','pouet'])
            else:
                P.ylabel('C')
            P.ylim(-0.3,0.6)

            print 'Color: %f'%(N.corrcoef(data[:,i][(C<0.1)],C[(C<0.1)])[0,1])
            print 'Sig: %f'%(Statistics.correlation_significance(N.corrcoef(data[:,i][(C<0.1)],C[(C<0.1)])[0,1],len(data[:,i][(C<0.1)]),sigma=True))
            print ''


            P.subplot(4,NSUBPLOT,(i+NSUBPLOT_WINDOW[1]))
            RHO=N.corrcoef(data[:,i],Mu)[0,1]
            SIGMA=Statistics.correlation_significance(RHO,len(Mu),sigma=True)
            if i==2 and not EMFA:
                P.text(60,2.2,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
            else:
                P.text(N.min(data[:,i]),2.2,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))

            D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],Mu)[0,1])
            P.scatter(data[:,i],Mu,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
            P.errorbar(data[:,i],Mu, linestyle='', xerr=error[:,i],yerr=Mu_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
            P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']):
                P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            else:
                P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  
            if i!=0:
                P.yticks([min(Mu)-3.*N.std(Mu),max(Mu)+3.*N.std(Mu)],['toto','pouet'])
            else:
                P.ylabel('$\Delta \mu_B$')
            P.ylim(-1,2.5)



            P.subplot(4,NSUBPLOT,(i+NSUBPLOT_WINDOW[2]))
            RHO=N.corrcoef(data[:,i],Residuals)[0,1]
            SIGMA=Statistics.correlation_significance(RHO,len(Residuals),sigma=True)
            if i == 2 and not EMFA:
                P.text(60,max(Residuals)+0.17,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
            else:
                P.text(N.min(data[:,i]),max(Residuals)+0.17,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))

            D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],Residuals)[0,1])
            scat=P.scatter(data[:,i],Residuals,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
            P.scatter(data[:,i],Residuals)
            P.errorbar(data[:,i],Residuals, linestyle='', xerr=error[:,i],yerr=N.sqrt(HF.y_error_corrected**2+Mu_err**2),ecolor='grey',alpha=0.7,marker='.',zorder=0)

            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']): 
                P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            else:
                P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  

            if i!=0:
                P.yticks([min(Residuals)-3.*N.std(Residuals),max(Residuals)+3.*N.std(Residuals)],['toto','pouet'])
            else:
                P.ylabel('$\Delta \mu_B + \\alpha X_1 - \\beta C$')
                
            if not EMFA and not len(data[:,i])==N.sum(self.dic['filter']):
                P.ylim(min(Residuals[self.dic['filter']])-2*N.std(Residuals[self.dic['filter']]),max(Residuals[self.dic['filter']])+2*N.std(Residuals[self.dic['filter']]))
            else:
                P.ylim(min(Residuals)-2*N.std(Residuals),max(Residuals)+2*N.std(Residuals))

            if not EMFA:
                P.xlabel(self.SI_latex[i]+' ($\AA$)')     
            else:
                if OTHER_vector:
                    P.xlabel('$q_{%i}$'%(i+6),fontsize=16)
                else:
                    P.xlabel('$q_{%i}$'%(i+1),fontsize=16)


        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, 0.07, 0.02, 0.83])
        cb=fig.colorbar(scat,format='%.1f', cax=cbar_ax)
        cb.set_label(r'Pearson correlation coefficient')

        if SAVE:
            if not EMFA:
                P.savefig(rep_save+'SALT2_param_VS_SI_first_part.pdf')
            else:
                P.savefig(rep_save+'SALT2_param_vs_EMFA.pdf')

        
        if not EMFA:

            fig2=P.figure(figsize=(25,15))
            P.subplots_adjust(hspace=0.07,wspace = 0.05)
            
            for j in range(6):
                i=j+7
                P.subplot(4,6,(j+1))

                RHO=N.corrcoef(data[:,i],X1)[0,1]
                SIGMA=Statistics.correlation_significance(RHO,len(X1),sigma=True)
                P.text(N.min(data[:,i]),max(X1)+1,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
                D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],X1)[0,1])
                P.scatter(data[:,i],X1,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
                P.errorbar(data[:,i],X1, linestyle='', xerr=error[:,i],yerr=X1_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
                P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
                else:
                    P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  
                if j!=0:
                    P.yticks([min(X1)-3.*N.std(X1),max(X1)+3.*N.std(X1)],['toto','pouet'])
                else:
                    P.ylabel('$X_1$')
                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.ylim(min(X1[self.dic['filter']])-2*N.std(X1[self.dic['filter']]),max(X1[self.dic['filter']])+2*N.std(X1[self.dic['filter']]))
                else:
                    P.ylim(min(X1)-2*N.std(X1),max(X1)+2*N.std(X1))


                P.subplot(4,6,(j+7))
                RHO=N.corrcoef(data[:,i],C)[0,1]
                SIGMA=Statistics.correlation_significance(RHO,len(C),sigma=True)
                P.text(N.min(data[:,i]),0.51,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
                D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],C)[0,1])
                P.scatter(data[:,i],C,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
                P.errorbar(data[:,i],C, linestyle='', xerr=error[:,i],yerr=C_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
                P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
                else:
                    P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  
                if j!=0:
                    P.yticks([min(C)-3.*N.std(C),max(C)+3.*N.std(C)],['toto','pouet'])
                else:
                    P.ylabel('C')
                P.ylim(-0.3,0.6)


                P.subplot(4,6,(j+13))
                RHO=N.corrcoef(data[:,i],Mu)[0,1]
                SIGMA=Statistics.correlation_significance(RHO,len(Mu),sigma=True)
                P.text(N.min(data[:,i]),2.2,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
                D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],Mu)[0,1])
                P.scatter(data[:,i],Mu,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
                P.errorbar(data[:,i],Mu, linestyle='', xerr=error[:,i],yerr=Mu_err,ecolor='grey',alpha=0.7,marker='.',zorder=0)
                P.xticks([min(data[:,i])-3.*N.std(data[:,i]),max(data[:,i])+3.*N.std(data[:,i])],['toto','pouet'])
                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
                else:
                    P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  
                if j!=0:
                    P.yticks([min(Mu)-3.*N.std(Mu),max(Mu)+3.*N.std(Mu)],['toto','pouet'])
                else:
                    P.ylabel('$\Delta \mu_B$')
                P.ylim(-1,2.5)
            
                print 'Color: %f'%(N.corrcoef(data[:,i][(C<0.1)],C[(C<0.1)])[0,1])
                print 'Sig: %f'%(Statistics.correlation_significance(N.corrcoef(data[:,i][(C<0.1)],C[(C<0.1)])[0,1],len(data[:,i][(C<0.1)]),sigma=True))
                print ''


                P.subplot(4,6,(j+19))
                RHO=N.corrcoef(data[:,i],Residuals)[0,1]
                SIGMA=Statistics.correlation_significance(RHO,len(Residuals),sigma=True)
                P.text(N.min(data[:,i]),max(Residuals)+0.17,r'$\rho=%.2f, \sigma=%.2f$'%((RHO,SIGMA)))
                D_bis=N.ones(len(X1))*(N.corrcoef(data[:,i],Residuals)[0,1])
                scat=P.scatter(data[:,i],Residuals,c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
                P.errorbar(data[:,i],Residuals, linestyle='', xerr=error[:,i],yerr=N.sqrt(HF.y_error_corrected**2+Mu_err**2),ecolor='grey',alpha=0.7,marker='.',zorder=0)

                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.xlim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
                else:
                    P.xlim(min(data[:,i])-2*N.std(data[:,i]),max(data[:,i])+2*N.std(data[:,i]))  

                if j!=0:
                    P.yticks([min(Residuals)-3.*N.std(Residuals),max(Residuals)+3.*N.std(Residuals)],['toto','pouet'])
                else:
                    P.ylabel('$\Delta \mu_B + \\alpha X_1 - \\beta C$')

                if i in [9,10,11,12]:
                    if not len(data[:,i])==N.sum(self.dic['filter']):
                        XA=N.linspace(min(data[:,i][self.dic['filter']])-0.5*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+0.5*N.std(data[:,i][self.dic['filter']]),3)
                    else:
                        XA=N.linspace(min(data[:,i])-0.5*N.std(data[:,i]),max(data[:,i])+0.5*N.std(data[:,i]),3)

                    XB=[]
                    for xa in range(len(XA)):
                        XA[xa]=(int(XA[xa])/10)*10
                        XB.append('%i'%(XA[xa]))
                    P.xticks(XA,XB)
                if not len(data[:,i])==N.sum(self.dic['filter']):
                    P.ylim(min(Residuals[self.dic['filter']])-2*N.std(Residuals[self.dic['filter']]),max(Residuals[self.dic['filter']])+2*N.std(Residuals[self.dic['filter']]))
                else:
                    P.ylim(min(Residuals)-2*N.std(Residuals),max(Residuals)+2*N.std(Residuals))
                P.xlabel(self.SI_latex[i]+' ($\AA$)')     

            fig2.subplots_adjust(right=0.83)
            cbar_ax = fig2.add_axes([0.85, 0.07, 0.02, 0.83])
            cb=fig2.colorbar(scat,format='%.1f', cax=cbar_ax)
            cb.set_label(r'Pearson correlation coefficient')
            if SAVE:
                P.savefig(rep_save+'SALT2_param_VS_SI_second_part.pdf')


    def plot_correlation_Host_properties(self,HOST_pkl,LOCAL_HOST_pkl,META_pkl,rep_save='../These_plot/plot_phd/',LOCAL=0,SN_list=None,EMFA=False,OTHER_vector=False,SAVE=False):

        """
        if LOCAL=0 local propertie --> Log10(Local_Luminosity_H_alpha)
        if LOCAL=1 local propertie --> Log10(Local_SFR)
        if LOCAL=2 local propertie --> Log10(Local_sSFR)

        """
        if LOCAL not in [0,1,2]:
            raise ValueError('LOCAL should be 0,1 or 2')
        
        import return_H_alpha_2013

        if not EMFA:
            NSUBPLOT=5
            data=self.dic['data']
            error=self.dic['error']
        else:
            NSUBPLOT=2
            data=passage_error(self.dic['Norm_data'],self.dic['Norm_err'],self.dic['vec'],sub_space=5)
            error,covX=passage_error_error_sn(self.dic['Norm_err'],self.dic['vec'],5)
            if OTHER_vector:
                data=passage_error(self.dic['Norm_data'],self.dic['Norm_err'],self.dic['vec'],sub_space=10)
                error,covX=passage_error_error_sn(self.dic['Norm_err'],self.dic['vec'],10)
                data=data[:,5:]
                error=error[:,5:]


        HOST=cPickle.load(open(HOST_pkl))
        LOCAL_HOST=cPickle.load(open(LOCAL_HOST_pkl))
        META=cPickle.load(open(META_pkl))

        Masse=N.zeros(len(data))
        Masse_minus=N.zeros(len(data))
        Masse_plus=N.zeros(len(data))

        Local_Masse=N.zeros(len(data))
        Local_Masse_minus=N.zeros(len(data))
        Local_Masse_plus=N.zeros(len(data))

        H_alpha=N.zeros(len(data))
        H_alpha_minus=N.zeros(len(data))
        H_alpha_plus=N.zeros(len(data))
        
        Filtre_M=N.array([True]*len(data))
        Filtre_H=N.array([True]*len(data))

        for i in range(len(self.dic['sn_name'])):
            sn=self.dic['sn_name'][i]
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
            
        self.Local_mass=Local_mass
        self.H_alpha=H_alpha
        self.H_alpha_minus=H_alpha_minus
        self.H_alpha_plus=H_alpha_plus
        self.Filtre_H=Filtre_H

        
        fig=P.figure(figsize=(25,15))
        P.subplots_adjust(hspace=0.1,wspace = 0.25)
        
        if LOCAL==0:
            XH_mean=38.35
            XH_min=36.5
            XH_max=40.5
        else:
            if LOCAL==2:
                cst=0.5
            else:
                cst=1
            XH_mean=N.median(H_alpha[Filtre_H])
            H_wRMS=H.comp_rms(H_alpha[Filtre_H], 10, err=False, variance=H_alpha_minus[Filtre_H]**2)
            XH_min=XH_mean-cst*H_wRMS
            XH_max=XH_mean+cst*H_wRMS

        MASS_SUP=(Masse[Filtre_M]>10)
        Xmass_SUP=N.linspace(10,14,10)
        Xmass_low=N.linspace(6,10,10)

        label_local=[r'$\log\left(\Sigma_{H_{\alpha}} [\mathrm{\mathsf{erg}} \ \mathrm{\mathsf{s}}^{-1} \ \mathrm{\mathsf{kpc}}^{-2}]  \right)$',
                     r'$\log\left(\mathrm{\mathsf{SFR}} [\mathrm{\mathsf{M}}_{\odot} \ \mathrm{\mathsf{yr}}^{-1} \ \mathrm{\mathsf{kpc}}^{-2}]  \right)$',
                     r'$\log\left(\mathrm{\mathsf{sSFR}} [\mathrm{\mathsf{yr}}^{-1}]  \right)$']

        
        H_SUP=(H_alpha[Filtre_H]>XH_mean)
        XH_low=N.linspace(XH_min,XH_mean,10)
        XH_SUP=N.linspace(XH_mean,XH_max,10)

        D=N.linspace(0,5,N.sum(Filtre_M))


        if SN_list is not None:
            #Filtre_M=N.array([True]*len(data))
            filtreM_list=N.array([False]*N.sum(Filtre_M))
            filtreM_not_list=N.array([False]*N.sum(Filtre_M))
            SNN=N.array(self.dic['sn_name'])[Filtre_M]
            SN_outlier=N.array(self.dic['sn_name'])[~self.dic['filter']]
            for i in range(len(SNN)):
                if SNN[i] in SN_list:
                    if SNN[i] not in SN_outlier: 
                        filtreM_list[i]=True
                else:
                    if SNN[i] not in SN_outlier: 
                        filtreM_not_list[i]=True

            filtreH_list=N.array([False]*N.sum(Filtre_H))
            filtreH_not_list=N.array([False]*N.sum(Filtre_H))
            SNN=N.array(self.dic['sn_name'])[Filtre_H]
            SN_outlier=N.array(self.dic['sn_name'])[~self.dic['filter']]
            for i in range(len(SNN)):
                if SNN[i] in SN_list:
                    if SNN[i] not in SN_outlier: 
                        filtreH_list[i]=True
                else:
                    if SNN[i] not in SN_outlier: 
                        filtreH_not_list[i]=True

        else:
            Masse_err=[Masse_minus[Filtre_M],Masse_plus[Filtre_M]]
            H_alpha_err=[H_alpha_minus[Filtre_H],H_alpha_plus[Filtre_H]]


        for i in range(len(data[0])):
            
            P.subplot(NSUBPLOT,3,(i+1))

            cmap=P.cm.YlOrRd
            norm=(D-D.min())/(D-D.min()).max()
            col=cmap(norm)

            if i==0:
                print 'Nombre supernova masse local = %i'%(N.sum(Filtre_M))

            M_SI_SUP=N.average(data[:,i][Filtre_M][MASS_SUP],weights=1./error[:,i][Filtre_M][MASS_SUP]**2)
            wRMS_sup=H.comp_rms(data[:,i][Filtre_M][MASS_SUP]-M_SI_SUP, 10, err=False, variance=error[:,i][Filtre_M][MASS_SUP]**2)
            M_SI_SUP=N.average(data[:,i][Filtre_M][MASS_SUP],weights=1./(error[:,i][Filtre_M][MASS_SUP]**2+wRMS_sup**2))
            M_SI_SUP_err=N.sqrt((1./N.sum(1./(error[:,i][Filtre_M][MASS_SUP]**2+wRMS_sup**2))))

            M_SI_low=N.average(data[:,i][Filtre_M][~MASS_SUP],weights=1./error[:,i][Filtre_M][~MASS_SUP]**2)
            wRMS_low=H.comp_rms(data[:,i][Filtre_M][~MASS_SUP]-M_SI_low, 10, err=False, variance=error[:,i][Filtre_M][~MASS_SUP]**2)
            M_SI_low=N.average(data[:,i][Filtre_M][~MASS_SUP],weights=1./(error[:,i][Filtre_M][~MASS_SUP]**2+wRMS_low**2))
            M_SI_low_err=N.sqrt((1./N.sum(1./(error[:,i][Filtre_M][~MASS_SUP]**2+wRMS_low**2))))

            pull=abs(M_SI_SUP-M_SI_low)/N.sqrt(M_SI_SUP_err**2+M_SI_low_err**2)  
            print 'pull = %f'%(pull)
            rho=N.corrcoef(data[:,i][Filtre_M],Masse[Filtre_M])[0,1]
            print 'corr coeff = %f'%(rho)
            print 'significance = %f'%(Statistics.correlation_significance(rho,N.sum(Filtre_M),sigma=True))
            print ''
            D_bis=N.ones(N.sum(Filtre_M))*pull
  
            X_SI=N.linspace(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]),100)


            if SN_list is not None:
                P.scatter(Masse[Filtre_M][filtreM_not_list],data[:,i][Filtre_M][filtreM_not_list],marker='^',facecolors='none',edgecolors='b',s=100,label='Bottom Tree')
                P.scatter(Masse[Filtre_M][filtreM_list],data[:,i][Filtre_M][filtreM_list],marker=',',facecolors='none',edgecolors='r',s=100,label='Top Tree')
                P.errorbar(Masse[Filtre_M][filtreM_not_list],data[:,i][Filtre_M][filtreM_not_list],
                           linestyle='', yerr=error[:,i][Filtre_M][filtreM_not_list],
                           xerr=[Masse_minus[Filtre_M][filtreM_not_list],Masse_plus[Filtre_M][filtreM_not_list]]
                           ,ecolor='grey',alpha=0.9,marker='',zorder=0)
                P.errorbar(Masse[Filtre_M][filtreM_list],data[:,i][Filtre_M][filtreM_list],
                           linestyle='', yerr=error[:,i][Filtre_M][filtreM_list],
                           xerr=[Masse_minus[Filtre_M][filtreM_list],Masse_plus[Filtre_M][filtreM_list]]
                           ,ecolor='grey',alpha=0.9,marker='',zorder=0)

            else:
                P.text(max(Xmass_low),max(data[:,i][self.dic['filter']])+0.9*N.std(data[:,i][self.dic['filter']]),'$\sigma=%.3f$'%(pull),fontsize=14)
                scat=P.scatter(Masse[Filtre_M],data[:,i][Filtre_M],c=D_bis,edgecolor='none',cmap=(cmap),vmin=0,vmax=5,visible=True)
                P.errorbar(Masse[Filtre_M],data[:,i][Filtre_M],linestyle='', yerr=error[:,i][Filtre_M],xerr=Masse_err,ecolor='grey',alpha=0.9,marker='.',zorder=0)
            P.plot(10*N.ones(len(X_SI)),X_SI,'k-.',linewidth=2)
            P.plot(Xmass_low,N.ones(len(Xmass_low))*M_SI_low,'b')
            P.fill_between(Xmass_low,N.ones(len(Xmass_low))*M_SI_low-M_SI_low_err,N.ones(len(Xmass_low))*M_SI_low+M_SI_low_err,color='b',alpha=0.5)
            P.plot(Xmass_SUP,N.ones(len(Xmass_SUP))*M_SI_SUP,'b')
            P.fill_between(Xmass_SUP,N.ones(len(Xmass_SUP))*M_SI_SUP-M_SI_SUP_err,N.ones(len(Xmass_SUP))*M_SI_SUP+M_SI_SUP_err,color='b',alpha=0.5)

            P.ylim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            if not EMFA:
                if i in [0,1,2,3,4,5,6,7,8,9]:
                    P.xticks([min(Masse[Filtre_M])-3.*N.std(Masse[Filtre_M]),max(Masse[Filtre_M])+3.*N.std(Masse[Filtre_M])],['toto','pouet'])
                else:
                    P.xlabel('$\log(M/M_{\odot})$',fontsize=16)
            else:
                if i in [0,1]:
                    P.xticks([min(Masse[Filtre_M])-3.*N.std(Masse[Filtre_M]),max(Masse[Filtre_M])+3.*N.std(Masse[Filtre_M])],['toto','pouet'])
                else:
                    P.xlabel('$\log(M/M_{\odot})$',fontsize=16)

            P.xlim(6,14)

            if not EMFA:
                P.ylabel(self.SI_latex[i]+' ($\AA$)')     
            else:
                if OTHER_vector:
                    P.ylabel('$q_%i$'%(i+6),fontsize=16)
                else:
                    P.ylabel('$q_%i$'%(i+1),fontsize=16)

        if SN_list is None:
            fig.subplots_adjust(right=0.83)
            cbar_ax = fig.add_axes([0.85, 0.07, 0.02, 0.83])
            cb=fig.colorbar(scat,format='%.1f', cax=cbar_ax)
            cb.set_label(r'Significance of two distributions ($\sigma$)')
        else:
            leg=P.legend(bbox_to_anchor = (2., 0.5),fancybox=True, shadow=False)
            leg.get_frame().set_alpha(0.0)



        if SAVE:
            P.savefig(rep_save+'Host_param_VS_SI_first_part.pdf')

        fig2=P.figure(figsize=(25,15))
        P.subplots_adjust(hspace=0.1,wspace = 0.25)

        D=N.linspace(0,5,N.sum(Filtre_M))

        for i in range(len(data[0])):
            
            P.subplot(NSUBPLOT,3,(i+1))
                        
            cmap=P.cm.YlOrRd
            norm=(D-D.min())/(D-D.min()).max()
            col=cmap(norm)

            if i==0:
                print 'Nombre supernova masse local = %i'%(N.sum(Filtre_H))
                print 'medianne = %f'%(max(XH_low))
            
            H_SI_SUP=N.average(data[:,i][Filtre_H][H_SUP],weights=1./error[:,i][Filtre_H][H_SUP]**2)
            wRMS_sup=H.comp_rms(data[:,i][Filtre_H][H_SUP]-H_SI_SUP, 10, err=False, variance=error[:,i][Filtre_H][H_SUP]**2)
            H_SI_SUP=N.average(data[:,i][Filtre_H][H_SUP],weights=1./(error[:,i][Filtre_H][H_SUP]**2+wRMS_sup**2))
            H_SI_SUP_err=N.sqrt((1./N.sum(1./(error[:,i][Filtre_H][H_SUP]**2+wRMS_sup**2))))


            H_SI_low=N.average(data[:,i][Filtre_H][~H_SUP],weights=1./error[:,i][Filtre_H][~H_SUP]**2)
            wRMS_low=H.comp_rms(data[:,i][Filtre_H][~H_SUP]-H_SI_low, 10, err=False, variance=error[:,i][Filtre_H][~H_SUP]**2)
            H_SI_low=N.average(data[:,i][Filtre_H][~H_SUP],weights=1./(error[:,i][Filtre_H][~H_SUP]**2+wRMS_low**2))
            H_SI_low_err=N.sqrt((1./N.sum(1./(error[:,i][Filtre_H][~H_SUP]**2+wRMS_low**2))))

            pull=abs(H_SI_SUP-H_SI_low)/N.sqrt(H_SI_SUP_err**2+H_SI_low_err**2)  
            print 'pull = %f'%(pull)
            rho=N.corrcoef(data[:,i][Filtre_H],H_alpha[Filtre_H])[0,1]
            print 'corr coeff = %f'%(rho)
            print 'significance = %f'%(Statistics.correlation_significance(rho,N.sum(Filtre_H),sigma=True))
            print ''
            D_bis=N.ones(N.sum(Filtre_H))*pull
            
            X_SI=N.linspace(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]),100)

            if SN_list is not None:
                P.scatter(H_alpha[Filtre_H][filtreH_not_list],data[:,i][Filtre_H][filtreH_not_list],marker='^',facecolors='none',edgecolors='b',s=100,label='Bottom Tree')
                P.scatter(H_alpha[Filtre_H][filtreH_list],data[:,i][Filtre_H][filtreH_list],marker=',',facecolors='none',edgecolors='r',s=100,label='Top Tree')
                P.errorbar(H_alpha[Filtre_H][filtreH_not_list],data[:,i][Filtre_H][filtreH_not_list],
                           linestyle='', yerr=error[:,i][Filtre_H][filtreH_not_list],
                           xerr=[H_alpha_minus[Filtre_H][filtreH_not_list],H_alpha_plus[Filtre_H][filtreH_not_list]]
                           ,ecolor='grey',alpha=0.9,marker='',zorder=0)
                P.errorbar(H_alpha[Filtre_H][filtreH_list],data[:,i][Filtre_H][filtreH_list],
                           linestyle='', yerr=error[:,i][Filtre_H][filtreH_list],
                           xerr=[H_alpha_minus[Filtre_H][filtreH_list],H_alpha_plus[Filtre_H][filtreH_list]]
                           ,ecolor='grey',alpha=0.9,marker='',zorder=0)
            else:
                P.text(max(XH_low),max(data[:,i][self.dic['filter']])+0.9*N.std(data[:,i][self.dic['filter']]),'$\sigma=%.3f$'%(pull),fontsize=14)
                scat=P.scatter(H_alpha[Filtre_H],data[:,i][Filtre_H],c=D_bis,edgecolor='none',cmap=(cmap),vmin=0,vmax=5,visible=True)
                P.errorbar(H_alpha[Filtre_H],data[:,i][Filtre_H],linestyle='', yerr=error[:,i][Filtre_H],xerr=H_alpha_err,ecolor='grey',alpha=0.9,marker='.',zorder=0)

            P.plot(XH_mean*N.ones(len(X_SI)),X_SI,'k-.',linewidth=2)
            P.plot(XH_low,N.ones(len(XH_low))*H_SI_low,'b')
            P.fill_between(XH_low,N.ones(len(XH_low))*H_SI_low-H_SI_low_err,N.ones(len(XH_low))*H_SI_low+H_SI_low_err,color='b',alpha=0.5)
            P.plot(XH_SUP,N.ones(len(XH_SUP))*H_SI_SUP,'b')
            P.fill_between(XH_SUP,N.ones(len(XH_SUP))*H_SI_SUP-H_SI_SUP_err,N.ones(len(XH_SUP))*H_SI_SUP+H_SI_SUP_err,color='b',alpha=0.5)
            P.ylim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))

            if not EMFA:
                if i in [0,1,2,3,4,5,6,7,8,9]:
                    P.xticks([min(H_alpha[Filtre_H])-3.*N.std(H_alpha[Filtre_H]),max(H_alpha[Filtre_H])+3.*N.std(H_alpha[Filtre_H])],['toto','pouet'])
                else:
                    P.xlabel(label_local[LOCAL],fontsize=16)
            else:
                if i in [0,1]:
                    P.xticks([min(H_alpha[Filtre_H])-3.*N.std(H_alpha[Filtre_H]),max(H_alpha[Filtre_H])+3.*N.std(H_alpha[Filtre_H])],['toto','pouet'])
                else:
                    P.xlabel(label_local[LOCAL],fontsize=16)

            P.xlim(XH_min,XH_max)
            if not EMFA:
                P.ylabel(self.SI_latex[i]+' ($\AA$)')     
            else:
                if OTHER_vector:
                    P.ylabel('$q_%i$'%(i+6),fontsize=16)
                else:
                    P.ylabel('$q_%i$'%(i+1),fontsize=16)

        if SN_list is None:
            fig2.subplots_adjust(right=0.83)
            cbar_ax = fig2.add_axes([0.85, 0.07, 0.02, 0.83])
            cb=fig2.colorbar(scat,format='%.1f', cax=cbar_ax)
            cb.set_label(r'Significance of two distributions ($\sigma$)')
        else:
            leg=P.legend(bbox_to_anchor = (2., 0.5),fancybox=True, shadow=False)
            leg.get_frame().set_alpha(0.0)

        if SAVE:
            P.savefig(rep_save+'Host_param_VS_SI_second_part.pdf')

        P.show()

        if SN_list is not None:
            P.figure()
            
            P.hist(Masse[Filtre_M][filtreM_not_list],histtype='stepfilled',alpha=0.5,color='b',label='Bottom Tree',bins=50)
            P.hist(Masse[Filtre_M][filtreM_list],histtype='stepfilled',alpha=0.5,color='r',label='Top Tree',bins=50)
            P.ylabel('number of supernovae')
            P.xlabel('$\log(M/M_{\odot})$',fontsize=16)
            P.xlim(6,14)
            P.legend()
            if SAVE:
                p.savefig(rep_save+'dist_Host_param_VS_SI_first_part.png')
            
            P.figure()
            P.hist(H_alpha[Filtre_H][filtreH_not_list],histtype='stepfilled',alpha=0.5,color='b',label='Bottom Tree',bins=40)
            P.hist(H_alpha[Filtre_H][filtreH_list],histtype='stepfilled',alpha=0.5,color='r',label='Top Tree',bins=40)
            P.ylabel('number of supernovae')
            P.xlabel(label_local[LOCAL],fontsize=16)
            P.xlim(XH_min,XH_max)
            P.legend()
            if SAVE:
                p.savefig(rep_save+'dist_Host_param_VS_SI_second_part.png')

        
    def plot_3D_in_2D_pca(self,Branch_and_Wang=True,Filtre=True,add_core=True,EDGE=False,HISTOGRAM=False,List_SN=None,Name_family=None):


        data=self.dic['Norm_data']
        error=self.dic['Norm_err']


        new_base=passage_error(data,error,self.dic['vec'],sub_space=5)
        new_err,covX=passage_error_error_sn(error,self.dic['vec'],5)

        if Filtre:
            FILTRE=self.dic['filter']
        else:
            FILTRE=N.array([True]*len(self.dic['filter']))

        self.new_base=new_base[FILTRE]
        self.new_err=new_err[FILTRE]
        self.covX=covX[FILTRE]
        sn_name=N.array(self.dic['sn_name'])[FILTRE]


        if Branch_and_Wang: 
            P.figure(figsize=(28,16))
        else:
            P.figure(figsize=(12,12))
        if not HISTOGRAM:
            P.subplots_adjust(bottom=0.10,right=1.1, top=1.1,hspace=0.01,wspace = 0.01)
        else:
            P.subplots_adjust(bottom=0.10,right=0.95, top=0.95,hspace=0.01,wspace = 0.01)
        
        I=[0,0,1,0,1,2,0,1,2,3]
        J=[1,2,2,3,3,3,4,4,4,4]
        pos=[6,11,12,16,17,18,21,22,23,24]
        
        for t in range(len(pos)):

            P.subplot(5,5,pos[t])
            i=I[t]
            j=J[t]
    
            X=self.new_base[:,i]
            Y=self.new_base[:,j]

            X_error=self.new_err[:,i]
            Y_error=self.new_err[:,j]
            Cov_XY=self.covX[:,i,j]


       
            #PlotEllipse(X,Y,X_error,Y_error,Cov_XY,P.gca(),color='grey',alpha=0.4)


             
            if Branch_and_Wang:
                Data=self.dic['data'][FILTRE]
                subfilterSS=branch_filter(Data[:,6],Data[:,5],0.,80.,-5.,30.)
                subfilterCN=branch_filter(Data[:,6],Data[:,5],80.,105.,-5.,30.)
                subfilterBL=branch_filter(Data[:,6],Data[:,5],105.,N.inf,-5.,30.)
                subfilterCL=branch_filter(Data[:,6],Data[:,5],0.,N.inf,30.,N.inf)
                filterNorm=wang_filter(Data[:,12],0.,11800.)
                filterHV=wang_filter(Data[:,12],11800.,N.inf)
            
                filtre_branch_wang=[subfilterSS,subfilterCN,subfilterBL,subfilterCL,filterNorm,filterHV]
                legend_branch_wang=['Branch SS','Branch CN','Branch BL','Branch CL','Wang Norm','Wang HV']
                sn_reject=N.array(self.dic['sn_name'])[~self.dic['filter']]
                for i in range(6):
                    print legend_branch_wang[i]
                    for SNNN in sn_reject:
                        if SNNN in N.array(self.dic['sn_name'])[filtre_branch_wang[i]]:
                            print SNNN
                    print ''


                marker_branch_wang=['^','o',',','D','o',',']
                color_branch_wang=['g','k','b','r','k','b']
                size_branch_wang=[50,50,50,50,80,80]
                zorder_branch_wang=[100,100,100,100,10,10]
                ALPHA=[1,1,1,1,0.5,0.5]
                for p in range(len(filtre_branch_wang)):
                #P.scatter(Data[:,12][filtre_branch_wang[p]],Data[:,6][filtre_branch_wang[p]],facecolors='none',edgecolors=color_branch_wang[p],s=50,marker=marker_branch_wang[p],label=legend_branch_wang[p])
                    if p==4:
                        continue

                    if p in [0,1,2]:
                        P.scatter(X[filtre_branch_wang[p]],Y[filtre_branch_wang[p]],facecolors='none',edgecolors=color_branch_wang[p],s=size_branch_wang[p],zorder=zorder_branch_wang[p],marker=marker_branch_wang[p],label=legend_branch_wang[p])
                    else:
                        P.scatter(X[filtre_branch_wang[p]],Y[filtre_branch_wang[p]],c=color_branch_wang[p],s=size_branch_wang[p],zorder=zorder_branch_wang[p],marker=marker_branch_wang[p],label=legend_branch_wang[p],alpha=ALPHA[p])

                    if p==2 or p==5:
                        print len(self.dic['sn_name']),len(filtre_branch_wang[p])
                        print N.array(self.dic['sn_name'])[filtre_branch_wang[p]]


                if not Filtre:
                    P.scatter(X[~self.dic['filter']],Y[~self.dic['filter']],c='grey',s=100,zorder=5,alpha=0.7,label='oultier emfa')
                        
                        

                if pos[t] in [6,11,16,21,22,23,24]:
                    P.xlabel('$q_{%i}$'%(i+1),fontsize=30)
                    if pos[t] in [22,23,24]:
                        P.yticks([min(Y)-3.*N.std(Y),max(Y)+3.*N.std(Y)],['toto','pouet'])
                    else:
                        P.ylabel('$q_{%i}$'%(j+1),fontsize=30)
                    if pos[t] in [6,11,16]:
                        P.xticks([min(X)-3.*N.std(X),max(X)+3.*N.std(X)],['toto','pouet'])
                else:
                    P.xticks([min(X)-3.*N.std(X),max(X)+3.*N.std(X)],['toto','pouet'])
                    P.yticks([min(Y)-3.*N.std(Y),max(Y)+3.*N.std(Y)],['toto','pouet'])


                P.xlim(min(X)-1.5,max(X)+1.5)
                P.ylim(min(Y)-1.5,max(Y)+1.5)


                #leg=P.legend(bbox_to_anchor = (0.9, 4.),fancybox=True, shadow=False)
                #leg.get_frame().set_alpha(0.0)


             
            if List_SN is not None:
                #,List_SN=None,Name_family=None
                filtre_branch_wang=[]
                for TT in range(len(List_SN)):
                    FF=N.array([False]*len(sn_name))
                    for sn in range(len(sn_name)):
                        if sn_name[sn] in List_SN[TT]:
                            FF[sn]=True
                    filtre_branch_wang.append(copy.deepcopy(FF))



                if add_core:
                    FF=N.array([True]*len(sn_name))
                    for TT in range(len(List_SN)):
                        for sn in range(len(sn_name)):
                            if sn_name[sn] in List_SN[TT]:
                                FF[sn]=False

                    filtre_branch_wang.append(copy.deepcopy(FF))
                    
                legend_branch_wang=[]
                
                if Name_family is not None:
                    for TT in range(len(List_SN)):
                        legend_branch_wang.append(Name_family[TT])
                else:
                    legend_branch_wang.append('Family %i'%(TT+1))

                if add_core:
                    legend_branch_wang.append('Core of the Tree')
                
                marker_branch_wang=['^','D','o']
                color_branch_wang=['b','r','grey']
                size_branch_wang=[50,50,50,50,80,80]
                zorder_branch_wang=[100,100,10,100,10,10]
 
                for p in range(len(filtre_branch_wang)):
                    if p in [2] or EDGE:
                        P.scatter(X[filtre_branch_wang[p]],Y[filtre_branch_wang[p]],facecolors='none',edgecolors=color_branch_wang[p],s=size_branch_wang[p],zorder=zorder_branch_wang[p],marker=marker_branch_wang[p],label=legend_branch_wang[p])
                    else:
                        P.scatter(X[filtre_branch_wang[p]],Y[filtre_branch_wang[p]],c=color_branch_wang[p],s=size_branch_wang[p],zorder=zorder_branch_wang[p],marker=marker_branch_wang[p],label=legend_branch_wang[p])

                if pos[t] in [6,11,16,21,22,23,24]:
                    P.xlabel('$q_{%i}$'%(i+1),fontsize=30)
                    if pos[t] in [22,23,24]:
                        P.yticks([min(Y)-3.*N.std(Y),max(Y)+3.*N.std(Y)],['toto','pouet'])
                    else:
                        P.ylabel('$q_{%i}$'%(j+1),fontsize=30)
                    if pos[t] in [6,11,16]:
                        P.xticks([min(X)-3.*N.std(X),max(X)+3.*N.std(X)],['toto','pouet'])
                else:
                    P.xticks([min(X)-3.*N.std(X),max(X)+3.*N.std(X)],['toto','pouet'])
                    P.yticks([min(Y)-3.*N.std(Y),max(Y)+3.*N.std(Y)],['toto','pouet'])


                P.xlim(min(X)-1.5,max(X)+1.5)
                P.ylim(min(Y)-1.5,max(Y)+1.5)

        leg=P.legend(bbox_to_anchor = (0.9, 4.),fancybox=True, shadow=False)
        leg.get_frame().set_alpha(0.0)
        
        if HISTOGRAM:
            HIST_pos=[1,7,13,19,25]
            for G in range(5):
                X=self.new_base[:,G]
                P.subplot(5,5,HIST_pos[G])
                for p in range(len(filtre_branch_wang)):
                    #P.hist(X[filtre_branch_wang[p]])
                    pdf_SI, bins_SI, patches_SI=P.hist(X[filtre_branch_wang[p]],histtype='stepfilled')
                    P.setp(patches_SI, 'facecolor', color_branch_wang[p], 'alpha', 0.7)

                if G!=4:
                    P.xticks([min(X)-3.*N.std(X),max(X)+3.*N.std(X)],['toto','pouet'])
                else:
                    P.xlabel('$q_{%i}$'%(G+1),fontsize=30)
                P.yticks([-5,100],['toto','pouet'])
                P.xlim(min(X)-1.5,max(X)+1.5)
                P.ylim(0,30)



        if Branch_and_Wang:
            P.figure()
            for p in range(4):
                if p==0 or p==2:
                    P.scatter(Data[:,6][filtre_branch_wang[p]],Data[:,5][filtre_branch_wang[p]],facecolors='none',edgecolors=color_branch_wang[p],s=50,marker=marker_branch_wang[p],label=legend_branch_wang[p])
                else:
                    P.scatter(Data[:,6][filtre_branch_wang[p]],Data[:,5][filtre_branch_wang[p]],c=color_branch_wang[p],s=50,marker=marker_branch_wang[p],label=legend_branch_wang[p])
            

            Data=self.dic['data'][~self.dic['filter']]
            P.scatter(Data[:,6],Data[:,5],c='grey',s=100,marker='o',label='oultier emfa')

            P.plot(N.linspace(10,210,10),N.ones(10)*30.,'k-.',linewidth=4)
            P.plot(N.ones(10)*80,N.linspace(-5,30,10),'k-.',linewidth=4)
            P.plot(N.ones(10)*105,N.linspace(-5,30,10),'k-.',linewidth=4)

            P.errorbar(self.dic['data'][:,6],self.dic['data'][:,5], linestyle='', xerr=self.dic['error'][:,6],yerr=self.dic['error'][:,5],ecolor='grey',alpha=0.7,marker='',zorder=0)

            P.ylabel(r'pEW Si II $\lambda$ 5972 $[\AA]$')
            P.xlabel(r'pEW Si II $\lambda$ 6355 $[\AA]$')
            P.ylim(-5,70)
            P.xlim(10,210)
            leg=P.legend(loc=2,fancybox=True)
            leg.get_frame().set_alpha(0.0)


            Data=self.dic['data'][self.dic['filter']]
            Data[:,12]=((6355.-Data[:,12])/6335.)*299792458*10.**(-3)
            P.figure()
            for i in range(2):
                p=i+4
                if i ==1:
                    P.scatter(Data[:,12][filtre_branch_wang[p]],Data[:,6][filtre_branch_wang[p]],c=color_branch_wang[p],s=50,marker=marker_branch_wang[p],label=legend_branch_wang[p])
                else:
                    P.scatter(Data[:,12][filtre_branch_wang[p]],Data[:,6][filtre_branch_wang[p]],facecolors='none',edgecolors=color_branch_wang[p],s=50,marker=marker_branch_wang[p],label=legend_branch_wang[p])
                    
            

            Data=self.dic['data'][~self.dic['filter']]
            Data[:,12]=((6355.-Data[:,12])/6335.)*299792458*10.**(-3)
            P.scatter(Data[:,12],Data[:,6],c='grey',s=100,label='oultier emfa')


            Data=self.dic['data']
            err=(self.dic['error'][:,12]/6335.)*299792458*10.**(-3)
            Data[:,12]=((6355.-Data[:,12])/6335.)*299792458*10.**(-3)
            P.errorbar(Data[:,12],Data[:,6], linestyle='', xerr=err,yerr=self.dic['error'][:,6],ecolor='grey',alpha=0.7,marker='',zorder=0)

            P.plot(N.ones(10)*11800,N.linspace(10,210,10),'k-.',linewidth=4)

            P.ylabel(r'pEW Si II $\lambda$ 6355 $[\AA]$')
            P.xlabel(r'V Si II $\lambda$ 6355 $ [\mathrm{\mathsf{km}} \ \mathrm{\mathsf{s}}^{-1} ]$')
            P.ylim(10,210)
            P.xlim(5000,17500)
            leg=P.legend(loc=2,fancybox=True)
            leg.get_frame().set_alpha(0.0)


    def plot_histo_SI_space(self,Name_save=None,N_vec=None,Plot_outlier=True,YMAX=60):
        
        data=self.dic['Norm_data']
        error=self.dic['Norm_err']


        COLOR1='b'
        COLOR2='r'

        if Plot_outlier:
            FILTRE=N.array([True]*len(data[:,0]))
        else:
            FILTRE=self.dic['filter']

        if N_vec is not None:
            new_base=passage_error(data[FILTRE],error[FILTRE],self.dic['vec'],sub_space=N_vec)
            self.reconstruct_data=N.zeros(N.shape(data))
            for sn in range(len(data[:,0][FILTRE])):
                for vec in range(N_vec):
                    self.reconstruct_data[sn]+=new_base[sn,vec]*self.dic['vec'][:,vec]
            


        P.figure(figsize=(28,14))
        P.subplots_adjust(hspace=0.35,wspace = 0.2)
        for k in range(13):
            if k>9:
                P.subplot(3,5,k+2)
            else:
                P.subplot(3,5,k+1)
                
            pdf_SI, bins_SI, patches_SI=P.hist(data[:,k][FILTRE],histtype='stepfilled')
            P.setp(patches_SI, 'facecolor', COLOR1, 'alpha', 0.7)

            if N_vec is not None:
                pdf_RSI, bins_RSI, patches_RSI=P.hist(self.reconstruct_data[:,k],histtype='stepfilled')
                P.setp(patches_RSI, 'facecolor', COLOR2, 'alpha', 0.7)

            if k+1 in [2,3,4,5,7,8,9,10,13,14]:
                P.yticks([-1,YMAX+2],['toto','pouet'])

            P.ylim(0,YMAX)
            P.xlabel('Normalized '+self.SI_latex[k])     

            if k+1 in [1,6,11]:
                P.ylabel('Number of supernovae')

                
        p1 = P.Rectangle((0, 0), 1, 1, fc=COLOR1,alpha=0.7)
        X_legend=[p1]
        Y_legend=['Observed distribution']
        X_leg=0
        if N_vec is not None:
            p2 = P.Rectangle((0, 0), 1, 1, fc=COLOR2,alpha=0.7)
            X_legend.append(p2)
            Y_legend.append('Reconstruct distribution (%i PC)'%(N_vec))
            X_leg+=0.2
        P.legend(X_legend, Y_legend,
                 bbox_to_anchor=(2.3+X_leg, 0.70),
                 fancybox=True, shadow=True)

        if Name_save is not None :
            P.savefig(Name_save)


    def plot_2D(self,i,j,Zcolor=None,Zname='Francis',rep_save=None,key_save='all',Plot_outlier=True,Outlier_emfa=False,INTERACTIF=False):

        data=self.dic['data']
        error=self.dic['error']

        if Plot_outlier:
            FILTRE=N.array([True]*len(data[:,0]))
        else:
            FILTRE=self.dic['filter']
        
        P.figure(figsize=(8,8))
        if i!=j:
            
            if Zcolor is not None:
                cmap=P.cm.Blues
                norm=(Zcolor-Zcolor.min())/(Zcolor-Zcolor.min()).max()
                col=cmap(norm)
                            
                                
            P.errorbar(data[:,j][FILTRE],data[:,i][FILTRE], linestyle='', xerr=error[:,j][FILTRE],yerr=error[:,i][FILTRE],ecolor='grey',alpha=0.7,marker='.',zorder=0)
            if Outlier_emfa:
                P.scatter(data[:,j][~self.dic['filter']],data[:,i][~self.dic['filter']],marker='^',s=150,c='r',zorder=50,label='Outlier')
                                   
            if Zcolor is not None:
                scat=P.scatter(data[:,j][FILTRE],data[:,i][FILTRE],c=Zcolor,edgecolor='none',cmap=(cmap),visible=True,zorder=100)
                cb=P.colorbar(scat,format='%.3f')
                cb.set_label(Zname)                    
    
            else:
                scat=P.scatter(data[:,j][FILTRE],data[:,i][FILTRE],zorder=100)
            
                                            
            P.xlabel(self.SI_latex[j]+' ($\AA$)')
            
            P.xlim(min(data[:,j][self.dic['filter']])-2*N.std(data[:,j][self.dic['filter']]),max(data[:,j][self.dic['filter']])+2*N.std(data[:,j][self.dic['filter']]))
            P.ylim(min(data[:,i][self.dic['filter']])-2*N.std(data[:,i][self.dic['filter']]),max(data[:,i][self.dic['filter']])+2*N.std(data[:,i][self.dic['filter']]))
            P.ylabel(self.SI_latex[i]+' ($\AA$)')     
                

            if Outlier_emfa:
                P.legend()

            if INTERACTIF:
                browser=MPL.PointBrowser(data[:,j][FILTRE], data[:,i][FILTRE],N.array(self.dic['sn_name'])[FILTRE],scat) 
                P.show()


    def plot_histo_chi2(self):
        
        #P.plot(S.stats.chi2.pdf(N.linspace(0,50,100),13))
        pdf, bins, patches =  P.hist(self.dic['chi2_empca'][self.dic['filter']],normed=False)
        P.cla()
        Filtre=copy.deepcopy(self.dic['filter'])


        SN=['PTF09dnp','SN2005cf','SN2006do','SN2007cq','SNF20070403-000','SNF20070714-007','SNF20070803-005','SNF20071021-000','SNF20080905-005','SNF20080913-031','SNF20080919-002']

        for i in range(len(self.dic['filter'])):
            if self.dic['sn_name'][i] in SN:
                Filtre[i]=True

        toto, tata, patches=P.hist(self.dic['chi2_empca'][Filtre],53,normed=False,histtype='stepfilled',label='Observed $\chi^2$')
        P.setp(patches, 'facecolor', 'b', 'alpha', 0.3)
        X=N.linspace(0,max(self.dic['chi2_empca']),500)
        P.plot(X,S.stats.chi2.pdf(X,13)*N.sum(pdf * N.diff(bins))*0.8,'b',linewidth=4,label='$\chi^2$ law (ndof=13)')
        print N.sum(pdf * N.diff(bins))
        P.xlabel('$\chi^2$',fontsize=14)
        P.ylabel('number of supernovae')
        P.plot([28.3,28.3],[0,43],linewidth=4,label='$3\sigma$ cut')
        #P.text(62., 20, 'Rejected',fontsize=20)
        P.ylim(0,20)
        P.legend()


    def plot_vector(self,other_vector=None,Comment=''):

        TT=1
        nsil =  ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
                 'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
                 'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
                 'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
                 'VSi II $\lambda$6355']

        for vec in range(13):
            if vec%5==0:
                P.figure(figsize=(15,15))
                P.subplots_adjust(hspace = 0.01,top=0.85)
                TT=1

            P.subplot(5,1,TT)
            P.plot(N.zeros(13),'k',linewidth=2)

            P.plot(self.dic['vec'][:,vec],'b',label=r'$\Lambda_{%i}$'%(vec+1))

            if other_vector is not None :  
                if self.dic['vec'][:,vec][0]*other_vector[:,vec][0]<0:
                    if vec ==8 or vec==9:
                        P.plot(other_vector[:,vec],'r',label=r'$\Lambda_{%i}$'%(vec+1)+Comment)
                    else:
                        P.plot(-other_vector[:,vec],'r',label=r'$\Lambda_{%i}$'%(vec+1)+Comment)
                else:
                    P.plot(other_vector[:,vec],'r',label=r'$\Lambda_{%i}$'%(vec+1)+Comment)

            if TT==5 or (vec>10 and TT%5==3) :
                rotation=45
                xstart, xplus, ystart = -0.004, 0.082 ,5.01
                xx=[0,0,0,0,0,0,0,0,0,0,0,0,0]
                if vec>10 and TT%5==3:
                    ystart-=2

                x = xstart
                toto=1
                for leg in nsil:
                    P.annotate(leg, (x+xx[toto-1],ystart), xycoords='axes fraction',
                               size='large', rotation=rotation, ha='left', va='bottom')

                    toto+=1
                    x+= xplus

            TT+=1
            #if vec==2:
            leg=P.legend(fancybox=True)
            leg.get_frame().set_alpha(0.0)
            #else:
            #    P.legend(loc=4)
            P.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],['', '', '', '','','', '', '', '', '','','',''])
            P.yticks([-0.5,0,0.5],['-0.5','0','0.5'])
            P.ylim(-1,1)



    def plot_corr_coeff_emfa_3_first(self):

        data=(self.dic['Norm_data'])[self.dic['filter']]
        err=(self.dic['Norm_err'])[self.dic['filter']]
       
        self.emfa_vec=self.dic['vec']
        
        new_base=passage_error(data,err,self.emfa_vec,sub_space=3)
        new_err,covX=passage_error_error_sn(err,self.emfa_vec,3)


        nsil =  ['EWCa II H&K', r'EWSi II $\lambda$4131', 'EWMg II',
                 'EWFe $\lambda$4800', 'EWS II W', 'EWSi II $\lambda$5972',
                 'EWSi II $\lambda$6355', 'EWO I $\lambda$7773', 'EWCa II IR',
                 'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
                 'VSi II $\lambda$6355']
        
  
        self.corr_vec1,corr_vec1e=N.zeros(len(nsil)),N.zeros(len(nsil))
        self.corr_vec2,corr_vec2e=N.zeros(len(nsil)),N.zeros(len(nsil))
        self.corr_vec3,corr_vec3e=N.zeros(len(nsil)),N.zeros(len(nsil))      

        X,Y = [[],[],[]], [[],[],[]]
        self.neff = [[],[],[],]

        for i in range(len(nsil)):
            
            self.corr_vec1[i],corr_vec1e[i]=Statistics.correlation_weighted(data[:,i],new_base[:,0], w=1./(err[:,i]*new_err[:,0]),error=True, symmetric=True)
            self.corr_vec2[i],corr_vec2e[i]=Statistics.correlation_weighted(data[:,i],new_base[:,1], w=1./(err[:,i]*new_err[:,1]),error=True, symmetric=True)
            self.corr_vec3[i],corr_vec3e[i]=Statistics.correlation_weighted(data[:,i],new_base[:,2], w=1./(err[:,i]*new_err[:,2]),error=True, symmetric=True)

 
            self.neff[0].append(Statistics.neff_weighted(1./(err[:,i]*new_err[:,0])))
            self.neff[1].append(Statistics.neff_weighted(1./(err[:,i]*new_err[:,1])))
            self.neff[2].append(Statistics.neff_weighted(1./(err[:,i]*new_err[:,2])))


            X[0].append(data[:,i])
            X[1].append(data[:,i])
            X[2].append(data[:,i])
      
            Y[0].append(new_base[:,0])
            Y[1].append(new_base[:,1])
            Y[2].append(new_base[:,2])

        
        cmap = P.matplotlib.cm.get_cmap('Blues',9)
    
        fig = P.figure(figsize=(14,5.5),dpi=100)
        ax = fig.add_axes([0.055,0.10,0.9,0.9])#fig.add_axes([0.07,-0.03,0.9,1.3])
        rotation=45
        xstart, xplus, ystart = 0.03, 0.0777 ,1.01
 
        cmap.set_over('c')
        bounds = [0, 1, 2, 3, 4, 5]
        norm = P.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                
        ylabels=[r'$vec_1$','$vec_2$',r'$vec_3$']
        self.corrs = [self.corr_vec1,self.corr_vec2,self.corr_vec3]

        for i,corr in enumerate(self.corrs):
            sig = N.array([Statistics.correlation_significance(N.abs(c),n, sigma=True) for c, n in zip(corr,self.neff[i])])
            Sig=copy.deepcopy(sig)
            sig /= bounds[-1]
            cols = cmap(sig)
            mat = [[[0.25,rho*0.25],[rho*0.25,0.25]] for rho in corr]
            MPL.errorellipses(ax, range(1, len(nsil)+1), [4-i]*len(corr),
                              mat, color=cols, alpha=1, **{'ec':'k'})
            for j,c in enumerate(corr):
                x = (X[i][j]-N.min(X[i][j]))/N.max(X[i][j]-N.min(X[i][j]))-0.5
                y = (Y[i][j]-N.min(Y[i][j]))/N.max(Y[i][j]-N.min(Y[i][j]))-0.5
                x += (j+1)
                y += -N.mean(y) + (4-i)
                esty = loess(x, y)
                isort = N.argsort(x)
                lkwargs = SP_set_kwargs({}, 'loess', c='b', alpha=0.7, ls='-', lw=1)
                if Sig[j]>4 and Sig[j]<5:
                    if c<0.9:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',ha='center',va='center',)
                    else:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',fontsize=9,ha='center',va='center',)
                
                else:
                    ax.annotate('%.2f'%c,(j+1,4-i),ha='center',va='center',)
        x = xstart
        toto=1
        for leg in nsil:
            ax.annotate(leg, (x,ystart), xycoords='axes fraction',
                        size='large', rotation=rotation, ha='left', va='bottom')
            if toto==12:
                toto+=1
                x+= 0.06
            else:
                toto+=1
                x+= xplus    
        
        ax.set_xticks([])
        ax.set_yticks([4,3,2])
        ax.set_yticklabels(ylabels, size='xx-large', rotation=90)
        ax.set_ylim(ymin=1.4,ymax=4.6)
        ax.set_xlim(xmin=0.4, xmax=len(nsil)+0.6)
        ax.set_aspect('equal', adjustable='box-forced', anchor='C')
        
        im = ax.imshow([[0,5]], cmap=cmap,
                       extent=None, origin='upper',
                       interpolation='none', visible=False)
        cax, kw = P.matplotlib.colorbar.make_axes(ax, orientation='horizontal',
                                                  pad=0.02) 
        cb = P.matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                                norm=norm,
                                                boundaries=bounds+[9],
                                                extend='max',
                                                ticks=bounds,
                                                spacing='proportional',
                                                orientation='horizontal')
        cb.set_label('Pearson correlation coefficient significance ($\sigma$)',
                     size='x-large')
    


def plot_twins(data_max,SN1,SN2):

    dic=cPickle.load(open(data_max))

    Y1=dic[SN1]['spectra']['Y']
    Y1_err=N.sqrt(dic[SN1]['spectra']['V'])
    
    Y2=dic[SN2]['spectra']['Y']
    Y2_err=N.sqrt(dic[SN2]['spectra']['V'])

    X=dic[SN1]['spectra']['X']


    cst=N.mean((Y1+Y2)/2.)

    P.figure(figsize=(10,8))

    P.subplots_adjust(hspace=0.1)
    P.subplot(2,1,1)
    
    if dic[SN1]['spectra']['phase_salt2']<2:
        DAY=' day'
    else:
        DAY=' days'

    P.plot(X,Y1-cst,'b',label=SN1+', SALT2.4 phase: %.1f'%(dic[SN1]['spectra']['phase_salt2'])+DAY)
    P.fill_between(X,Y1-Y1_err-cst,Y1+Y1_err-cst,color='b',alpha=0.5)

    P.plot(X,Y2-cst,'r',label=SN2+', SALT2.4 phase: %.1f'%(dic[SN2]['spectra']['phase_salt2'])+DAY)
    P.fill_between(X,Y2-Y2_err-cst,Y2+Y2_err-cst,color='r',alpha=0.5)
    
    P.ylabel('Mag AB + Const.')
    P.xticks([3000,10000],['toto','pouet'])
    P.ylim(min([min(Y1),min(Y2)])-0.7-cst,max([max(Y1),max(Y2)])+0.2-cst)
    
    P.xlim(X[0]-60,X[-1]+60)

    P.gca().invert_yaxis()
    P.legend()

    P.subplot(2,1,2)
    P.plot(X,Y1-Y2,'k')
    P.plot(X,N.zeros(len(X)),'k')
    P.fill_between(X,(Y1-Y2)-N.sqrt(Y2_err**2+Y1_err**2),(Y1-Y2)+N.sqrt(Y2_err**2+Y1_err**2),color='k',alpha=0.5)
    P.ylim(-max(abs(Y1-Y2))-0.2,max(abs(Y1-Y2))+0.2)
    P.xlim(X[0]-60,X[-1]+60)
    P.ylabel('difference (mag)')
    P.xlabel('wavelength [$\AA$]')


def plot_average_flux_space(dic_at_max,list_sn,list_sn_average=None,SN_to_compare=None,redshift_to_compare=None,LABEL='',COLOR='r',LABEL_compare=None):

        

    dic=cPickle.load(open(dic_at_max))

    if list_sn_average is None:
        sn_name=N.array(dic.keys())
    else:
        sn_name=list_sn_average

    FILTRE=N.array([True]*len(sn_name))
    FILTRE_list=N.array([False]*len(sn_name))
    Y=[]
    Y_err=[]


    for i in range(len(sn_name)):
        if i == 0:
            X=dic[sn_name[i]]['spectra']['X']
        Y.append(dic[sn_name[i]]['spectra']['Y_flux'])
        Y_err.append(N.sqrt(dic[sn_name[i]]['spectra']['V_flux']))
        if N.sum(N.isfinite(Y[i]))<len(X):
            FILTRE[i]=False

        if sn_name[i] in list_sn:
            FILTRE_list[i]=True

    if N.sum(FILTRE_list)!=len(list_sn):
        print 'Attention il manque des sn dans le dico ou le nom est errone dans la liste' 

    Y=N.array(Y)
    Y_err=N.array(Y_err)
    
    AVERAGE=N.average(Y[FILTRE],axis=0,weights=1./Y_err[FILTRE]**2)
    for i in range(len(sn_name)):
        if not FILTRE[i]:
            Y[i][~N.isfinite(Y[i])]=AVERAGE[~N.isfinite(Y[i])]
            Y_err[i][~N.isfinite(Y[i])]=10**15


    Y_average=N.average(Y,axis=0,weights=1./Y_err**2)
    Y_average_list=N.average(Y[FILTRE_list],axis=0,weights=1./Y_err[FILTRE_list]**2)

    if SN_to_compare is not None :
        Y_compare=[]
        FILTRE_SPLINE=N.array([True]*len(X))
        import scipy.interpolate as inter
        #P.figure()
        for i in range(len(SN_to_compare)):

            SN_compare=N.loadtxt(SN_to_compare[i])
            if redshift_to_compare is not None:
                SN_compare[:,0]=SN_compare[:,0]/(1.+redshift_to_compare[i])

                Spline=inter.InterpolatedUnivariateSpline(SN_compare[:,0],SN_compare[:,1])
                Filtre_spline=( (X>SN_compare[:,0][0]) & ((X<SN_compare[:,0][-1])) )
                Y_spline=Spline(X)
                cst_flux=N.mean(Y_average_list[Filtre_spline]/Y_spline[Filtre_spline])
                Y_compare.append(Y_spline*cst_flux)
                FILTRE_SPLINE=((FILTRE_SPLINE) & (Filtre_spline))
                #P.plot(X[Filtre_spline],cst_flux*Y_spline[Filtre_spline],'k')

        #P.plot(X,Y_average_list,'b')
        Y_compare=N.array(Y_compare)
        Y_spline=N.mean(Y_compare,axis=0)[FILTRE_SPLINE]
        Filtre_spline=FILTRE_SPLINE
        
        #P.figure()
        #P.plot(X[Filtre_spline],Y_spline,'k')
        #P.plot(X,Y_average_list,'b')
        #P.show()

    P.figure(figsize=(10,8))

    gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
    P.subplots_adjust(hspace=0.1)
    P.subplot(gs[0])

    P.plot(X,Y_average,'k',linewidth=3,label='All spectrum average')

    P.plot(X,Y_average_list,COLOR,linewidth=3,label=LABEL)
    
    if SN_to_compare is not None :
        P.plot(X[Filtre_spline],Y_spline,'k--',linewidth=4,label=LABEL_compare)

    P.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$] + Const.')
    P.xticks([3000,10000],['toto','pouet'])
    #P.ylim(min([min(Y1),min(Y2)])-0.7-cst,max([max(Y1),max(Y2)])+0.2-cst)
    
    P.xlim(X[0]-60,X[-1]+60)
    
    P.legend()

    P.subplot(gs[1])
    P.plot(X,N.ones(len(X)),'k')
    P.plot(X,Y_average_list/Y_average,'k',linewidth=3)
    #if SN_to_compare is not None :
    #    P.plot(X[Filtre_spline],Y_average_list[Filtre_spline]/(Y_spline),'k--',linewidth=3)

    if max(abs(Y_average_list/Y_average))<1:
        P.ylim(0,1+min(abs(Y_average_list/Y_average))+0.2)
    else:
        P.ylim(0,max(abs(Y_average_list/Y_average))+0.2)
    P.xlim(X[0]-60,X[-1]+60)
    P.ylabel('ratio')
    P.xlabel('wavelength [$\AA$]')


    Y_average_list=N.average(Y[FILTRE_list],axis=0,weights=1./Y_err[FILTRE_list]**2)


    P.figure(figsize=(12,8))

    P.subplots_adjust(hspace=0.1)
    cst=0
    for i in range(sum(FILTRE_list)):
        P.plot(X,Y[FILTRE_list][i]+cst,'b',linewidth=3)
        P.text(X[-1]+200,Y[FILTRE_list][i][-1]+cst,sn_name[FILTRE_list][i])
        cst+=0.2


    P.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$] + Const.')
    

    P.xlim(X[0]-60,X[-1]+1500)
    P.ylim(0,2.1)
    P.xlabel('wavelength [$\AA$]')
    


    fichier=open('for_snid.txt','w')

    for Bin in range(len(X)):
        
        fichier.write('%f    %.5E    %.5E \n'%((X[Bin],Y_average_list[Bin],0.001)))
        
    fichier.close()


    
def plot_phase_correlation(dic_at_max,dic_EMFA=None,TITLE=None):


    phase=[]

    nsil =  ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
             'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
             'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
             'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
             'VSi II $\lambda$6355']

    dic=cPickle.load(open(dic_at_max))    
    if dic_EMFA is None:
        sn_name=N.array(dic.keys())
        FILTRE=N.array([True]*len(sn_name))
        FILTRE_list=N.array([False]*len(sn_name))
        Y=[]
        Y_err=[]

        for i in range(len(sn_name)):
            phase.append(dic[sn_name[i]]['spectra']['phase_salt2'])
            Y.append(dic[sn_name[i]]['spectral_indicators'])
            Y_err.append(dic[sn_name[i]]['spectral_indicators_error'])
            if N.sum(N.isfinite(Y[i]))<len(Y[i]):
                FILTRE[i]=False

        Y=N.array(Y)[FILTRE]
        Y_err=N.array(Y_err)[FILTRE]
        phase=N.array(phase)[FILTRE]
        sn_name=sn_name[FILTRE]

    else:
        dic_emfa= cPickle.load(open(dic_EMFA))
        FILTRE=dic_emfa['filter']
        sn_name=N.array(dic_emfa['sn_name'])[FILTRE]
        Y=passage_error(dic_emfa['Norm_data'][FILTRE],dic_emfa['Norm_err'][FILTRE],dic_emfa['vec'],sub_space=5)
        Y_err,covX=passage_error_error_sn(dic_emfa['Norm_err'][FILTRE],dic_emfa['vec'],5)
        for i in range(len(sn_name)):
            phase.append(dic[sn_name[i]]['spectra']['phase_salt2'])

    if dic_EMFA is None:
        Number_line=3
        X_line=20
        Y_line=12
    else:
        X_line=25
        Y_line=5
        Number_line=1

    fig=P.figure(figsize=(25,Y_line))
    #fig=P.figure()
    P.subplots_adjust(hspace=0.1,wspace = 0.4)

    D=N.linspace(-1,1,len(phase))
    for i in range(len(Y[0])):

        if i>8:
            P.subplot(Number_line,5,i+2)
        else:
            P.subplot(Number_line,5,i+1)


        cmap=P.cm.jet
        norm=(D-D.min())/(D-D.min()).max()
        col=cmap(norm)


        D_bis=N.ones(len(phase))*N.corrcoef(Y[:,i],phase)[0,1]

        P.errorbar(phase,Y[:,i], linestyle='', xerr=None,yerr=Y_err[:,i],ecolor='grey',alpha=0.7,marker='.',zorder=0)
        scat=P.scatter(phase,Y[:,i],c=D_bis,edgecolor='none',cmap=(cmap),vmin=-1,vmax=+1,visible=True,zorder=100)
        
       

        if i in [0,1,2,3,5,6,7,8] and dic_EMFA is None :
            P.xticks([min(phase)-3.*N.std(phase),max(phase)+3.*N.std(phase)],['toto','pouet'])
        else:
            P.xlabel('SALT2 phase (days)')
        rho=N.corrcoef(Y[:,i],phase)[0,1]
        SIG=Statistics.correlation_significance(rho,len(Y[:,i]),sigma=True)
        P.text(min(phase)-1.,min(Y[:,i])-1.5*N.std(Y[:,i]),r'$\rho=%.3f$, $\sigma=%.3f$'%((rho,SIG)))
        P.xlim(min(phase)-1.5*N.std(phase),max(phase)+1.5*N.std(phase))
        P.ylim(min(Y[:,i])-2*N.std(Y[:,i]),max(Y[:,i])+2*N.std(Y[:,i]))
        if dic_EMFA is None:
            P.ylabel(nsil[i]+' $(\AA)$')
        else:
            P.ylabel('$q_%i$'%(i+1),fontsize=16)

    if TITLE is not None:

        P.suptitle(TITLE)

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.85, 0.07, 0.02, 0.83])
    cb=fig.colorbar(scat,format='%.1f', cax=cbar_ax)
    cb.set_label(r'Pearson correlation coefficient ($\rho$)')
    #cb=P.colorbar(fig,format='%.1f')




def plot_pf_corr_bis(data,err,new_base,new_err,split=None,SPLIT=False):

    if split!=None and not SPLIT:
        plot_pf_corr_bis(data,err,new_base[:,split:],new_err[:,split:],split=split,SPLIT=True)
        new_base=new_base[:,:split]
        new_err=new_err[:,:split]


    nsil =  ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
             'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
             'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
             'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
             'VSi II $\lambda$6355']

    dic_corr_vec={}
    dic_corr_vece={}
    neff=[]
    X=[]
    Y=[]
    
    for i in range(len(new_base[0])):
        dic_corr_vec.update({'corr_vec%i'%(i):N.zeros(len(nsil))})
        dic_corr_vece.update({'corr_vec%ie'%(i):N.zeros(len(nsil))})
        neff.append([])
        X.append([])
        Y.append([])
    
    for j in range(len(new_base[0])):
        for i in range(len(nsil)):
            dic_corr_vec['corr_vec%i'%(j)][i],dic_corr_vece['corr_vec%ie'%(j)][i]=Statistics.correlation_weighted(data[:,i],new_base[:,j], w=1./(err[:,i]*new_err[:,j]),error=True, symmetric=True)

            neff[j].append(Statistics.neff_weighted(1./(err[:,i]*new_err[:,j])))
            
            X[j].append(data[:,i])

            Y[j].append(new_base[:,j])


        
    cmap = P.matplotlib.cm.get_cmap('Blues',9)
    
    
    if SPLIT:
        fig = P.figure(figsize=(14,20.5*(len(new_base[0])/3.)**2),dpi=100)
        ax = fig.add_axes([0.05,-0.023,0.9,0.9])#fig.add_axes([0.07,-0.03,0.9,1.3])
    else:
        fig = P.figure(figsize=(14,5.5*(len(new_base[0])/3.)**2),dpi=100)
        ax = fig.add_axes([0.05,0.10,0.9,0.9])#fig.add_axes([0.07,-0.03,0.9,1.3])
    rotation=45
    xstart, xplus, ystart = 0.03, 0.0777 ,1.01
 
    cmap.set_over('c')
    bounds = [0, 1, 2, 3, 4, 5]
    norm = P.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                
    ylabels=[]
    corrs=[]
    Ticks=[]
    for j in range(len(new_base[0])):
        corrs.append(dic_corr_vec['corr_vec%i'%(j)])
        if SPLIT:
            ylabels.append(r'$vec_{%i}$'%(j+1+split))
        else:
            ylabels.append(r'$vec_{%i}$'%(j+1))
        Ticks.append(4-j)
        
    for i,corr in enumerate(corrs):
        sig = N.array([Statistics.correlation_significance(N.abs(c),n, sigma=True) for c, n in zip(corr,neff[i])])
        Sig=copy.deepcopy(sig)
        sig /= bounds[-1]
        cols = cmap(sig)
        mat = [[[0.25,rho*0.25],[rho*0.25,0.25]] for rho in corr]
        MPL.errorellipses(ax, range(1, len(nsil)+1), [4-i]*len(corr),
                              mat, color=cols, alpha=1, **{'ec':'k'})
        for j,c in enumerate(corr):
            x = (X[i][j]-N.min(X[i][j]))/N.max(X[i][j]-N.min(X[i][j]))-0.5
            y = (Y[i][j]-N.min(Y[i][j]))/N.max(Y[i][j]-N.min(Y[i][j]))-0.5
            x += (j+1)
            y += -N.mean(y) + (4-i)
            esty = loess(x, y)
            isort = N.argsort(x)
            lkwargs = SP_set_kwargs({}, 'loess', c='b', alpha=0.7, ls='-', lw=1)
            if Sig[j]>4 and Sig[j]<5:
                if c<0.9:
                    ax.annotate('%.2f'%c,(j+1,4-i),color='w',ha='center',va='center',)
                else:
                    ax.annotate('%.2f'%c,(j+1,4-i),color='w',fontsize=9,ha='center',va='center',)
                
            else:
                ax.annotate('%.2f'%c,(j+1,4-i),ha='center',va='center',)
    x = xstart
    toto=1
    for leg in nsil:
        ax.annotate(leg, (x,ystart), xycoords='axes fraction',
                        size='large', rotation=rotation, ha='left', va='bottom')
        if toto==12:
            toto+=1
            x+= 0.06
        else:
            toto+=1
            x+= xplus    
        
    ax.set_xticks([])

    ax.set_yticks(Ticks)
    ax.set_yticklabels(ylabels, size='xx-large', rotation=90)
    ax.set_ylim(ymin=4.4-len(new_base[0]),ymax=4.6)
    ax.set_xlim(xmin=0.4, xmax=len(nsil)+0.6)
    ax.set_aspect('equal', adjustable='box-forced', anchor='C')
        
    im = ax.imshow([[0,5]], cmap=cmap,
                   extent=None, origin='upper',
                   interpolation='none', visible=False)
    cax, kw = P.matplotlib.colorbar.make_axes(ax, orientation='horizontal',
                                                  pad=0.02) 
    cb = P.matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                            norm=norm,
                                            boundaries=bounds+[9],
                                            extend='max',
                                            ticks=bounds,
                                            spacing='proportional',
                                            orientation='horizontal')
    cb.set_label('Pearson correlation coefficient significance ($\sigma$)',
                 size='x-large')
    





if __name__=='__main__':

    liste_91T=['SNF20070803-005','SN2007cq','LSQ12fhe','PTF11bju','SNNGC2691','PTF11mkx','PTF13asv','SNF20060618-023']
    liste_je_sais_pas=['SNF20080913-031','PTF12iiq','SNF20070714-007','PTF10ops','PTF13ajv','PTF12dxm','SNF20061108-001','SNhunt89','SNF20061024-000','SNF20070403-000','SNF20080905-005','SNF20080919-002','SNF20080614-010']
    #liste_je_sais_pas_sans_3SN=['PTF10ops','PTF13ajv','PTF12dxm','SNF20061108-001','SNhunt89','SNF20061024-000','SNF20070403-000','SNF20080905-005','SNF20080919-002','SNF20080614-010']
    liste_des_4SN=['SNF20080913-031','PTF12iiq','SNF20070714-007','PTF13anh']
    liste_je_sais_pas_sans_4SN = ['PTF10ops','PTF13ajv','PTF12dxm','SNF20061108-001','SNhunt89','SNF20061024-000','SNF20070403-000','SNF20080905-005','SNF20080815-017','PTF13ayw','SNF20050729-002','PTF09dnp','SNBOSS38','SNNGC4076']
    rep_bg='/sps/snovae/user/leget/91_bg_at_max/'
    liste_Cfa_91bg=[rep_bg+'sn2000dk-20000921.flm']
    liste_Cfa_91T=[rep_bg+'sn1999gp-20000102.flm']
    
    T91T_new = ['SNF20070803-005','SN2007cq','LSQ12fhe','PTF11bju','SNNGC2691','PTF11mkx','PTF13asv','SNF20060618-023','PTF11qmo','SNF20080919-001','SNF20071108-021','SNF20060512-001','SNF20080909-030']
    b91bg_new = ['SNF20080913-031','PTF12iiq','SNF20070714-007','PTF10ops','PTF13ajv','PTF12dxm','SNF20061108-001','SNhunt89','SNF20061024-000','SNF20070403-000','SNF20080905-005','PTF13anh','SNF20080815-017','PTF13ayw','SNF20050729-002','PTF09dnp','SNBOSS38','SNNGC4076']

    bottom_group=['SNF20060526-003','SNF20070831-015','SNF20080803-000','SNF20080822-005','SNF20060618-014',
                  'PTF10ndc','SNF20070725-001','PTF11bgv','PTF12efn','PTF10qjq','SNNGC4424','SNF20080516-022',
                  'SNF20061108-004','SNF20080510-005','SNF20071108-021','LSQ12fxd','SNF20080821-000','SNF20080626-002',
                  'SNF20070712-003','PTF11qmo','SNF20061011-005','LSQ12dbr','SNF20080919-001','PTF09fox','SNF20080918-000',
                  'SNF20050821-007', 'PTF11mty', 'SNF20080512-010', 'SNF20060618-023', 'SNhunt46', 'SNF20060609-002',
                  'PTF12ena', 'SNF20080725-004', 'SNF20070506-006', 'PTF10icb', 'PTF10hmv', 'SNF20080522-011',
                  'PTF12evo', 'PTF11mkx', 'PTF12fuu', 'SNF20080514-002', 'SN2012cu', 'SNF20080919-000', 'SNF20080914-001',
                  'SNF20050729-002', 'SNF20050624-000', 'PTF12ikt', 'SN2010ex', 'SNF20080522-000', 'SN2005hj', 'SNF20080909-030',
                  'SNF20070330-024', 'SNF20080516-000', 'CSS130502_01', 'LSQ12hjm', 'SNF20070817-003', 'SNF20080918-002', 'SNF20070701-005',
                  'PTF12jqh', 'SNF20070427-001', 'SNF20080323-009', 'SN2005hc', 'SNF20080612-003', 'SNF20061021-003', 'SN2006cj',
                  'SNF20060512-001', 'SNF20080507-000', 'PTF11bju', 'SNF20061022-005', 'SNF20071015-000', 'SNPGC027923',
                  'SNF20060521-001', 'PTF13asv', 'PTF10wnm', 'SNF20070727-016', 'SNF20060621-015']

    top_group=['SNF20080802-006', 'PTF10xyt', 'CSS120424_01', 'SNF20080610-000', 'SNF20070818-001', 'SNF20060512-002',
               'SN2006ob', 'SNF20070417-002', 'SNF20070424-003', 'SNF20070531-011', 'PTF11cao', 'SNF20070806-026',
               'PTF12dxm', 'PTF10qyz', 'SNIC3573', 'SNF20080720-001', 'CSS110918_01', 'SNF20070630-006', 'PTF13ayw',
               'SNF20080510-001', 'SNF20060907-000', 'PTF10mwb', 'SNF20050728-006', 'SNF20080623-001', 'SNF20071003-016',
               'SNF20070810-004', 'SNF20080806-002', 'SNF20070820-000', 'SNF20060511-014', 'SN2010kg', 'PTF12grk',
               'PTF10ufj', 'PTF09foz', 'SNF20080810-001', 'SNF20080918-004', 'SNF20080920-000', 'SN2008ec', 'SNF20080714-008',
               'PTF11pdk', 'PTF12ghy', 'SN2007nq', 'SN2007kk', 'SNF20070403-001', 'SNF20061030-010', 'PTF10tce', 'SNF20080531-000',
               'SNF20060908-004', 'SNF20061111-002', 'SNF20080825-010', 'SN2006dm', 'PTF09dlc', 'PTF10wof', 'PTF13azs', 'PTF12eer',
               'SNF20070331-025', 'SNF20070717-003', 'SNF20061020-000', 'SN2007bd', 'PTF09dnl', 'SN2005ir', 'SNF20080620-000', 'PTF11bnx',
               'PTF11drz', 'SN2004ef', 'SNF20080717-000', 'SNNGC0927', 'SNF20070902-018', 'SNF20061108-001', 'SNNGC7589', 'SNF20061024-000',
               'PTF10zdk', 'SNF20060912-000', 'CSS110918_02', 'SNF20070802-000', 'SNF20080815-017', 'SNF20070902-021', 'SNF20080614-010',
               'SN2010dt']


    
    #dic=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO.pkl'))
    #SN=N.array(dic['sn_name'])[dic['filter']]
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',liste_je_sais_pas,list_sn_average=None,SN_to_compare=liste_Cfa_91bg,redshift_to_compare=[0.01743],LABEL='Top branch average',LABEL_compare='SN2000dk @ max (91bg)')
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',b91bg_new ,list_sn_average=None,SN_to_compare=liste_Cfa_91bg,redshift_to_compare=[0.01743],LABEL='Top branch average',LABEL_compare='SN2000dk @ max (91bg)')
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',liste_91T,list_sn_average=None,SN_to_compare=liste_Cfa_91T,redshift_to_compare=[0.026],LABEL='Bottom branch average',COLOR='b',LABEL_compare='SN1999gp @ max (91T)')
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',T91T_new,list_sn_average=None,SN_to_compare=liste_Cfa_91T,redshift_to_compare=[0.026],LABEL='Bottom branch average',COLOR='b',LABEL_compare='SN1999gp @ max (91T)')
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',bottom_group,list_sn_average=None,SN_to_compare=None,redshift_to_compare=None,LABEL='Core bottom branch average',COLOR='b',LABEL_compare=None)
    #plot_average_flux_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',top_group,list_sn_average=None,SN_to_compare=None,redshift_to_compare=None,LABEL='Core top branch average',COLOR='r',LABEL_compare=None)
    #[0.0104,0.01664,0.00212,0.01743],LABEL=' (top branch in the tree)')

    #dic=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_without_filter_CABALLO.pkl'))

    #plot_phase_correlation('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',dic_EMFA=None,TITLE='Windows=[-2.5,2.5]')
    #plot_phase_correlation('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI_window_5_days.pkl',dic_EMFA=None,TITLE='Windows=[-5,5]')
    #plot_phase_correlation('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl',dic_EMFA='/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO_without_MFR_issue.pkl',TITLE='Windows=[-2.5,2.5]')
    #PEMFA=plot_emfa_spectral_indicator_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO.pkl')
    PEMFA=plot_emfa_spectral_indicator_space('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO_without_MFR_issue.pkl')
    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=True, Filtre=False)
    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=False, Filtre=True,add_core=False,EDGE=True,List_SN=[bottom_group,top_group],Name_family=['Core bottom branch','Core top branch'] ) 

    #SN=N.array(PEMFA.dic['sn_name'])[PEMFA.dic['filter']]
    #DAYS=N.array(PEMFA.dic['DAYS'])[PEMFA.dic['filter']]

    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=False, Filtre=True,add_core=False,EDGE=True,List_SN=[SN[(DAYS<55250)],SN[(DAYS>55250)]],Name_family=['Without calibration problem','With calibration problem'],HISTOGRAM=True) 

    #PEMFA.Compare_TO_SUGAR_parameter()

    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=False, Filtre=False,List_SN=[liste_91T,liste_je_sais_pas],Name_family=['Bottom branch','Top branch'] )
    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=False, Filtre=False,List_SN=[T91T_new,b91bg_new],Name_family=['Bottom branch','Top branch'] )
    #PEMFA.plot_3D_in_2D_pca(Branch_and_Wang=False, Filtre=False,List_SN=[liste_des_4SN,liste_je_sais_pas_sans_4SN],Name_family=['Top LP Tree','Top Tree'] )
    #PEMFA.plot_correlation_SALT2_param('/sps/snovae/user/leget/CABALLO/META.pkl',rep_save='../These_plot/plot_phd/Chapitre5/',EMFA=True,OTHER_vector=False,SAVE=True)
    #PEMFA.plot_BIC()    
    #PEMFA.plot_correlation_Host_properties('/sps/snovae/user/leget/BEDELL/Host.pkl','/sps/snovae/user/leget/CABALLO/localhost_idr.pkl','/sps/snovae/user/leget/CABALLO/META.pkl',LOCAL=2,SN_list=None,rep_save='../These_plot/plot_phd/Chapitre5/',EMFA=True,OTHER_vector=False,SAVE=True)
    #PEMFA.plot_histo_SI_space(Name_save=None,N_vec=None,Plot_outlier=False)
    #for i in range(10):
      #  print i+1
     #   PEMFA.plot_histo_SI_space(Name_save='../These_plot/plot_phd/reconstruct_distribution_%i_eigenvector.pdf'%(i+1),N_vec=i+1,Plot_outlier=False)
    #PEMFA.plot_histo_SI_space(Name_save='../These_plot/plot_phd/distribution.pdf',Plot_outlier=False)
    #    P.close()
    #PEMFA.plot_vector(other_vector=dic['vec'],Comment=' with outlier') 
    #PEMFA.plot_2D(0,1,Zcolor=None,Zname='Francis',rep_save=None,key_save='all',Plot_outlier=True,Outlier_emfa=True,INTERACTIF=True)
    #PEMFA.plot_histo_chi2()
    #P.savefig('../chi2_emfa.pdf')
   #P#EMFA.plot_SI_space(Zcolor=None,Zname=r'$\chi^2$',rep_save='../These_plot/plot_phd/Chapitre4/',key_save='all',Plot_outlier=True,Outlier_emfa=False)
    ##PEMFA.plot_SI_space(Zcolor=None,Zname=r'$\chi^2$',rep_save='../These_plot/plot_phd/Chapitre4/',key_save='all_target_on_outlier',Plot_outlier=True,Outlier_emfa=True)
    #PEMFA.plot_SI_space(Zcolor=None,Zname=r'$\chi^2$',rep_save=None,key_save=None,Plot_outlier=True,Outlier_emfa=False)
    #PEMFA.plot_corr_coeff_emfa_3_first()
    #PEMFA.plot_corr_coeff_emfa(split=3)

    ## SALT2 twins

    # inner 1
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20050624-000','PTF12jqh')
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20080612-003','SNF20060908-004')
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20050729-002','SN2007kk')
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20061024-000','SNF20080512-010')
    # inner 2
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20060511-014','PTF10mwb')
    # inner 3
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','CSS130502_01','PTF11qmo')
    # inner 4
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20070727-016','SN2010ex')
    # inner 5
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_SI.pkl','SNF20060908-004','SNF20080612-003')



    ## SI twins 


    # inner 1
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SNF20061020-000','SNF20070902-018')
    #P.savefig('../These_plot/plot_phd/inner_1_SI_CABALLO_SNF20061020-000_SNF20070902-018.pdf')
    # inner 2
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SNF20080612-003','SN2010ex')
    #P.savefig('../These_plot/plot_phd/inner_2_SI_CABALLO_SNF20080612-003_SN2010ex.pdf')
    # inner 3
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','PTF11bju','SNNGC2691')
    #P.savefig('../These_plot/plot_phd/inner_3_SI_CABALLO_PTF11bju_SNNGC2691.pdf')
    # inner 4
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SNF20060618-014','SN2005hc')
    #P.savefig('../These_plot/plot_phd/inner_4_SI_CABALLO_SNF20060618-014_SN2005hc.pdf')
    # inner 5
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','CSS130502_01','SNhunt46')
    #P.savefig('../These_plot/plot_phd/inner_5_SI_CABALLO_CSS13050201_SNhunt46.pdf')
    #pas twins
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','PTF09dlc','SNF20061020-000')
    #P.savefig('../These_plot/plot_phd/pas_twins_SI_CABALLO_PTF09dlc_SNF20061020-000.pdf')
    #pas twins
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SN2007cq','SN2006cj')
    #P.savefig('../These_plot/plot_phd/pas_twins_SI_CABALLO_SN2007cq_SN2006cj.pdf')
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SNF20080914-001','SNF20070330-024')
    #P.savefig('../These_plot/plot_phd/pas_twins_SI_CABALLO_SNF20080914-001_SNF20070330-024.pdf')
    #plot_twins('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI.pkl','SNF20061030-010','SNF20060621-015')
    #P.savefig('../These_plot/plot_phd/pas_twins_SI_CABALLO_SNF20061030-010_SNF20060621-015.pdf')
