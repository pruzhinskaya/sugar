import numpy as N
import pylab as P
import cosmogp
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import copy
import scipy.interpolate as inter
import cPickle
from build_mag import mag
from ToolBox.Wrappers import SALT2model
from ToolBox import Statistics


def d_l(z,H_0=69.999999999999996,Omega_m =0.26900000000000002,SNLS=False):

    c=299792.458
    Ol= 0.73099999999999998
    Om= 0.26900000000000002
    h0= 0.69999999999999996
    w= -1.0609999999999999
    co=Cosmology.Cosmology(h=h0, Om=Om, Ol=Ol, w=w)
    dl=co.Dl(z)/Cosmology.PARSEC
    if SNLS:
        return dl # the law for a universe with dark energy and dark matter (for compare to Nico's code)                                                                     
    else:
        return (c/H_0)*(z+(1.-0.75*Omega_m)*z*z)*10**6 #for low redshift like snf data                                                                                       




def plot_filter_PF(WRITE=False):

    dic=cPickle.load(open('../sugar/data_input/spectra_snia.pkl'))
    Y=dic['PTF09dnl']['1']['Y_flux']
    X=dic['PTF09dnl']['1']['X']

    fig,ax1=P.subplots(figsize=(16,8))
    P.subplots_adjust(left=0.07, right=0.93,bottom=0.09,top=0.99)
    ax1.plot(X,Y,'k',linewidth=3,label='PTF09dnl @ max')
    #ax1.plot(X,-Y-5,'b',linewidth=4,label='B filter')
    ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
    ax1.set_xlim(3300,9200)
    ax1.set_ylim(0,0.65)
    ax1.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]',fontsize=20)
    ax1.legend(fontsize=20)
    U_PFL=N.linspace(3360,4048.2,1000)
    B_SNf=N.linspace(4048.2,4877.3,1000)
    V_SNf=N.linspace(4877.3,5876.3,1000)
    R_SNf=N.linspace(5876.3,7079.9,1311)
    I_PFL=N.linspace(7079.9,8530,931)

    OneB=N.ones(len(B_SNf))
    OneV=N.ones(len(V_SNf))
    OneR=N.ones(len(R_SNf))
    OneU=N.ones(len(U_PFL))
    OneI=N.ones(len(I_PFL))

    #A=SALT2model.Salt2Model() 
    #FF=A.filters['STANDARD']['B']

    ax2=ax1.twinx()
    #ax2.plot(FF.lbda,FF.flux,'b',linewidth=4,label='B filter')
    ax2.fill_between(U_PFL,0*OneU,OneU,color='purple',alpha=0.5 )
    ax2.fill_between(B_SNf,0*OneB,OneB,color='b',alpha=0.5 )
    ax2.fill_between(V_SNf,0*OneV,OneV,color='g',alpha=0.5 )
    ax2.fill_between(R_SNf,0*OneR,OneR,color='r',alpha=0.5 )
    ax2.fill_between(I_PFL,0*OneI,OneI,color='k',alpha=0.5 )
    ax2.set_ylim(0,1.15)
    ax2.set_ylabel('Filters transmission',fontsize=20)
    ax2.set_xlim(3300,9200)
    p1 = P.Rectangle((0, 0), 1, 1, fc="purple")
    p2 = P.Rectangle((0, 0), 1, 1, fc="b")
    p3 = P.Rectangle((0, 0), 1, 1, fc="g")
    p4 = P.Rectangle((0, 0), 1, 1, fc="r")
    p5 = P.Rectangle((0, 0), 1, 1, fc="k")

    ax2.legend([p1, p2, p3, p4, p5], ['$U_{S}$', '$B_{S}$','$V_{S}$','$R_{S}$','$I_{S}$'],fontsize=20,loc=4)
  
    
    P.show()

    if WRITE: 
        fichier=open('B_filter.dat','w')
        for i in range(len(FF.lbda)):
            fichier.write('%f %f \n'%((FF.lbda[i],FF.flux[i])))

        fichier.close()
        fichier=open('BD17.dat','w')
        for i in range(len(A.RefSpec.x)):
            fichier.write('%f %.8E \n'%((A.RefSpec.x[i],A.RefSpec.y[i])))
        fichier.close()
        
        

class build_SUGAR_ligh_curves:

    def __init__(self,SUGAR_asci,hyperparameters_dat):

        SUGAR=N.loadtxt(SUGAR_asci)
        Hyper=N.loadtxt(hyperparameters_dat)
        self.SUGAR_phase=SUGAR[:,0]
        self.SUGAR_wavelength=SUGAR[:,1]
        self.SUGAR_M0=SUGAR[:,2]
        self.SUGAR_alpha=SUGAR[:,3:6]
        self.SUGAR_red=SUGAR[:,6]
        self.Model=SALT2model.Salt2Model()
        self.hyper_l=Hyper[:,1]
        self.hyper_sigma=Hyper[:,3]

        
    def comp_SUGAR_in_mag(self,Q,Av,Grey):
        
        self.Y=copy.deepcopy(self.SUGAR_M0)
        self.Y+=Av*self.SUGAR_red
        self.Y+=Grey
        self.Y+=self.SUGAR_alpha[:,0]*Q[0]
        self.Y+=self.SUGAR_alpha[:,1]*Q[1]
        self.Y+=self.SUGAR_alpha[:,2]*Q[2]


    def align_time(self,New_time):

        self.Phase=New_time
        self.wavelength=N.zeros(190)
        self.SUGAR_Wavelength=N.zeros(190*len(self.Phase))
        Time=N.linspace(-12,42,19)
        DELTA=len(self.Phase)
        self.Y_new_binning=N.zeros(190*len(self.Phase))
        for Bin in range(190):


            #gp = cosmogp.gaussian_process(self.Y[Bin*19:(Bin+1)*19], N.linspace(-12,42,19), kernel='RBF1D',
            #                              y_err=None, diff=None, Mean_Y=self.Y[Bin*19:(Bin+1)*19],
            #                              Time_mean=N.linspace(-12,42,19), substract_mean=True)

            #gp.hyperparameters = [self.hyper_sigma[Bin],
            #                      self.hyper_l[Bin]]

            #gp.get_prediction(new_binning = self.Phase,COV=False)

            self.Y_new_binning[Bin*DELTA:(Bin+1)*DELTA]=cosmogp.mean.interpolate_mean_1d(N.linspace(-12,42,19),
                                                                                         self.Y[Bin*19:(Bin+1)*19],
                                                                                         self.Phase)
            self.SUGAR_Wavelength[Bin*DELTA:(Bin+1)*DELTA]=copy.deepcopy(self.SUGAR_wavelength[Bin*19:(Bin+1)*19][0])
            self.wavelength[Bin]=self.SUGAR_wavelength[Bin*19:(Bin+1)*19][0]

        reorder = N.arange(190*DELTA).reshape(190, DELTA).T.reshape(-1)
        self.Y_new_binning=self.Y_new_binning[reorder]
        self.SUGAR_Wavelength=self.SUGAR_Wavelength[reorder]
        
    def go_to_flux(self,ABmag0=48.59):

        Flux_nu=10**(-0.4*(self.Y_new_binning+ABmag0))
        f = self.SUGAR_Wavelength**2 / 299792458. * 1.e-10
        self.Flux=Flux_nu/f




    def build_light_curves(self,BEST_PART=True):

        
        U_PFL=N.linspace(3360,4048.2,1000)
        B_SNf=N.linspace(4048.2,4877.3,1000)
        V_SNf=N.linspace(4877.3,5876.3,1000)
        R_SNf=N.linspace(5876.3,7079.9,1311)
        I_PFL=N.linspace(7079.9,8530,931)


        self.B_max=-999
        self.BVR=N.zeros((5,len(self.Phase)))
        self.B_band=N.zeros(len(self.Phase))
        A=SALT2model.Salt2Model()


        for time in range(len(self.Phase)):
            print self.Phase[time]
            self.BVR[0,time]=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=B_SNf[0],lambda_max=B_SNf[-1],var=None, step=None,Model=self.Model,AB=False)[0]
            self.BVR[1,time]=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=V_SNf[0],lambda_max=V_SNf[-1],var=None, step=None,Model=self.Model,AB=False)[0]
            self.BVR[2,time]=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=R_SNf[0],lambda_max=R_SNf[-1],var=None, step=None,Model=self.Model,AB=False)[0]
            self.BVR[3,time]=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=U_PFL[0],lambda_max=U_PFL[-1],var=None, step=None,Model=self.Model,AB=False)[0]
            self.BVR[4,time]=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=I_PFL[0],lambda_max=I_PFL[-1],var=None, step=None,Model=self.Model,AB=False)[0]
            #self.B_band[time]=A.mag(self.wavelength,self.Flux[time*190:(time+1)*190], var=None, step=None, syst='STANDARD', filter='B')[0]

            #if self.Phase[time]==0:
            #    if not BEST_PART:
            #        self.B_max=A.mag(self.wavelength,self.Flux[time*190:(time+1)*190], var=None, step=None, syst='STANDARD', filter='B')
            #    else:
            #        self.B_max=mag(self.wavelength,self.Flux[time*190:(time+1)*190],lambda_min=6330,lambda_max=6600,var=None, step=None,Model=self.Model,AB=False)
                #print self.B_max


class MC_light_curve(build_SUGAR_ligh_curves):

    def __init__(self,SUGAR_asci,HYPERPARAMETER,Time_binning):

        build_SUGAR_ligh_curves.__init__(self,SUGAR_asci,HYPERPARAMETER)
        #self.SUGAR_change_phase(Time_binning)
        self.Time=Time_binning
        self.redshift=0.001
        


    def build_fixed_light_curves(self,q,AV,GREY):

        self.q=q
        self.AV=AV
        self.Grey=GREY

        self.comp_SUGAR_in_mag(q,AV,GREY)
        self.align_time(self.Time)
        self.go_to_flux()

        self.build_light_curves()
        self.BVR_err=N.ones(N.shape(self.BVR))*0.05


class SUGAR_parameter_photometry:

    def __init__(self,BVR,BVR_err,Time,redshift,SUGAR_model='/sps/snovae/user/leget/CABALLO/SUGAR_model.asci',
                 HYPER='/sps/snovae/user/leget/CABALLO/Prediction_GP/hyperparameters.dat'):

        Filtre=((Time>=-12) & (Time<=42))

        self.BVR=BVR[:,Filtre]
        self.BVR_err=BVR_err[:,Filtre]
        self.Time=Time[Filtre]
        self.redshift=redshift
        self.Model=SUGAR_model
        self.HYPER=HYPER

        self.SLC=build_SUGAR_ligh_curves(SUGAR_model,HYPER)
        

    def comp_chi2(self,q1,q2,q3,Av,Grey):
        Q=N.array([q1,q2,q3])
             
        self.SLC.comp_SUGAR_in_mag(Q,Av,Grey)
        self.SLC.align_time(self.Time)
        self.SLC.go_to_flux()

        self.SLC.build_light_curves()

        self.chi2=0.
        self.residu=self.BVR-self.SLC.BVR
        for i in range(3):
            self.chi2+=N.dot(self.residu[i],N.dot(N.diag(1./self.BVR_err[i]**2),self.residu[i].reshape(len(self.residu[i]),1)))


    def fit_SUGAR_parameter(self):

        
        def _comp_chi2(q1,q2,q3,Av,Grey):
            Q=N.array([q1,q2,q3])
            
            self.SLC.comp_SUGAR_in_mag(Q,Av,Grey)
            self.SLC.align_time(self.Time)
            self.SLC.go_to_flux()

            self.SLC.build_light_curves()

            self.chi2=0.
            residu=self.BVR-self.SLC.BVR
            for i in range(3):
                self.chi2+=N.dot(residu[i],N.dot(N.diag(1./self.BVR_err[i]**2),residu[i].reshape(len(residu[i]),1)))
                
            print 'q1 : ', q1, ' q2: ', q2, ' q3: ',q3 ,' Av: ',Av ,' grey: ', Grey,' chi2/dof: ', self.chi2/((len(residu[0])*3)-5.)
            

            return self.chi2


        Find_Sugar=minuit.Minuit(_comp_chi2)#, q1=0.1, q2=-0.1, q3=0.1, Av=-0.3, Grey=5.*N.log10(d_l(self.redshift,SNLS=True))-5.)
        name=['q1','q2','q3','Av','Grey']
        pars=[0.1, -0.1, 0.1, -0.3, 5.*N.log10(d_l(self.redshift,SNLS=True))-5. ]
        #pars=[0.0,0.0,0.0,0.0,0.0]
        Find_Sugar.values = dict(zip(name,pars))

        Find_Sugar.migrad()
        
        self.SUGAR_parameters=Find_Sugar.values
        
        self.SUGAR_parameters_Covariance=Find_Sugar.covariance


        self.SLC_result=build_SUGAR_ligh_curves(self.Model,self.HYPER)

        Q=N.array([self.SUGAR_parameters['q1'],self.SUGAR_parameters['q2'],self.SUGAR_parameters['q3']])
        self.SLC_result.comp_SUGAR_in_mag(Q,self.SUGAR_parameters['Av'],self.SUGAR_parameters['Grey'])
        self.SLC_result.align_time(self.Time)
        self.SLC_result.go_to_flux()

        self.SLC_result.build_light_curves()

        
    def plot_result(self):
        COLOR=['b','g','k']

        for i in range(3):
            P.plot(self.SLC_result.Phase,self.SLC_result.BVR[i],COLOR[i],linewidth=4)
            P.scatter(self.Time,self.BVR[i],c=COLOR[i],s=50)
        
        P.gca().invert_yaxis()



class plot_SUGAR_interactif:

    def __init__(self,SUGAR_model):

        self.SLC=build_SUGAR_ligh_curves(SUGAR_model)
        self.SLC.SUGAR_change_phase(N.linspace(-12,42,25))

    def plot_parameter(self,q1,q2,q3,Av,Grey):

        Q=N.array([q1,q2,q3])
        self.SLC.go_to_flux(Q,Av,Grey)
        self.SLC.build_light_curves()
        COLOR=['b','g','k']
        for i in range(3):
            P.plot(N.linspace(-12,42,25),self.SLC.BVR[i],COLOR[i],linewidth=4)
            
        P.gca().invert_yaxis()



    def plot_interactif(self):

        P.figure(figsize=(12,12))
        P1=P.subplot(111)

        self.plot_parameter(0,0,0,0,0)
        P.xlim(-13,43)
        MAX=N.max(self.SLC.BVR)
        MIN=N.min(self.SLC.BVR)
        P.ylim(MIN-2,MAX+2)
        P.gca().invert_yaxis()

        axcolor = 'lightgoldenrodyellow'

        axq1 = P.axes([0.1, 0.03, 0.3, 0.03], axisbg=axcolor)
        sq1 = Slider(axq1, 'q1', -10, 10, valinit=0)

        axq2 = P.axes([0.6, 0.03, 0.3, 0.03], axisbg=axcolor)
        sq2 = Slider(axq2, 'q2', -10, 10, valinit=0)
        
        P.subplot(111)

        def update(val):

            P.cla()
            P1=P.subplot(111)

            self.plot_parameter(sq1.val,sq2.val,0,0,0)
            P.xlim(-13,43)
            P.ylim(MIN-2,MAX+2)
            P.gca().invert_yaxis()

        sq1.on_changed(update)
        sq2.on_changed(update)
        P.show()



def plot_light_curve(SN,BVR_data,BVR_phase,BVR_model,BVR_model_phase,BVR_model_on_data):

    
        COLOR=['b','g','r','purple','k']
        cst=[0,-1,-2,2,-3]
        Label=['$B_{PFL}$','$V_{PFL} \ %i$'%(cst[1]),'$R_{PFL} \ %i$'%(cst[2])]
        P.figure(figsize=(12,16))
        gs = gridspec.GridSpec(6, 1,height_ratios=[6,1,1,1,1,1])
        P.subplots_adjust(left=0.07, right=0.99,bottom=0.05,top=0.97,hspace=0.001)
        P.subplot(gs[0])
        
        P.scatter(BVR_phase,BVR_data[4]+cst[4],c=COLOR[4],s=30,label='$I_{PFL} \ %i$'%(cst[4]))
        P.plot(BVR_model_phase,BVR_model[4]+cst[4],COLOR[4],linewidth=2)
        for i in range(3):
            #P.errorbar(BVR_phase,BVR_data[2-i]+cst[2-i],xerr=None,yerr=BVR_err[2-i],linestyle='',color=COLOR[2-i],marker='.',alpha=0.5,zorder=0)
            P.scatter(BVR_phase,BVR_data[2-i]+cst[2-i],c=COLOR[2-i],s=30,label=Label[2-i])
            P.plot(BVR_model_phase,BVR_model[2-i]+cst[2-i],COLOR[2-i],linewidth=2)

        P.scatter(BVR_phase,BVR_data[3]+cst[3],c=COLOR[3],s=30,label='$U_{PFL}\ + %i$'%(cst[3]))
        P.plot(BVR_model_phase,BVR_model[3]+cst[3],COLOR[3],linewidth=2)


        P.ylabel('BD17 mags')
        P.legend(loc=3)
        P.title(SN)
        P.xlim(-14,44)
        P.gca().invert_yaxis()
        
        RMS=[]
        NMAD=[]

        P.subplot(gs[5])
        P.ylim(-0.19,0.19)
        P.scatter(BVR_phase,BVR_data[3]-BVR_model_on_data[3],c=COLOR[3])
        P.plot(BVR_model_phase,N.zeros(len(BVR_model_phase)),COLOR[3],linewidth=2)
        P.yticks([-0.10,0.10],['-0.10','0.10'])
        P.xlim(-14,44)
        P.xlabel('Time (days)')
        P.gca().invert_yaxis()
        for i in range(3):
            P.subplot(gs[2+i])
            P.ylim(-0.19,0.19)
            #P.errorbar(BVR_phase,BVR_data[2-i]-BVR_model_on_data[2-i],
            #          xerr=None,yerr=BVR_err[2-i],linestyle='',color=COLOR[2-i],marker='.',alpha=0.5,zorder=0)
            P.scatter(BVR_phase,BVR_data[2-i]-BVR_model_on_data[2-i],c=COLOR[2-i])
            P.plot(BVR_model_phase,N.zeros(len(BVR_model_phase)),COLOR[2-i],linewidth=2)
            P.yticks([-0.10,0.10],['-0.10','0.10'])
            P.xlim(-14,44)
            if i ==1:
                P.ylabel('residuals (mag)')
                
            p1 = Rectangle((0, 0), 0, 0, alpha=0.0)
            Filtre=((BVR_phase>-12) & (BVR_phase<42))
            nMAD=0#Statistics.nMAD(BVR_data[2-i][Filtre]-BVR_model_on_data[2-i][Filtre],weights=1./BVR_err[2-i][Filtre]**2)
            #leg=P.legend([p1], ['RMS=%.4f'%(N.sqrt((1./sum(Filtre))*N.sum((BVR_data[2-i][Filtre]-BVR_model_on_data[2-i][Filtre])**2)))],
            #             bbox_to_anchor=(1.01, 0.42),
            #             fancybox=True, shadow=False)
            #leg=P.legend([p1], ['nMAD=%.4f'%(nMAD)],
            #             bbox_to_anchor=(1.01, 0.42),
            #             fancybox=True, shadow=False)
            RMS.append(N.sqrt((1./sum(Filtre))*N.sum((BVR_data[2-i][Filtre]-BVR_model_on_data[2-i][Filtre])**2)))
            NMAD.append(nMAD)
            #leg.get_frame().set_alpha(0)
            #leg.get_frame().set_edgecolor('white')
            P.gca().invert_yaxis()

        P.subplot(gs[1])
        P.ylim(-0.19,0.19)
        P.scatter(BVR_phase,BVR_data[4]-BVR_model_on_data[4],c=COLOR[4])
        P.plot(BVR_model_phase,N.zeros(len(BVR_model_phase)),COLOR[4],linewidth=2)
        P.yticks([-0.10,0.10],['-0.10','0.10'])
        P.xlim(-14,44)
        P.gca().invert_yaxis()
        

        return RMS,NMAD



def plot_qi_effectlight_curve(SUGAR_model,Hyper,X1=4.46,X2=3.69,X3=1.36,AV=0.30):

    x1=N.array([X1,0,0,0])
    x2=N.array([0,X2,0,0])
    x3=N.array([0,0,X3,0])
    Av=N.array([0,0,0,AV])
    Time=N.linspace(-12,42,30)
    #Time=N.linspace(-12,42,19)
    

    SN_model=MC_light_curve(SUGAR_model,Hyper,Time)
    SN_model.build_fixed_light_curves(N.array([0,0,0]),0,38)

    MEAN=copy.deepcopy(SN_model.BVR)
    
    COLOR=['b','g','r','purple','k']
    cst=[0,-2.1,-4,+2.3,-6]
    Label=['$B_{PFL} \ \pm 1 \sigma$','$V_{PFL} \ %i \ \pm 1 \sigma$'%(cst[1]),'$R_{PFL} \ %i \ \pm 1 \sigma$'%(cst[2]),'$U_{PFL} \ +%i \ \pm 1 \sigma$'%(cst[3]),'$I_{PFL} \ %i \ \pm 1 \sigma$'%(cst[4])]
    TITLE=['q1','q2','q3','Av']

    for par in range(4):

        SN_model.build_fixed_light_curves(N.array([x1[par],x2[par],x3[par]]),Av[par],38)
        BVR_sigma_plus=copy.deepcopy(SN_model.BVR)
        SN_model.build_fixed_light_curves(N.array([-x1[par],-x2[par],-x3[par]]),-Av[par],38)
        BVR_sigma_moins=copy.deepcopy(SN_model.BVR)

        #P.figure(figsize=(12,18))
        P.figure(figsize=(12,14))
        P.subplots_adjust(left=0.05, right=0.99,bottom=0.05,top=0.97,wspace=0.15)
        P.subplot(1,2,1)
        i=3
        P.fill_between(Time,BVR_sigma_moins[2-i]+cst[2-i],BVR_sigma_plus[2-i]+cst[2-i],color=COLOR[2-i],alpha=0.5 )
        P.plot(Time,MEAN[2-i]+cst[2-i],COLOR[2-i],linewidth=2,label=Label[2-i])
        for i in range(3):
            P.fill_between(Time,BVR_sigma_moins[2-i]+cst[2-i],BVR_sigma_plus[2-i]+cst[2-i],color=COLOR[2-i],alpha=0.5 )
            P.plot(Time,MEAN[2-i]+cst[2-i],COLOR[2-i],linewidth=2,label=Label[2-i])

        i=4
        P.fill_between(Time,BVR_sigma_moins[2-i]+cst[2-i],BVR_sigma_plus[2-i]+cst[2-i],color=COLOR[2-i],alpha=0.5 )
        P.plot(Time,MEAN[2-i]+cst[2-i],COLOR[2-i],linewidth=2,label=Label[2-i])

        P.ylabel('BD17 mags')
        P.legend(loc=3)
        P.xlim(-14,44)
        P.ylim(3.5,16.)
        P.gca().invert_yaxis()
        P.xlabel('Time (days)')
        P.title(TITLE[par]+' $\pm 1 \sigma$ light curves effect')


        P.subplot(1,2,2)
        #P.figure(figsize=(8,16))
        #P.subplots_adjust(left=0.07, right=0.97,bottom=0.05,top=0.97)

        #U-B
        P.plot(Time,MEAN[3]-MEAN[0],'purple',linewidth=2,label='$U_{PFL}-B_{PFL}$')
        P.fill_between(Time,BVR_sigma_moins[3]-BVR_sigma_moins[0],BVR_sigma_plus[3]-BVR_sigma_plus[0],color='purple',alpha=0.5 )
        #B-V
        P.plot(Time,MEAN[0]-MEAN[1],'b',linewidth=2,label='$B_{PFL}-V_{PFL}$')
        P.fill_between(Time,BVR_sigma_moins[0]-BVR_sigma_moins[1],BVR_sigma_plus[0]-BVR_sigma_plus[1],color='b',alpha=0.5 )
        #V-I
        P.plot(Time,MEAN[1]-MEAN[2],'g',linewidth=2,label='$V_{PFL}-R_{PFL}$')
        P.fill_between(Time,BVR_sigma_moins[1]-BVR_sigma_moins[2],BVR_sigma_plus[1]-BVR_sigma_plus[2],color='g',alpha=0.5 )
        #R-I
        P.plot(Time,MEAN[2]-MEAN[4],'r',linewidth=2,label='$R_{PFL}-I_{PFL}$')
        P.fill_between(Time,BVR_sigma_moins[2]-BVR_sigma_moins[4],BVR_sigma_plus[2]-BVR_sigma_plus[4],color='r',alpha=0.5 )

        P.ylabel('Color')
        P.legend(loc=4)
        P.xlim(-14,44)
        #P.ylim(3.5,16.)
        P.xlabel('Time (days)')
        P.title(TITLE[par]+' $\pm 1 \sigma$ colors effect')

        #P.savefig('../These_plot/plot_phd/Chapitre9/alpha%i_effect_LC.pdf'%(par+1))



def Make_movie(SUGAR_model,Hyperparameter,Q='1',Var=3):


    
    
    TIME=N.linspace(-12,42,120)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='SUGAR model %i factor variation'%(int(Q)), artist='Matplotlib',
                    comment='SUGAR model')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    Name_mp4='SUGAR_model_%i_factor_variation.mp4'%(int(Q))

    if Q in ['1','2','3']:

        Q=int(Q)-1
        BS=build_SUGAR_ligh_curves(SUGAR,Hyperparameter)
        AV=0
        GREY=0
        QQ=N.zeros(3)
        BS.comp_SUGAR_in_mag(QQ,AV,GREY)
        BS.align_time(TIME)
        MEAN=copy.deepcopy(BS.Y_new_binning)
        QQ[Q]=Var
        CST=N.mean(MEAN)

        BS.comp_SUGAR_in_mag(QQ,AV,GREY)
        BS.align_time(TIME)
        MEAN_plus=copy.deepcopy(BS.Y_new_binning)

        BS.comp_SUGAR_in_mag(-QQ,AV,GREY)
        BS.align_time(TIME)
        MEAN_minus=copy.deepcopy(BS.Y_new_binning)
        
        fig=plt.figure(figsize=(12,12))
        P.subplots_adjust(left=0.07, right=0.99,bottom=0.05,top=0.99,hspace=0.001)
        P1=plt.subplot(111)


        with writer.saving(fig, Name_mp4, len(TIME)):

            for t in range(len(TIME)):

                if t!=0:
                    plt.cla()
                    gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
                    P1=plt.subplot(111)
                    plt.cla()
                    

                plt.plot(BS.wavelength,MEAN[i*190:(i+1)*190]-CST,'b')
                plt.fill_between(BS.wavelength,MEAN[t*190:(t+1)*190]-CST,MEAN_plus[t*190:(t+1)*190]-CST,color='m',alpha=0.7 )
                plt.fill_between(BS.wavelength,MEAN[t*190:(t+1)*190]-CST,MEAN_minus[t*190:(t+1)*190]-CST,color='g',alpha=0.7)


                print Bin
                writer.grab_frame()


        
if __name__=='__main__':

    #dic=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_for_phd/SUGAR_validation_3_eigenvector_CABALLO_test_RV.pkl'))
    #data=dic['data']
    #x1=N.mean(data[:,0])+N.sqrt(N.var(data[:,0]))
    #x2=N.mean(data[:,1])+N.sqrt(N.var(data[:,1]))
    #x3=N.mean(data[:,2])+N.sqrt(N.var(data[:,2]))
    #print x1
    #print x2
    #print x3

    plot_qi_effectlight_curve('../sugar/data_output/SUGAR_model_v1.asci','../sugar/data_output/gaussian_process/gp_info.dat',X1=1.,X2=1.,X3=1.)    
    
    #plot_filter_PF(WRITE=False)
