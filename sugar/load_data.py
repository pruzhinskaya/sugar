import numpy as N
import scipy as S
import pylab as P
import cPickle
from _savitzky_golay import savgol_filter 
import copy
import pySnurp
import merged_spectrum as ms
from ToolBox import Astro,Cosmology
import scipy.interpolate as inter
import multilinearfit as M
import SnfMetaData
import mpl_to_delate as mpl
from build_mag import mag
from ToolBox.Wrappers import SALT2model 
from SUGAR_photometry_fitting import build_SUGAR_ligh_curves


class spec:

    def __init__(self,y,x,v,step):
        self.y=y
        self.x=x
        self.v=v
        self.step=step


def go_to_flux(X,Y,ABmag0=48.59):
    
    Flux_nu=10**(-0.4*(Y+ABmag0))
    f = X**2 / 299792458. * 1.e-10
    Flux_lambda=Flux_nu/f
    return Flux_lambda


def check_bounds(X,lmin,lmax,step):
    """check the bounds"""
    spec_min=max([N.min(X[sn]) for sn in range(len(X))])
    spec_max=min([N.max(X[sn]) for sn in range(len(X))])
    if spec_min > lmin:
        lmin=spec_min
    if spec_max<lmax+step:
        lmax=spec_max-step
    return lmin,lmax


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


class load_time_spectra:

    def __init__(self,meta):

        self.dicM=cPickle.load(open(meta))
        self.sn_name=self.dicM.keys()
        self.BAD=False

    def load_spectra(self):

        self.dic={}
        self.X=[]
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC={}
            ind=0
            for j,pause in enumerate(self.dicM[sn]['spectra'].keys()):
                print 'processing '+sn+' pause '+pause 
                      #Sndata = selector('target.name','obs.exp','idr.spec_B','idr.spec_R','target.mwebv','host.zhelio',structured=True)

                Spec=pySnurp.Spectrum(self.dicM[sn]['spectra'][pause]['idr.spec_merged'])
                Spec.deredden(self.dicM[sn]['target.mwebv'])
                Spec.deredshift(self.dicM[sn]['host.zhelio'])
                    
                if len(Spec.x)==2691:

                    SPEC.update({'%i'%(ind):{'Y':Spec.y,
                                             'X':Spec.x,
                                             'V':Spec.v,
                                             'step':Spec.step,
                                             'days':self.dicM[sn]['spectra'][pause]['obs.mjd'],
                                             'z_cmb':self.dicM[sn]['host.zcmb'],
                                             'z_helio':self.dicM[sn]['host.zhelio'],
                                             'z_err':self.dicM[sn]['host.zhelio.err'],
                                             'phase_salt2':self.dicM[sn]['spectra'][pause]['salt2.phase'],
                                             'pause':pause}})
                    self.X.append(Spec.x)
                    ind+=1

            self.dic.update({sn:SPEC})


    def build_resampled_specs(self,lmin=3200,lmax=8900,step=10,delta_lambda=0.005):
        
        self.dic_binned={}
        lmin,lmax=check_bounds(self.X,lmin,lmax,step=step)
        #rebinarray=N.arange(lmin,lmax+step,step)
        rebinarray=[lmin]
        i=0
        while rebinarray[i]<lmax:
            rebinarray.append(rebinarray[i]*delta_lambda+rebinarray[i])
            i+=1
        if rebinarray[i]>lmax:
            del rebinarray[i]

        rebinarray=N.array(rebinarray)    
        print rebinarray
        print "%i filters created"%(len(rebinarray)-1)

        
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic[sn])
            for j,pause in enumerate(self.dic[sn].keys()):
                print 'processing '+sn+' pause '+pause 
            
                Spec=spec(SPEC[pause]['Y'],SPEC[pause]['X'],SPEC[pause]['V'],SPEC[pause]['step'])
                SPEC[pause]['X'],SPEC[pause]['Y'],SPEC[pause]['V']=ms.rebin(Spec,rebinarray)

            self.dic_binned.update({sn:SPEC})


    def to_AB_mag(self):

        self.dic_AB={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_binned[sn])
            for j,pause in enumerate(self.dic_binned[sn].keys()):
                print 'processing '+sn+' pause '+pause
                SPEC[pause].update({'V_flux':copy.deepcopy(SPEC[pause]['V'])})
                SPEC[pause].update({'Y_flux_without_cosmology':copy.deepcopy(SPEC[pause]['Y'][(SPEC[pause]['X']>3340.)])})
                SPEC[pause]['V']=Astro.Coords.flbda2ABmag(SPEC[pause]['X'],SPEC[pause]['Y'],var=SPEC[pause]['V'])
                SPEC[pause]['Y']=Astro.Coords.flbda2ABmag(SPEC[pause]['X'],SPEC[pause]['Y'])

            self.dic_AB.update({sn:SPEC})


    def Cosmology_corrected(self):
        self.dic_cosmo={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_AB[sn])
            for j,pause in enumerate(self.dic_AB[sn].keys()):
                print 'processing '+sn+' pause '+pause
                SPEC[pause]['Y']+= -5.*N.log10(d_l(SPEC[pause]['z_cmb'],SNLS=True))+5.
                SPEC[pause].update({'Y_flux':go_to_flux(SPEC[pause]['X'],copy.deepcopy(SPEC[pause]['Y']))})    
 
 
            self.dic_cosmo.update({sn:SPEC})

            

    def reorder_and_clean_dico_cosmo(self,SALT2=True):
        dic_cosmo={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            spec={}
            Ind=0
            for j in range(len(self.dic_cosmo[sn].keys())):
                Min=(10.**23)
                ind='0'
                for k,pause in enumerate(SPEC):
                    phase=SPEC[pause]['phase_salt2']
                    if phase<Min:
                        Min=phase
                        ind=pause
                if N.sum(N.isfinite(SPEC[ind]['Y']))==len(SPEC[ind]['Y']):
                    SPEC[ind]['Y']=SPEC[ind]['Y'][(SPEC[ind]['X']>3340.)]
                    SPEC[ind]['V']=SPEC[ind]['V'][(SPEC[ind]['X']>3340.)]
                    SPEC[ind]['Y_flux']=SPEC[ind]['Y_flux'][(SPEC[ind]['X']>3340.)]
                    SPEC[ind]['V_flux']=SPEC[ind]['V_flux'][(SPEC[ind]['X']>3340.)]
                    SPEC[ind]['X']=SPEC[ind]['X'][(SPEC[ind]['X']>3340.)]
                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
                    Ind+=1
                else:
                    Delta=abs(N.sum(N.isfinite(SPEC[ind]['Y']))-len(SPEC[ind]['Y']))
                    if Delta<100:
                        Filtre=N.isfinite(SPEC[ind]['Y'])
                        SPLINE=inter.InterpolatedUnivariateSpline(SPEC[ind]['X'][Filtre],SPEC[ind]['Y'][Filtre])
                        SPEC[ind]['Y'][~Filtre]=SPLINE(SPEC[ind]['X'][~Filtre])
                        SPEC[ind]['V'][~Filtre]=100.
                        SPEC[ind]['Y']=SPEC[ind]['Y'][(SPEC[ind]['X']>3340.)]
                        SPEC[ind]['V']=SPEC[ind]['V'][(SPEC[ind]['X']>3340.)]
                        SPEC[ind]['Y_flux']=SPEC[ind]['Y_flux'][(SPEC[ind]['X']>3340.)]
                        SPEC[ind]['V_flux']=SPEC[ind]['V_flux'][(SPEC[ind]['X']>3340.)]
                        SPEC[ind]['X']=SPEC[ind]['X'][(SPEC[ind]['X']>3340.)]
                        spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
                        Ind+=1


                del SPEC[ind]


            dic_cosmo.update({sn:spec})
    
        self.dic_cosmo=dic_cosmo

        
    def kill_blue_runaway(self):
        X=self.dic_cosmo['PTF09dlc']['0']['X']
        Filtre_UV=N.array([False]*len(X))
        for i in range(len(Filtre_UV)):
            if X[i]>3700 and X[i]<3900:
                Filtre_UV[i]=True

        dic_cosmo={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            spec={}
            Ind=0
            for j in range(len(self.dic_cosmo[sn].keys())):
                x=X[Filtre_UV]
                y=SPEC['%i'%(j)]['Y'][Filtre_UV]
                y_err=N.sqrt(SPEC['%i'%(j)]['V'][Filtre_UV])
                Multi=M.Multilinearfit(x,y,xerr=None,yerr=None,covx=None,Beta00=None) 
                Multi.Multilinearfit(adddisp=False) 

                if Multi.alpha[0]<0 and N.mean(y)<-10.:
                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC['%i'%j])})
                    Ind+=1
                del SPEC['%i'%(j)]
            dic_cosmo.update({sn:spec})

        self.dic_cosmo=dic_cosmo


    def select_sn_in_good_sample(self,META):

        meta = SnfMetaData.SnfMetaData(META)
        
        meta.add_filter(idr__subset__in = ['training','validation'])#,'bad','auxiliary'])
        sn_name=meta.targets('target.name',sort_by='target.name')
        dic_cosmo={}
        for i,sn in enumerate(sn_name):
            print '%i/%i'%(((i+1),len(sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            #if len(SPEC.keys())>4:
            #    dic_cosmo.update({sn:SPEC})
            dic_cosmo.update({sn:SPEC})
            
        self.sn_name=dic_cosmo.keys()
        self.dic_cosmo=dic_cosmo



    def select_night_in_previous_dico(self,dico_pkl):
 
        dic = cPickle.load(open(dico_pkl))

        sn_name=dic.keys()
        dic_cosmo={}
        for i,sn in enumerate(self.sn_name):
            print sn 
            if sn in sn_name:
                SPEC=copy.deepcopy(self.dic_cosmo[sn])
                SPec={}
                PAUSE=[]
                IND=0
                for j in range(len(dic[sn].keys())):
                    PAUSE.append(dic[sn]['%i'%(j)]['pause'])
                for j in range(len(SPEC.keys())):
                    if SPEC['%i'%(j)]['pause'] in PAUSE:
                        SPec.update({'%i'%(IND):SPEC['%i'%(j)]})
                        IND+=1
                dic_cosmo.update({sn:SPec})
                if len(SPec.keys())!=len(dic[sn].keys()):
                    print 'ANDALOUSE ma gueule'

        self.sn_name=dic_cosmo.keys()
        self.dic_cosmo=dic_cosmo




    def kill_Blue_runaway_eye_control(self):
        X=self.dic_cosmo['PTF09dlc']['0']['X']
        dic_cosmo={}
        dic_bad={}
        self.BAD=True
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            OFFSET=[]

            P.figure()
            indice_spec=[]
            Wave=[]
            Flux=[]
            for j in range(len(self.dic_cosmo[sn].keys())):
                OFFSET.append(j+0.2+15)
                indice_spec.append(j)
                Wave.append(SPEC['%i'%(j)]['X'][60])
                Flux.append(SPEC['%i'%(j)]['Y'][60]+OFFSET[j])

                P.plot(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y']+OFFSET[j],'k')
                moins=SPEC['%i'%(j)]['Y']+OFFSET[j]-N.sqrt(SPEC['%i'%(j)]['V'])
                plus=SPEC['%i'%(j)]['Y']+OFFSET[j]+N.sqrt(SPEC['%i'%(j)]['V'])
                P.fill_between(SPEC['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                
            indice_spec=N.array(indice_spec)
            Wave=N.array(Wave)
            Flux=N.array(Flux)
            
            scat=P.scatter(Wave,Flux,s=50,c='b')
            browser=mpl.PointBrowser_TO_DELATE(Wave,Flux,indice_spec,scat)
            P.title(sn)
            P.ylabel('Mag AB + cst')
            P.xlabel('wavelength [$\AA$]')
            P.gca().invert_yaxis()
            
            P.show()
            

            if len(browser.LIST_TO_DELATE) == 0 :
                spec=SPEC
            
            else:
                spec={}
                spec_bad={}
                Ind=0

                for j in range(len(self.dic_cosmo[sn].keys())):
                    if j in browser.LIST_TO_DELATE:
                        print 'au revoir'
                        spec_bad.update({SPEC['%i'%j]['pause']:SPEC['%i'%j]})
                    else:
                        spec.update({'%i'%(Ind):copy.deepcopy(SPEC['%i'%j])})
                        Ind+=1
                    del SPEC['%i'%(j)]
                dic_bad.update({sn:spec_bad})
            dic_cosmo.update({sn:spec})

        #self.dic_cosmo.update(dic_cosmo)
        #self.dic_bad.update(dic_bad)


        self.dic_cosmo=dic_cosmo
        self.dic_bad=dic_bad


    def Build_without_Cosmology_corrected(self):
        self.dic_without_cosmo={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            for j,pause in enumerate(self.dic_cosmo[sn].keys()):
                print 'processing '+sn+' pause '+pause
                SPEC[pause]['Y']-= -5.*N.log10(d_l(SPEC[pause]['z_cmb'],SNLS=True))+5.

            self.dic_without_cosmo.update({sn:SPEC})

                            
        
    def write_pkl(self,pkl_name):
        
        File=open(pkl_name,'w')
        cPickle.dump(self.dic_cosmo,File)
        File.close()

        if self.BAD:
            File=open('/sps/snovae/user/leget/ALLAIRE/bad_spectra.pkl','w')
            cPickle.dump(self.dic_bad,File)
            File.close()

    def write_without_cosmo_pkl(self,pkl_name):
        
        File=open(pkl_name,'w')
        cPickle.dump(self.dic_without_cosmo,File)
        File.close()


class build_at_max_data:

    def __init__(self,dico_cosmo,phrenology):

        self.Token_list=['EWCaIIHK','EWSiII4000','EWMgII','EWFe4800','EWSIIW','EWSiII5972','EWSiII6355','EWOI7773','EWCaIIIR','vSiII_4128_lbd','vSiII_5454_lbd','vSiII_5640_lbd','vSiII_6355_lbd']
        self.dic_cosmo=cPickle.load(open(dico_cosmo))
        self.phrenology=cPickle.load(open(phrenology))
        self.sn_name=self.dic_cosmo.keys()
        
    def select_spectra_at_max(self,RANGE=[-2.5,2.5]):
        self.dic_cosmo_at_max={}

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo[sn])
            PPhase=-999
            for j,pause in enumerate(self.dic_cosmo[sn].keys()):
                if abs(SPEC[pause]['phase_salt2'])<abs(PPhase):
                    spec=SPEC[pause]
                    PPhase=SPEC[pause]['phase_salt2']
            if PPhase>RANGE[0] and PPhase<RANGE[1]:
                self.dic_cosmo_at_max.update({sn:spec})
        self.sn_name=self.dic_cosmo_at_max.keys()

    def select_spectral_indicators(self):

        dic_cosmo_at_max={}

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
            spectral_indicators=[]
            spectral_indicators_error=[]
            for SI in range(len(self.Token_list)):
                PAUSE=SPEC['pause']
                spectral_indicators.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.Token_list[SI]])
                spectral_indicators_error.append(self.phrenology[sn]['spectra'][PAUSE]['phrenology.'+self.Token_list[SI]+'.err'])
            dic_sn={'spectra':SPEC,
                    'spectral_indicators':N.array(spectral_indicators),
                    'spectral_indicators_error':N.array(spectral_indicators_error)}

            dic_cosmo_at_max.update({sn:dic_sn})
        
        self.dic_cosmo_at_max=dic_cosmo_at_max

        
    def select_sn_list(self,sn_name):       

        dic_cosmo_at_max={}
        for i,sn in enumerate(sn_name):
            if sn in self.sn_name:
                print '%i/%i'%(((i+1),len(sn_name)))
                SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
                dic_cosmo_at_max.update({sn:SPEC})

        self.sn_name=dic_cosmo_at_max.keys()
        self.dic_cosmo_at_max=dic_cosmo_at_max 


    def select_sn_in_good_sample(self,META,SAMPLE=['training']):

        meta = SnfMetaData.SnfMetaData(META)

        meta.add_filter(idr__subset__in = SAMPLE)
        sn_name=meta.targets('target.name',sort_by='target.name')
        dic_cosmo_at_max={}
        for i,sn in enumerate(sn_name):
            if sn in self.sn_name:
                print '%i/%i'%(((i+1),len(sn_name)))
                SPEC=copy.deepcopy(self.dic_cosmo_at_max[sn])
                dic_cosmo_at_max.update({sn:SPEC})

        self.sn_name=dic_cosmo_at_max.keys()
        self.dic_cosmo_at_max=dic_cosmo_at_max

    def write_pkl(self,pkl_name):

        File=open(pkl_name,'w')
        cPickle.dump(self.dic_cosmo_at_max,File)
        File.close()




class load_time_photometry:

    def __init__(self,meta,dic_selected):

        self.dicM=cPickle.load(open(meta))
        self.dic_selected=cPickle.load(open(dic_selected))
        self.sn_name=self.dicM.keys()

    def load_spectra(self):

        self.dic={}
        self.X=[]
        for i,sn in enumerate(self.sn_name):
            if sn in self.dic_selected.keys():
                print '%i/%i'%(((i+1),len(self.sn_name)))
                SPEC={}
                ind=0
                for j,pause in enumerate(self.dicM[sn]['spectra'].keys()):
                    print 'processing '+sn+' pause '+pause 
                      #Sndata = selector('target.name','obs.exp','idr.spec_B','idr.spec_R','target.mwebv','host.zhelio',structured=True)

                    Spec=pySnurp.Spectrum(self.dicM[sn]['spectra'][pause]['idr.spec_merged'])
                    Spec.deredden(self.dicM[sn]['target.mwebv'])
                    Spec.deredshift(self.dicM[sn]['host.zhelio'])
                    
                    if len(Spec.x)==2691:
                        
                        SPEC.update({'%i'%(ind):{'Y':Spec.y,
                                                 'X':Spec.x,
                                                 'V':Spec.v,
                                                 'step':Spec.step,
                                                 'days':self.dicM[sn]['spectra'][pause]['obs.mjd'],
                                                 'z_cmb':self.dicM[sn]['host.zcmb'],
                                                 'z_helio':self.dicM[sn]['host.zhelio'],
                                                 'z_err':self.dicM[sn]['host.zhelio.err'],
                                                 'phase_salt2':self.dicM[sn]['spectra'][pause]['salt2.phase'],
                                                 'pause':pause}})
                        self.X.append(Spec.x)
                        ind+=1

                self.dic.update({sn:SPEC})
        self.sn_name=self.dic.keys()


    def build_resampled_specs(self,lmin=3200,lmax=8900,step=10,delta_lambda=0.005):
        
        self.dic_binned={}
        lmin,lmax=check_bounds(self.X,lmin,lmax,step=step)
        #rebinarray=N.arange(lmin,lmax+step,step)
        rebinarray=[lmin]
        i=0
        while rebinarray[i]<lmax:
            rebinarray.append(rebinarray[i]*delta_lambda+rebinarray[i])
            i+=1
        if rebinarray[i]>lmax:
            del rebinarray[i]

        rebinarray=N.array(rebinarray)    
        print rebinarray
        print "%i filters created"%(len(rebinarray)-1)        

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic[sn])
            for j,pause in enumerate(self.dic[sn].keys()):
                print 'processing '+sn+' pause '+pause 
            
                Spec=spec(SPEC[pause]['Y'],SPEC[pause]['X'],SPEC[pause]['V'],SPEC[pause]['step'])
                SPEC[pause]['X'],SPEC[pause]['Y'],SPEC[pause]['V']=ms.rebin(Spec,rebinarray)

            self.dic_binned.update({sn:SPEC})


    def select_good_spectra(self):

        self.dic_for_photometry={}

        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_binned[sn])
            spec={}
            Ind=0
            PAUSE=[]
            for p,pause in enumerate(self.dic_selected[sn].keys()):
                PAUSE.append(self.dic_selected[sn][pause]['pause'])

            for j in range(len(self.dic_binned[sn].keys())):

                if SPEC['%i'%(j)]['pause'] in PAUSE :
                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC['%i'%j])})
                    Ind+=1
                del SPEC['%i'%(j)]
            self.dic_for_photometry.update({sn:spec})

    def reorder_and_clean_dico_cosmo(self):
        dic_for_photometry ={}
        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_for_photometry[sn])
            spec={}
            Ind=0
            for j in range(len(self.dic_for_photometry[sn].keys())):
                Min=(10.**23)
                ind='0'
                for k,pause in enumerate(SPEC):
                    phase=SPEC[pause]['phase_salt2']
                    if phase<Min:
                        Min=phase
                        ind=pause
                if N.sum(N.isfinite(SPEC[ind]['Y']))==len(SPEC[ind]['Y']):
                    spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
                    Ind+=1
                else:
                    Delta=abs(N.sum(N.isfinite(SPEC[ind]['Y']))-len(SPEC[ind]['Y']))
                    if Delta<5:
                        Filtre=N.isfinite(SPEC[ind]['Y'])
                        SPLINE=inter.InterpolatedUnivariateSpline(SPEC[ind]['X'][Filtre],SPEC[ind]['Y'][Filtre])
                        SPEC[ind]['Y'][~Filtre]=SPLINE(SPEC[ind]['X'][~Filtre])
                        SPEC[ind]['V'][~Filtre]=100.
                        spec.update({'%i'%(Ind):copy.deepcopy(SPEC[ind])})
                        Ind+=1
                del SPEC[ind]
            dic_for_photometry.update({sn:spec})

        self.dic_for_photometry=dic_for_photometry




    def build_BD17_mag(self):

        self.dic_photometry_BD17={}

        U_PFL=N.linspace(3360,4048.2,1000)
        B_SNf=N.linspace(4048.2,4877.3,1000)
        V_SNf=N.linspace(4877.3,5876.3,1000)
        R_SNf=N.linspace(5876.3,7079.9,1311)
        I_PFL=N.linspace(7079.9,8530,931)
        SALT=SALT2model.Salt2Model()


        for i,sn in enumerate(self.sn_name):
            print '%i/%i'%(((i+1),len(self.sn_name)))
            SPEC=copy.deepcopy(self.dic_for_photometry[sn])
            U=[]
            B=[]
            V=[]
            R=[]
            I=[]
            B_err=[]
            V_err=[]
            R_err=[]
            Time=[]

            BJ=[]
            BJ_err=[]
                  

            for j in range(len(SPEC)):
                print j,len(SPEC) 
                u=mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'],lambda_min=U_PFL[0],lambda_max=U_PFL[-1],var=None, step=None,AB=False)[0]
                b=mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'],lambda_min=B_SNf[0],lambda_max=B_SNf[-1],var=None, step=None,AB=False)[0]
                v=mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'],lambda_min=V_SNf[0],lambda_max=V_SNf[-1],var=None, step=None,AB=False)[0]
                r=mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'],lambda_min=R_SNf[0],lambda_max=R_SNf[-1],var=None, step=None,AB=False)[0]
                i=mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'],lambda_min=I_PFL[0],lambda_max=I_PFL[-1],var=None, step=None,AB=False)[0]
                bj=SALT.mag(SPEC['%i'%(j)]['X'],SPEC['%i'%(j)]['Y'], var=None, step=None, syst='STANDARD', filter='B')[0]

                U.append(u)
                B.append(b)
                V.append(v)
                R.append(r)
                I.append(i)
                BJ.append(bj)


                #U_err.append(u_err)
                #B_err.append(b_err)
                #V_err.append(v_err)
                #R_err.append(r_err)
                #I_err.append(i_err)
                #BJ_err.append(bj_err)

                Time.append(SPEC['%i'%(j)]['phase_salt2'])

            self.V=V
            self.R=R
            self.I=I
            print 'je suis une andalouse'    

            self.dic_photometry_BD17.update({sn:{'spectra':SPEC,
                                                 'photometry':{'BSNf':N.array(B),
                                                               'USNf':N.array(U),
                                                               'VSNf':N.array(V),
                                                               'RSNf':N.array(R),
                                                               'ISNf':N.array(I),
                                                               'B':N.array(BJ),
                                                               'Time':N.array(Time)}}})


        print 'je suis andalouse' 


    def write_pkl(self,pkl_name):

        File=open(pkl_name,'w')
        cPickle.dump(self.dic_photometry_BD17,File)
        File.close()




def write_dat_for_GP(directory,data_file,List_SN=None,MFR_issue=False):

    dic=cPickle.load(open(data_file))

    sn_name=dic.keys() 
    sn_name=N.array(sn_name)
    Filtre=N.array([True]*len(sn_name))

    if MFR_issue or List_SN is not None :
        
        for i in range(len(sn_name)):
            if MFR_issue and dic[sn_name[i]]['0']['days']>55250.:
                Filtre[i]=False
            if List_SN is not None:
                if sn_name[i] not in List_SN:
                    Filtre[i]=False

    
    for i,sn in enumerate(sn_name[Filtre]):
        fichier=open(directory+sn+'.dat','w')
        spec=dic[sn]
        print sn
        for j in range(len(spec.keys())):
            for Bin in range(len(spec['%i'%(j)]['X'])):
                if spec['%i'%(j)]['phase_salt2']>50.:
                    print 'non'
                else:
                    fichier.write('%.5f    %.5f    %.5f    %.5f \n'%((spec['%i'%(j)]['phase_salt2'],spec['%i'%(j)]['X'][Bin],spec['%i'%(j)]['Y'][Bin],N.sqrt(spec['%i'%(j)]['V'][Bin]))))

        fichier.close()
    

class compute_SUGAR_in_flux:

    def __init__(self,SUGAR_parameters,sn_name,SUGAR_asci,HYPERPARAMETER):

        self.dic=cPickle.load(open(SUGAR_parameters))
        self.sn_name=sn_name
        self.SUGAR_asci=SUGAR_asci
        self.HYPERPARAMETER=HYPERPARAMETER

    
    def compute_SUGAR_flux(self,dic_data):

        self.dic_data=cPickle.load(open(dic_data))
        self.dic_SUGAR={}

        for i in range(len(self.sn_name)):

            print '%i/%i'%((i+1,len(self.sn_name)))

            TIME=[]
            PAUSE=[]

            for t in range(len(self.dic_data[self.sn_name[i]].keys())):
                if self.dic_data[self.sn_name[i]]['%i'%(t)]['phase_salt2']>-12 and self.dic_data[self.sn_name[i]]['%i'%(t)]['phase_salt2']<42:
                    TIME.append(self.dic_data[self.sn_name[i]]['%i'%(t)]['phase_salt2'])
                    PAUSE.append(self.dic_data[self.sn_name[i]]['%i'%(t)]['pause'])

            dic_SN={}
            BS=build_SUGAR_ligh_curves(self.SUGAR_asci,self.HYPERPARAMETER)
            AV=self.dic[self.sn_name[i]]['Av']
            GREY=self.dic[self.sn_name[i]]['Grey']
            QQ=[self.dic[self.sn_name[i]]['x1'],self.dic[self.sn_name[i]]['x2'],self.dic[self.sn_name[i]]['x3']]
            BS.comp_SUGAR_in_mag(QQ,AV,GREY)
            BS.align_time(TIME)
            BS.go_to_flux()
            flux=BS.Flux

            for time in range(len(BS.Phase)):
                dic_SN.update({'%i'%(time):{'X':BS.wavelength,
                                            'Y':BS.Flux[time*190:(time+1)*190],
                                            'pause':PAUSE[time]}})

            self.dic_SUGAR.update({self.sn_name[i]:dic_SN})


    def get_idr_data(self,META):
        self.test=load_time_spectra(META)
        self.test.sn_name=self.sn_name
        self.test.load_spectra()
        self.test.build_resampled_specs(lmin=3300,lmax=8600,step=10) 
        self.dic_idr=self.test.dic_binned



    def compare_in_flux(self):

        for i in range(len(self.sn_name)):
            print '%i/%i'%((i+1,len(self.sn_name)))
            P.figure(figsize=(12,16))
            cst=0
            for t in range(len(self.dic_SUGAR[self.sn_name[i]].keys())):
                for TT in range(len(self.dic_idr[self.sn_name[i]].keys())):
                    if self.dic_SUGAR[self.sn_name[i]]['%i'%(t)]['pause']==self.dic_idr[self.sn_name[i]]['%i'%(TT)]['pause']:
                        X=self.dic_idr[self.sn_name[i]]['%i'%(TT)]['X']
                        Y=self.dic_idr[self.sn_name[i]]['%i'%(TT)]['Y']
                if t==0:
                    AAA=2*N.std(Y)
                P.plot(X,Y-cst,'k',linewidth=3)
                P.plot(self.dic_SUGAR[self.sn_name[i]]['%i'%(t)]['X'],self.dic_SUGAR[self.sn_name[i]]['%i'%(t)]['Y']-cst,'b',linewidth=3)
                cst+=AAA

    def write_data_dat(self,sn_name):

        fichier=open(sn_name+'.dat','w')
        Indice=0
        for t in range(len(self.dic_SUGAR[sn_name].keys())):
            for TT in range(len(self.dic_idr[sn_name].keys())):
                if self.dic_SUGAR[sn_name]['%i'%(t)]['pause']==self.dic_idr[sn_name]['%i'%(TT)]['pause']:
                    if abs(self.dic_idr[sn_name]['%i'%(TT)]['phase_salt2'])<1.5 and Indice==0:
                        for Lam in range(len(self.dic_SUGAR[sn_name]['%i'%(t)]['X'])):
                            fichier.write('%f  %.8E \n'%((self.dic_SUGAR[sn_name]['%i'%(t)]['X'][Lam],self.dic_SUGAR[sn_name]['%i'%(t)]['Y'][Lam])))
                        Indice+=1
        fichier.close()



if __name__=="__main__":


    test=load_time_spectra('/sps/snovae/user/leget/CABALLO/META.pkl')
    #test.sn_name=['PTF09dnl']
    test.load_spectra()
    test.build_resampled_specs(lmin=3300,lmax=8600,step=10)
    test.to_AB_mag()
    test.Cosmology_corrected()
    test.reorder_and_clean_dico_cosmo()
    test.select_night_in_previous_dico('/sps/snovae/user/leget/CABALLO/all_CABALLO_data_binning_training.pkl')
    #test.kill_Blue_runaway_eye_control()
    #


    #sn='PTF09dnl'
    #for i in range(len(test.dic_cosmo[sn].keys())):
    #    P.plot(dic_cosmo[sn]['%i'%(i)]['X'],dic_cosmo[sn]['%i'%(i)]['Y'],'r')
    #    P.fill_between(dic_cosmo[sn]['%i'%(i)]['X'],dic_cosmo[sn]['%i'%(i)]['Y']-N.sqrt(dic_cosmo[sn]['%i'%(i)]['V']),
    #                   dic_cosmo[sn]['%i'%(i)]['Y']+N.sqrt(dic_cosmo[sn]['%i'%(i)]['V']),color='r',alpha=0.5)
    #    
    #    P.plot(test.dic_cosmo[sn]['%i'%(i)]['X'],test.dic_cosmo[sn]['%i'%(i)]['Y'],'b')
    #    P.fill_between(test.dic_cosmo[sn]['%i'%(i)]['X'],test.dic_cosmo[sn]['%i'%(i)]['Y']-N.sqrt(test.dic_cosmo[sn]['%i'%(i)]['V']),
    #                   test.dic_cosmo[sn]['%i'%(i)]['Y']+N.sqrt(test.dic_cosmo[sn]['%i'%(i)]['V']),color='b',alpha=0.5)
    #    P.gca().invert_yaxis()
    #    P.show()


    #test.sn_name=test.dic_cosmo.keys()
    #test.select_sn_in_good_sample('/sps/snovae/user/leget/CABALLO/META.pkl')
    test.write_pkl('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_for_maria.pkl')

    #test.Build_without_Cosmology_corrected()
    #test.write_without_cosmo_pkl('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_without_cosmology.pkl')

    #SP=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_without_cosmology.pkl')) 
    #CSF=compute_SUGAR_in_flux('/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_parameters_without_cosmology.pkl',['PTF09dnl'],'/sps/snovae/user/leget/CABALLO/SUGAR_validation/SUGAR_model_v1.asci','/sps/snovae/user/leget/CABALLO/Prediction_GP_binning_speed_without_MFR_issue/hyperparameters.dat')
    #CSF.compute_SUGAR_flux('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl')
    #CSF.get_idr_data('/sps/snovae/user/leget/CABALLO/META.pkl')
    #CSF.compare_in_flux()
    #CSF.write_data_dat('PTF09dnl')


    #write_dat_for_GP('/sps/snovae/user/leget/CABALLO/data_for_GP/','/sps/snovae/user/leget/CABALLO/all_CABALLO_data.pkl')
    #write_dat_for_GP('/sps/snovae/user/leget/CABALLO/data_for_GP_binning_speed/','/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl')
    #DIC=cPickle.load(open('/sps/snovae/user/leget/CABALLO/SUGAR_validation/emfa_CABALLO.pkl'))
    #SN_rejected=N.array(DIC['sn_name'])[DIC['filter']]
    #write_dat_for_GP('/sps/snovae/user/leget/CABALLO/data_for_GP_binning_speed_without_MFR_issue/','/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl',List_SN=SN_rejected,MFR_issue=True)


    #LTP=load_time_photometry('/sps/snovae/user/leget/CABALLO/META.pkl','/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl')
    #LTP.sn_name=['PTF09fox']
    #LTP.load_spectra() 
    #LTP.build_resampled_specs(lmin=3300,lmax=8600,step=10) 
    #LTP.select_good_spectra()
    #LTP.reorder_and_clean_dico_cosmo()
    #LTP.build_BD17_mag()
    #LTP.write_pkl('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_with_BD17_photometry_SNf_filter.pkl')

    #BMD=build_at_max_data('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed.pkl','/sps/snovae/user/leget/CABALLO/phrenology_2016_12_01_CABALLOv1.pkl')
    #BMD=build_at_max_data('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_test_RV.pkl','/sps/snovae/user/leget/CABALLO/phrenology_2016_12_01_CABALLOv1.pkl')
    #BMD.select_spectra_at_max(RANGE=[-5,5])
    #BMD.select_spectral_indicators()
    #pca=cPickle.load(open('/sps/snovae/user/leget/BEDELL/pca_training.pkl'))
    #sn_name=pca['sn_name']
    #BMD.select_sn_list(sn_name)
    #BMD.write_pkl('/sps/snovae/user/leget/CABALLO/SUGAR_validation/all_CABALLO_data_binning_speed_with_SI_test_RV.pkl')


    #BMD.select_sn_in_good_sample('/sps/snovae/user/leget/CABALLO/META.pkl',SAMPLE=['training'])
