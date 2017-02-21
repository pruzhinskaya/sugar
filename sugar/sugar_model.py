#!/usr/bin/env python
######################################################################
## Filename:      SUGAR_model.py
## version:       $Revision: 1.0 $
## Description:   
## Author:        $Author: leget $
## Created at:    Sep 30 10:07:00 2014
## $Id:  $
######################################################################


import sys,os,optparse
from pylab import *
from multilinearfit import *
import cPickle
from Passage import passage_error,passage_error_error_sn
import copy
import numpy as N
import scipy as S
from scipy.sparse import block_diag
from ToolBox import EMfa as EM_manu
import EMfa_covariant_first as EMfa_cov
rep_pkl='/sps/snovae/user/leget/'

#################################################
################################################# 
# TO DO :
# - Put K-folding in SUGAR_model
# - Write Load_data better 
################################################# 
#################################################

#==============================================================================
# Compute Supernovae Usefull Generator And Reduction (SUGAR) model 
#==============================================================================


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

code_name = "<SUGAR_model.py> "

def read_option(): 
    
    usage = "usage: [%prog] -p pca_input -s spectra_input -m model_output [otheroptions]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--Parallel","-P",dest="Parallel",help="Use mpi4py to accelerate the script",default=False,action="store_true")
    parser.add_option("--Bloc","-B",dest="Bloc",help="Use if Parallel and CovY are Bloc diagonal",default=False,action="store_true")
    parser.add_option("--pca","-p",dest="pca",help="pca input file.",default=None)
    parser.add_option("--Gaussian","-G",dest="GP",help="gaussian process input directory.",default=None)
    parser.add_option("--SNlist","-S",dest="SN",help="gaussian process supernovae list.",default=None)
    parser.add_option("--ModelMax","-M",dest="M",help="Model at max.",default=None)
    parser.add_option("--spectra","-s",dest="spectra",help="spectra input file.",default=None)
    parser.add_option("--model","-m",dest="model",help="Model output file.",default=None)
    parser.add_option("--intrinsic","-i",dest="intrinsic",help="not intrinsic correction",default=True,action="store_false")
    parser.add_option("--color","-r",dest="color",help="Fit with reddening law",default=False,action="store_true")
    parser.add_option("--dispersion","-d",dest="dispersion",help="add disp matrix",default=False,action="store_true")
    parser.add_option("--IterMaxDisp","-I",dest="IterMaxDisp",help="Number Max of Iteration for disp Matrix",default=None)
    parser.add_option("--ccm","-c",dest="ccm",help="Fit with CCM law",default=False,action="store_true")
    parser.add_option("--grey","-g",dest="grey",help="Fit with delta M grey",default=False,action="store_true")
   # parser.add_option("--phot","-f",dest="phot",help="Fit in photometry",default=False,action="store_true")
    parser.add_option("--numbercomp","-n",dest="numbercomp",help="number of eigenvector",default='3')
    parser.add_option("--emfaresidual","-e",dest="emfa",help="emfa on residuals",default=False,action="store_true")
    parser.add_option("--emfadispmatrix","-E",dest="emfa_comp_disp",help="compute disp matrix with emfa",default=False,action="store_true")
    option,args = parser.parse_args()
    
    if not option.pca : raise parser.error(" ERROR: give me a pca inputs")
    #if not option.spectra or not option.GP : raise parser.error(" ERROR: give me a spectra inputs or directory for Gaussian process")
    #if option.GP is not None and option.spectra is not None : raise parser.error(" ERROR: you need to choice between GP and spectra inputs")
    #if option.GP is not None and not option.SNlist :  raise parser.error(" ERROR: give me a Supernovae list for GP")
    #if option.GP is not None and not option.M : raise parser.error(" ERROR: give me a model at max for GP")
    #if option.GP is not None and option.color : raise parser.error(" ERROR: reddening law find at max for GP")
    #if option.GP is not None and option.ccm : raise parser.error(" ERROR: reddening law find at max for GP")
    #if option.spectra is not None and option.M is not None : raise parser.error(" ERROR: don't need that")
    #if option.spectra is not None and option.SNlist is not None : raise parser.error(" ERROR: don't need that")


    if not option.model: raise parser.error(" ERROR: give me a model output")
    #if option.dir[-1] != '/': option.dir+='/'
    #if os.path.exists(option.dir) == False:
    #    print >> sys.stderr, code_name + "creating %s"%(option.dir)
    #    os.mkdir(option.dir)

    #option.exclude = option.exclude.split(',')
    #option.indicList = option.indicList.split(',')
    #option.NDindics = option.NDindics.split(',')

    return option


class Load_data:
    
    def __init__(self,dic_pca,dico_spectra_at_max,FILTRE=True):
        
        # To have spectral indicators, EM-PCA data, and SALT2 params


        dicpca = cPickle.load(open(dic_pca))
        self.pca_sn_name=N.array(dicpca['sn_name'])
        dicstandard = cPickle.load(open(dico_spectra_at_max))

        if FILTRE:
            FILTRE=dicpca['filter']
            for i in range(len(self.pca_sn_name)):
                if dicstandard[self.pca_sn_name[i]]['spectra']['days']>55250: 
                    FILTRE[i]=False

        else:
            FILTRE=N.array([True]*len(self.pca_sn_name))

        self.pca_error=dicpca['error'][FILTRE]
        self.pca_data=dicpca['data'][FILTRE]
        self.pca_val=dicpca['val']
        self.pca_vec=dicpca['vec']
        self.pca_norm=dicpca['norm']
        self.pca_Norm_data=dicpca['Norm_data'][FILTRE]
        self.pca_Norm_err=dicpca['Norm_err'][FILTRE]
        self.sn_name=self.pca_sn_name[FILTRE]

        print 'INFO POUR PF: tu as %i SNIa'%(len(self.sn_name))

        # Load spectrum and stat with or without correction in ABmag +cst of all supernovae
 

        self.Y_cosmo_corrected=[]
        self.Y_err=[]
        self.dm_z=[]

        for i in range(len(self.sn_name)):
            sn=self.sn_name[i]
            self.Y_cosmo_corrected.append(dicstandard[sn]['spectra']['Y'])
            self.Y_err.append(N.sqrt(dicstandard[sn]['spectra']['V']))
            pec_vel=0.001
            self.dm_z.append(5/N.log(10) * N.sqrt(dicstandard[sn]['spectra']['z_err']**2 + pec_vel**2) / dicstandard[sn]['spectra']['z_cmb'])

        self.Y_cosmo_corrected=N.array(self.Y_cosmo_corrected)
        self.Y_err=N.array(self.Y_err)
        self.dm_z=N.array(self.dm_z)

        self.X=dicstandard[self.sn_name[0]]['spectra']['X']


                

    # genererate param for EM-PCA standardization

    def compute_EM_PCA_data(self,number_eigenvector):
             
        
        dat=self.pca_Norm_data
        err=self.pca_Norm_err
        

        new_base=passage_error(dat,err,self.pca_vec,sub_space=number_eigenvector)
        new_err,cov_new_err=passage_error_error_sn(err,self.pca_vec,number_eigenvector)
            
        self.data=new_base
        self.Cov_error=cov_new_err
        self.err=None
        self.new_err=new_err
        self.key='%i_eigenvector_EMPCA'%(number_eigenvector)

    def compute_SI_norm_data(self,number_SI):
             
        
        dat=self.pca_Norm_data
        err=self.pca_Norm_err
        

        self.data=dat[:,:number_SI]
        self.Cov_error=N.zeros((len(self.data[:,0]),len(self.data[0]),len(self.data[0])))
        self.err=err[:,:number_SI]
        for i in range(len(self.data[:,0])):
            self.Cov_error[i]=N.diag(self.err[i]**2)
            print self.Cov_error[i] 
class load_data_GP_TW_by_PF:

    def __init__(self,pca_pkl,model_at_max,rep_GP,sn_list,Parallel=True,FILTRE=True):

        dicpca = cPickle.load(open(pca_pkl))
        pca_sn_name=N.array(dicpca['sn_name'])

        if FILTRE:
            FILTRE=dicpca['filter']
            for i in range(len(pca_sn_name)):
                if dicpca['DAYS'][i]>55250: 
                    FILTRE[i]=False
        else:
            FILTRE=N.array([True]*len(pca_sn_name))

        self.pca_error=dicpca['error'][FILTRE]
        self.pca_data=dicpca['data'][FILTRE]
        self.pca_val=dicpca['val']
        self.pca_vec=dicpca['vec']
        self.pca_norm=dicpca['norm']
        self.pca_Norm_data=dicpca['Norm_data'][FILTRE]
        self.pca_Norm_err=dicpca['Norm_err'][FILTRE]
        self.sn_name=pca_sn_name[FILTRE]

        dic_model=cPickle.load(open(model_at_max))

        self.sn_name_Av=dic_model['sn_name']
        self.Av=dic_model['Av_cardelli']
        self.RV=dic_model['RV']

        self.rep_GP=rep_GP
        self.sn_list=sn_list


        self.sn_predict=N.loadtxt(sn_list,dtype='string')
        #self.sn_predict=self.sn_predict[:,8]

#        self.Filtre=N.array([True]*len(self.pca_sn_name))
#
#        for i,sn in enumerate(self.pca_sn_name):
#            if sn+'.predict' in self.sn_predict:
#                print 'ok'
#            else:
#                self.Filtre[i]=False
#
#        self.pca_sn_name=self.pca_sn_name[self.Filtre]
#        self.pca_error=self.pca_error[self.Filtre]
#        self.pca_data=self.pca_data[self.Filtre]
#        self.pca_Norm_data=self.pca_Norm_data[self.Filtre]
#        self.pca_Norm_err=self.pca_Norm_err[self.Filtre]

        self.N_sn=len(self.sn_name)

        if Parallel:
            try: from mpi4py import MPI
            except ImportError : raise ImportError('you need to have mpi4py on your computer to use this option')
            self.MPI=MPI
            self.Parallel=True
            self.comm = self.MPI.COMM_WORLD
            size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

            self.njob=int(self.N_sn/size)

            if self.N_sn%size!=0:
                ValueError('Number of observation (supernovae) % MPI.COMM_WORLD.Get_size() need to be equal zeros (equal %i)'%(self.N_sn%size))

            self.START=self.njob*self.rank
            self.Number_loop=self.njob


        else:
            self.Parallel=False
            self.Number_loop=self.N_sn


        self.number_bin_phase=0
        self.number_bin_wavelength=0
        print self.rep_GP+self.sn_name[0]+'.predict'
        A=N.loadtxt(self.rep_GP+self.sn_name[0]+'.predict')
        phase=A[:,0]
        wavelength=A[:,1]
        self.TX=wavelength
        for i in range(len(wavelength)):
            if wavelength[i]==wavelength[0]:
                self.number_bin_phase+=1

        self.number_bin_wavelength=len(wavelength)/self.number_bin_phase


    def compute_EM_PCA_data(self,number_eigenvector):


        dat=self.pca_Norm_data
        err=self.pca_Norm_err


        new_base=passage_error(dat,err,self.pca_vec,sub_space=number_eigenvector)
        new_err,cov_new_err=passage_error_error_sn(err,self.pca_vec,number_eigenvector)

        self.data=new_base
        self.Cov_error=cov_new_err
        self.err=None
        self.new_err=new_err
        

    def load_spectra_GP(self,sn_name,Color=True):

        A=N.loadtxt(self.rep_GP+sn_name+'.predict')
        Y=A[:,2]
        for j,sn_av in enumerate(self.sn_name_Av):
            if sn_name ==sn_av:
                Y_cardelli_corrected=(Y-(self.Av[j]*Astro.Extinction.extinctionLaw(A[:,1],Rv=self.RV,law='CCM89')))

        if Color:
            return Y,Y_cardelli_corrected
        else:
            return Y


    def load_phase_wavelength(self,sn_name):
        
        A=N.loadtxt(self.rep_GP+sn_name+'.predict')
        phase=A[:,0]
        wavelength=A[:,1]
        del A
        
        return phase,wavelength


    def load_cov_matrix(self,sn_name):

        A=N.loadtxt(self.rep_GP+sn_name+'.predict')
        size_matrix=self.number_bin_phase*self.number_bin_wavelength
        COV=N.zeros((size_matrix,size_matrix))

        for i in range(self.number_bin_wavelength):
            cov=A[(i*self.number_bin_phase):((i+1)*self.number_bin_phase),3:]
            COV[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase] = cov

        return COV
        


    def load_spectra(self):

        self.phase,self.X=self.load_phase_wavelength(self.sn_name[0])
        self.Y_cardelli_corrected_cosmo_corrected=N.zeros((len(self.sn_name),len(self.X)))
        self.Y_cosmo_corrected=N.zeros((len(self.sn_name),len(self.X)))
        self.CovY=[]
        self.W=[]

        
        for i,sn in enumerate(self.sn_name):
            print sn 
            self.Y_cosmo_corrected[i],self.Y_cardelli_corrected_cosmo_corrected[i]=(self.load_spectra_GP(sn))

            if not self.Parallel:
                Cov=self.load_cov_matrix(sn)
                COV=[]
                for i in range(self.number_bin_wavelength):
                    COV.append(Cov[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase])
                self.CovY.append(block_diag(COV))


        if self.Parallel:
            for j in range(self.Number_loop):
                sn=self.START+j
                print self.sn_name[sn]
                COV=[]
                
                Cov=self.load_cov_matrix(self.sn_name[sn])
                for i in range(self.number_bin_wavelength):
                    
                    COV.append(Cov[i*self.number_bin_phase:(i+1)*self.number_bin_phase, i*self.number_bin_phase:(i+1)*self.number_bin_phase])

                AAA=block_diag(COV)
                self.CovY.append(AAA)
                self.comm.Barrier()





######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
###################################################################################################################### 




class SUGAR_model:
    
    def __init__(self,Mag,data,CovX,wavelength,sn_name,Mag_err=None,Mag_cov=None):

        


        ################################################# 
        # Load spectra
        #################################################  

        self.sn_name=sn_name
        self.Mag_no_corrected=copy.deepcopy(Mag)
        self.Mag_corrected=copy.deepcopy(Mag)
        self.Mag_all_sn_err=copy.deepcopy(Mag_err)
        self.X=wavelength

        self.Mag_cov=Mag_cov

        ############################################################ 
        # Load EM-PCA space (or other params like stretch and color)
        ############################################################  

        self.data=data
        self.Cov_error=CovX

        if self.data is not None:
            self.number_correction=len(self.data[0])
        else:
            self.number_correction=0
        ############################################################ 
        # prepare the fit result
        ############################################################  


        self.M0=zeros(len(self.X))
        self.reddening_law=zeros(len(self.X))
        
        if self.data is None:
            self.alpha=zeros((len(self.Mag_no_corrected[0]),self.number_correction))
            self.xplus=zeros(shape(self.data))
        else:
            self.alpha=None
            self.xplus=None

        self.delta_M_grey=zeros(len(self.sn_name))
        self.Av=zeros(len(self.sn_name))
        self.Av_cardelli=zeros(len(self.sn_name))
        self.RV=0

        self.disp=0.
        self.disp_matrix=zeros((len(self.X),len(self.X)))
        self.corr_matrix=zeros((len(self.X),len(self.X)))
        self.diag_std_disp_matrix=zeros(len(self.X))
    
        self.CHI2=[]
        self.WRMS_no_K_folding=zeros(len(self.X))
        self.WRMS_K_folding=zeros(len(self.X))

        self.Alpha_variation=None
        self.chi2_Alpha_variation=None
        self.M0_variation=None
        self.chi2_M0_variation=None
        self.xplus_variation=None
        self.chi2_xplus_variation=None
        self.red_law_variation=None
        self.chi2_red_law_variation=None
        self.Av_variation=None
        self.chi2_Av_variation=None
        self.grey_variation=None
        self.chi2_grey_variation=None
        self.inv_Rv_variation=None
        self.chi2_inv_Rv_variation=None

  
       
    ################################################################
    # Compute SUGAR model with Global_fit (from multilinearfit.py)
    ################################################################  

    def Compute_Sugar_Model(self,MAX_iter_disp=150,Alpha_init=None,Xplus_init=None,Grey_init=None,M0_init=None,B_minus_V=None,DMz=None,COMUNICATOR=None,COLOR=True,DELTA_M_GREY=True,Cardelli_fit=True,MAX_iter=N.inf,ADDDISP=False,jack=True,MAP_CHI2=False,EMFA_DISP_MATRIX=False,PARALLEL=False,BLoc=False,STAT=False):

        File=open('/sps/snovae/user/leget/File_%i.dat'%(len(self.data[0])),'w')
        File.write('nombre de vecteur : %i, nombre de supernova :%i'%((len(self.data[0]),len(self.sn_name))))
        File.close()

        self.COLOR=COLOR
        self.DELTA_M_GREY=DELTA_M_GREY

        if self.COLOR:
            self.Color=1
        else:
            self.Color=0

        if self.DELTA_M_GREY:
            self.grey=1
        else:
            self.grey=0

        self.DMz=DMz
            

        ##############################
        # Load and run the Global fit
        ############################## 

        GF=global_fit(self.Mag_no_corrected,self.X,
                      data=self.data,CovX=self.Cov_error,dY=self.Mag_all_sn_err,
                      CovY=self.Mag_cov,dm_z=DMz,alpha0=Alpha_init,reddening_law=None,
                      M00=M0_init,H0=Xplus_init,B_V=None,Delta_M0=Grey_init,Color=COLOR,delta_M_grey=DELTA_M_GREY,
                      CCM=Cardelli_fit,EMFA_disp_matrix=EMFA_DISP_MATRIX,Disp_matrix_Init=None,Communicator=COMUNICATOR,Parallel=PARALLEL)

        if Alpha_init is not None:
            IInit=False
        else:
            IInit=True
            
        if EMFA_DISP_MATRIX:
            MAX_iter_disp=1000
        self.GF=GF
        GF.run_Global_fit(MAX_ITER=MAX_iter,Max_iter_disp=MAX_iter_disp,Init=IInit,Bloc_diag=BLoc,
                          Addisp=ADDDISP,JACK=jack,
                          Norm_Chi2=False,Map_chi2=MAP_CHI2)

        if STAT:
            GF.comp_stat()
            self.WRMS=GF.WRMS

        print 'pouet'

        GF.separate_slopes_and_data()

        print 'tata'
        
        ##############################
        # Extract the fit result
        ############################## 
            
        self.M0=GF.M0
        self.reddening_law=GF.reddening_law
        
        if self.data is not None:
            self.alpha=GF.Alpha
            self.xplus=GF.xplus

        self.delta_M_grey=GF.delta_M_GREY
        self.Av=GF.Av
        self.intrinsic=GF.intrinsic

        if ADDDISP:
            self.disp=GF.disp_added
            self.disp_matrix=GF.disp_matrix*GF.disp_added**2

            #if self.disp_matrix==0:
            #    self.diag_std_disp_matrix=0
            #else:
            self.diag_std_disp_matrix=sqrt(diag(self.disp_matrix))
                
            self.Full_REML=GF.Full_REML
            for Bin_i in range(len(self.X)):
                for Bin_j in range(len(self.X)):
                    self.corr_matrix[Bin_i,Bin_j]=self.disp_matrix[Bin_i,Bin_j]/(self.diag_std_disp_matrix[Bin_i]*self.diag_std_disp_matrix[Bin_j])
                    

        if Cardelli_fit :
            self.RV=1./GF.inv_Rv
            if COLOR:
                self.Av_cardelli=GF.Av_cardelli
                    
            ##############################
            # Extract the CHI2 and wRMS
            ############################## 
            
        self.CHI2.append(GF.CHI2)
        
        if not PARALLEL:
            self.residuals=GF.residuals
            self.cov_residuals=GF.COVY

        self.GF=GF
        if jack:
            
            self.alpha_err_Jackknife = GF.alpha_err_Jackknife
            self.M0_err_Jackknife = GF.M0_err_Jackknife
            self.reddening_law_err_Jackknife = GF.reddening_law_err_Jackknife
            if Cardelli_fit:
                self.RV_err_Jackknife = GF.RV_err_Jackknife
            else:
                self.RV_err_Jackknife = 0
    
        else:
            self.alpha_err_Jackknife = 0
            self.M0_err_Jackknife = 0
            self.reddening_law_err_Jackknife = 0
            self.RV_err_Jackknife = 0
    
    
        if MAP_CHI2: 
            
            if self.intrinsic:
                self.Alpha_variation=GF.Alpha_variation
                self.chi2_Alpha_variation=GF.chi2_Alpha_variation
                
                self.xplus_variation=GF.xplus_variation
                self.chi2_xplus_variation=GF.chi2_xplus_variation
    
            else:
                self.Alpha_variation=None
                self.chi2_Alpha_variation=None
                
                self.xplus_variation=None
                self.chi2_xplus_variation=None
    
                
            self.M0_variation=GF.M0_variation
            self.chi2_M0_variation=GF.chi2_M0_variation
             
                
            if COLOR:
                self.red_law_variation=GF.red_law_variation
                self.chi2_red_law_variation=GF.chi2_red_law_variation
                
            if COLOR or Cardelli_fit:
                self.Av_variation=GF.Av_variation
                self.chi2_Av_variation=GF.chi2_Av_variation
        
            if DELTA_M_GREY:
                self.grey_variation=GF.grey_variation
                self.chi2_grey_variation=GF.chi2_grey_variation
        
            if Cardelli_fit:
                self.inv_Rv_variation=GF.inv_Rv_variation
                self.chi2_inv_Rv_variation=GF.chi2_inv_Rv_variation
         


    def EM_PCA_residuals(self,COV=True):

        if COV:
            W=N.zeros(N.shape(self.cov_residuals))
            for i in range(len(self.cov_residuals)):
                W[i]=N.linalg.inv(self.cov_residuals[i])
            EMPCA=EMfa_cov.EMfa_covariant(self.residuals,W)
            EMPCA.converge(10,niter=500)
        else:
            VAR=N.zeros(N.shape(self.residuals))

            for sn in range(len(self.sn_name)):
                VAR[sn]=N.diag(self.cov_residuals[sn])
                 
            EMPCA=EM_manu.EMPCA(self.residuals,1./VAR)
            EMPCA.converge(10,niter=500,center=True)
        self.Lambda=EMPCA.Lambda
        self.Z=EMPCA.Z



    ###############################################################################
    # Build spectra with the SUGAR model (with the supernovae used during the fit)
    ###############################################################################  
  
    def build_spectra(self,Filtre=None):

        #if Filtre !=None:
        #    continue
        #else:
        #    "pouet"
    
        sn_name=self.sn_name
        data=self.data
        Cov_error=self.Cov_error
        

        Y_build=zeros(shape(self.Mag_no_corrected))
        Y_build_error=zeros(shape(self.Mag_no_corrected))

        
        
        for sn in range(len(sn_name)):
            for Bin in range(len(self.X)):
                if self.intrinsic:
                    for vec in range(self.number_correction):
                    
                       ######################################
                       # Add intrinsic variability
                       #######################################


                        Y_build[sn,Bin]+=self.alpha[Bin,vec]*data[sn,vec]
                    
                        ######################################
                        # Compute error
                        #######################################

                        for k in range(self.number_correction):
                            if vec>k :
                                continue
                            else:
                                Y_build_error[sn,Bin] += self.alpha[Bin,vec]*self.alpha[Bin,k]*Cov_error[sn,vec,k]


                ######################################
                # add Mean spectrum
                #######################################
                
                Y_build[sn,Bin]+=self.M0[Bin]
                
                    
                ######################################
                # add Grey offset
                #######################################

                if self.DELTA_M_GREY:
                    Y_build[sn,Bin]+=self.delta_M_grey[sn]

                ######################################
                # add dust extinction
                #######################################

                if self.COLOR:
                    Y_build[sn,Bin]+=(self.Av[sn]*self.reddening_law[Bin])



        self.Y_build=Y_build
        self.Y_build_error=Y_build_error






    ######################################
    # Write the SUGAR model in a pkl file
    #######################################

    def write_pkl(self,name_pkl_file):

        if self.Mag_cov is not None:
            del self.Mag_cov

        dic_data=self.__dict__
        File=open(name_pkl_file,'w')        
        cPickle.dump(dic_data,File)
        File.close()



# ===================================================================


if __name__=="__main__":


    
    option = read_option()

    if option.GP is not None:
        LD=load_data_GP_TW_by_PF(option.pca,option.M,option.GP,option.SN,Parallel=option.Parallel)

        LD.load_spectra()
        ncomp=int(option.numbercomp)
        LD.compute_EM_PCA_data(ncomp)

        if option.Parallel:
            com=LD.comm
        else:
            com=None

        SUGAR=SUGAR_model(LD.Y_cardelli_corrected_cosmo_corrected,
                          LD.data,LD.Cov_error,LD.TX,LD.sn_name,Mag_cov=LD.CovY)

        del LD
        if option.IterMaxDisp is None:
            option.IterMaxDisp=150
        else:
            option.IterMaxDisp=int(option.IterMaxDisp)

        SUGAR.Compute_Sugar_Model(MAX_iter_disp=option.IterMaxDisp,B_minus_V=None,DMz=None,COMUNICATOR=com,BLoc=option.Bloc,
                                  Alpha_init=None,Xplus_init=None,Grey_init=None,M0_init=None,
                                  COLOR=False,DELTA_M_GREY=option.grey,Cardelli_fit=False,jack=False,
                                  ADDDISP=option.dispersion,MAX_iter=N.inf,PARALLEL=option.Parallel)
        SUGAR.build_spectra()

        SUGAR.write_pkl(option.model)

    else:
        LD=Load_data(option.pca,option.spectra)
        ncomp=int(option.numbercomp)
        LD.compute_EM_PCA_data(ncomp)
        #########LD.compute_SI_norm_data(ncomp)


        Y_ERR=LD.Y_err
        wavelength=LD.X
        if option.color or option.ccm:
            Y=LD.Y_cosmo_corrected
        else:
            Y=LD.Y_cardelli_corrected_cosmo_corrected

        if option.grey:
            DMZ=None
        else:
            'CALIB added'
            DMZ=N.sqrt(LD.dm_z**2+0.03**2)
            
        if not option.intrinsic:
            LD.data=None
            LD.Cov_error=None

        SUGAR=SUGAR_model(Y,LD.data,LD.Cov_error,wavelength,LD.sn_name,Mag_err=Y_ERR)

        del LD
        
        SUGAR.Compute_Sugar_Model(B_minus_V=None,DMz=DMZ,COLOR=option.color,
                                  DELTA_M_GREY=option.grey,Cardelli_fit=option.ccm,
                                  ADDDISP=option.dispersion,MAX_iter=N.inf,EMFA_DISP_MATRIX=option.emfa_comp_disp)
        SUGAR.build_spectra()

        if  option.emfa:
            a=copy.deepcopy(option.model)
            a=a[:-4]+'_save_before_PCA.pkl'
            SUGAR.write_pkl(a)
            SUGAR.EM_PCA_residuals()
            
        SUGAR.write_pkl(option.model)
