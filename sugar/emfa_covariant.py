#!/usr/bin/env python

"""
A comment : empca with  error handling
Following "The EM Algorithm for Mixtures of Factors Analyzers"
Z. Ghahramani and G. Hinton (1997)
The computation has been adapted in the case of known measurement errors
   but not necessarily identical for all data

Model is x_i = \Lambda z_i + n_i
with z = {\cal N}(0,I)
and n_i = {\cal N}(0,\Psi)

For usage, see docstring for class EMPCA below
"""

import numpy as N
import scipy as S
from scipy import linalg
import copy




class EMfa_covariant:

    """
This is the wrapper class to store input data and perform computation
Usage:
solver = EMPCA(data,weights)
    data[nobs,nvar] contains the input data
    weights[nobs,nvar,nvar] contains the inverse variance of data
        note that currently the code support covariance between
        data for a single observation and covariance between observations
solver.converge(nvec,niter)
    nvec : number of reconstructed vectors
    niter : number of iterations
    will return the eigen values and eigenvectors as in numpy.linalg.eig
    It can be made reentrant with solver.converge(nvec,niter,Lambda_init=solver.Lambda)
"""

    def __init__(self, data, weights):
        assert len(N.shape(
            data)) == 2, "data shall be a 2-dim array, got %i dim" % \
            len(N.shape(data))
        assert len(N.shape(
            weights)) == 3 , "weights shall be a 3-dim array, got %i dim, if weights is diagonal used EMfa.py" % \
            len(N.shape(data))
        """import data and weights and perform pre-calculations"""
        self.X = N.array(data)  # ensures also a copy is performed in case
        self.nobs, self.nvar = N.shape(self.X)
        self.Psim1_org = N.array(weights)
        assert len(data) == len(weights) and len(data[0]) == len(weights[0]), 'data and weihgts shall have the same dimension'

        # first term of chi2 doesn't depend on any further computation
        
        

    def init_lambda(self, nvec, Lambda_init=None):
        """Initialize the lambda matrix and the fit parameters.
           Default is an educated guess based on traditional PCA
           if no Lambda_init is provided"""
        assert nvec <= self.nvar, "nvec shall be lower than nvar=%d" % self.nvar
        assert nvec <= self.nobs, "nvec shall be lower than nobs=%d" % self.nobs
        self.nvec = nvec

        #self.Psim1 = 1. / (1. / self.Psim1_org)
        self.Psim1 = self.Psim1_org
        self.Z = N.zeros((self.nobs, self.nvec))
        self.beta = N.zeros((self.nobs, self.nvec, self.nvar))
        # Esperance of Z^2
        self.Ezz = N.zeros((self.nobs, self.nvec, self.nvec))
        # definition in store_ILPL
        self.ILPL = N.zeros((self.nobs, self.nvec, self.nvec))
        self.ILPLm1 = N.zeros((self.nobs, self.nvec, self.nvec))

        if Lambda_init == None:
            # self.Lambda=N.eye(self.nvar,self.nvec)
            # some heuristic to find suitable starting value :
            # Set LLt to XXt
            # limiting the dimension to sustainable computation
            maxvals = N.min((self.nobs, self.nvar, N.max((100, nvec))))
            filt = N.arange(maxvals) * self.nvar / maxvals
            valp, vecp = N.linalg.eig(
                self.X[:, filt].T.dot(self.X[:, filt]) / len(self.X))
            order = N.argsort(N.real(valp)[::-1])
            # division by 10 helps convergence : better start with too small
            # Lambda
            self.Lambda = N.zeros((self.nvar, self.nvec))
            self.Lambda[filt] = N.real(
                N.sqrt(valp[order]) * vecp[:, order])[:, :nvec] / 10.
            # LZZtLt offers usually a better starting point than LLt
            self.solve_z()
            valp, vecp = self.orthogonalize_LZZL()
            self.Lambda = N.sqrt(valp) * vecp
        else:
            self.Lambda = copy.copy(Lambda_init)

    ##### helper functions #####

    def store_ILPL(self):
        """ store intermediate computation of ( I + \Lambda^T \Psi^-1 \Lambda )
        shall be performed each time self.Lambda has changed """

        for i in xrange(self.nobs):
            self.ILPL[i] = N.eye(
                self.nvec) + (N.dot(self.Lambda.T , self.Psim1[i])).dot(self.Lambda)
            # TODO : check possible improvements
            self.ILPLm1[i] = N.linalg.inv(self.ILPL[i])


    ##### Z solving #####

    def solve_z(self):
        """
        Z = \beta X
        with \beta = (I + \Lambda^T Psi^-1 \Lambda)^-1 \Lambda^T Psi^-1 X
                   = ILPLm1 \Lambda^T Psi^-1 X
        """
        self.store_ILPL()
        for i in xrange(self.nobs):
            # TODO : is it possible to improve with cholesky ?
            # ILPLm1 seems anyway to be invoked
            self.beta[i] = self.ILPLm1[i].dot(N.dot(self.Lambda.T, self.Psim1[i]))
            self.Z[i] = self.beta[i].dot(self.X[i])
            self.Ezz[i] = self.ILPLm1[i] + N.outer(self.Z[i], self.Z[i])



    ##### Lambda solving #####


    ##def compute_slopes(self):
    def solve_Lambda_full(self):
    
        X=copy.deepcopy(self.X)

        self.Lambda_vector=N.zeros(len(self.Lambda[:,0])*len(self.Lambda[0]))
        
        self.Ezz_kron_Psi=N.zeros((len(self.Lambda_vector),len(self.Lambda_vector)))
        

        self.sum_Psim1XTz=N.zeros(N.shape(self.Lambda))
        self.sum_Psim1XTz_vector=N.zeros(len(self.Lambda_vector))
        
        for sn in range(self.nobs):
            self.Ezz_kron_Psi+=N.kron(self.Ezz[sn],self.Psim1_org[sn])
            self.sum_Psim1XTz+=N.dot(self.Psim1[sn],N.dot(N.matrix(X[sn]).T,N.matrix(self.Z[sn])))
        
        # deroulage des matrices en vecteur
        for i in range(len(self.Lambda[0])):
            self.sum_Psim1XTz_vector[i*self.nvar:][:self.nvar]=self.sum_Psim1XTz[:,i]
        
        X_cho=linalg.cho_factor(self.Ezz_kron_Psi)
        
        self.Lambda_vector=linalg.cho_solve(X_cho,self.sum_Psim1XTz_vector)
        
        new_eigenvectors=N.zeros(N.shape(self.Lambda))
        
        for i in range(len(self.Lambda[0])):
            new_eigenvectors[:,i]=self.Lambda_vector[i*self.nvar:][:self.nvar]
        
        self.Lambda=new_eigenvectors
    


    def orthogonalize_Lambda(self):
        """ transform Lambda in an array of orthogonal vectors respecting LLt = Cte
        it uses as an intermediate representation : L = LxI = L' x A where L' is orthonormal
         and then gets the eigenvector representation of AAt """
        normed = N.zeros(N.shape(self.Lambda))
        alphas = N.eye(self.nvec)
        for i in xrange(self.nvec):
            sub = self.Lambda[:, i].dot(
                self.Lambda[:, :i]) / N.sum(self.Lambda[:, :i] ** 2, axis=0)
            self.Lambda[:, i] -= N.sum(sub * self.Lambda[:, :i], axis=1)
            alphas[:i, i] = sub
        norm = N.sqrt(N.sum(self.Lambda ** 2, axis=0))
        alphas = (alphas.T * norm).T
        self.Lambda /= norm
        valp, vecp = N.linalg.eig(alphas.dot(alphas.T))
        self.Lambda = self.Lambda.dot(vecp) * N.sqrt(valp)
        self.solve_z()
        return valp, self.Lambda / N.sqrt(valp)


    def orthogonalize_LZZL(self):
        """ returns the PCA eigenvalues and eigenvectors as in N.linalg.inv
        orthonormalization is performed with LZZ^TL^T, that is, not assuming Var(Z)=I
        This provides fast convergence in the case where noise is vanishingly small """

        Lambda = copy.copy(self.Lambda)
        Z = copy.copy(self.Z)
        for i in xrange(self.nvec):
            for j in xrange(i):
                scal = N.dot(Lambda[:, i], Lambda[:, j])
                Z[:, j] += scal * Z[:, i]
                Lambda[:, i] -= scal * Lambda[:, j]
            norm = N.sqrt(N.sum(Lambda[:, i] ** 2))
            Lambda[:, i] /= norm
            Z[:, i] *= norm
        # return Z,Lambda
        eigvals, eigvecs = N.linalg.eig(Z.T.dot(Z) / self.nobs)
        return eigvals, Lambda.dot(eigvecs)


    def comp_Q(self):

        Q=0
        for sn in range(self.nobs):
            Q+=-2.*N.dot(self.X[sn],self.Psim1[sn]).dot(N.dot(self.Lambda,N.array(N.matrix(self.Z[sn]).T)))
            Q+=N.sum(N.diag((N.eye(self.nvec)+N.dot(self.Lambda.T,self.Psim1[sn]).dot(self.Lambda)).dot(self.Ezz[sn])))

        return Q
 
    ##### Chi2 and Log-L determination #####

    def log_likelihood(self, z_solved=False):
        """ Computes the log-likelihood which is minimized : x = Normal ( 0, LLt + Psi )
        if Z is already solved for, the computation is faster
        returns the Log-L and the associated chi2"""

        self.det_i = N.zeros(self.nobs)
        self.chi2_i = N.zeros(self.nobs)
        if z_solved:
            for i in xrange(self.nobs):
                # first term : log determinant of psi
                # protect against missing data (Psim1==0)
                val=N.linalg.eigvals(self.Psim1[i])
                self.det_i[i] = N.sum(N.log(val)) /2.
                #self.det_i[i] = N.log(N.linalg.det(self.Psim1[i])) /2.
                self.det_i[i] -= N.log(N.linalg.det(self.ILPL[i])) / 2.
                # observing that x^T (LLt+Psi)-1 x = x^T Psi-1 ( x - Lz )
                # this assumes z is "solved" (or converged) for L
                self.chi2_i[i] -= (N.dot(self.X[i],self.Psim1[i])).dot(
                    self.X[i] - self.Lambda.dot(self.Z[i])) / 2
        else:
            self.store_ILPL()
            for i in xrange(self.nobs):
                # first term : log determinant of psi

                #direct (slow) computation
                #mat = N.linalg.inv(self.Psim1[i]) + self.Lambda.dot(self.Lambda.T)
                #self.det_i[i] -= N.log(N.linalg.det( mat )) /2.
                #self.det_i[i] -= self.X[i] .dot( N.linalg.inv(mat) ).dot(self.X[i]) /2.

                self.det_i[i] -=N.log(N.linalg.det(self.Psim1[i] + self.Lambda.dot(self.Lambda.T)))
                
                # by me
                #self.det_i[i] = N.log(N.linalg.det(self.Psim1[i])) / 2.
                #self.det_i[i] -= N.log(N.linalg.det(self.ILPL[i])) / 2.

                # general case (slower):
                # matrix inversion lemma on (LLt + Psi)-1 ...
                # there is a '-' sign for the second term
                # self.store_ILPL()
                W = N.linalg.inv(self.Psim1[i] + self.Lambda.dot(self.Lambda.T))
                self.chi2_i[i] -= self.X[i].dot( W ).dot(self.X[i]) /2.
                
                #self.chi2_i[i] -= N.dot(self.X[i], self.Psim1[i]).dot(N.array(N.matrix(self.X[i]).T)) / 2.
                #projX = N.dot(self.Lambda.T,self.Psim1[i]).dot(self.X[i])
                #self.chi2_i[i] += projX.dot(self.ILPLm1[i].dot(projX)) / 2.

        return N.sum(self.det_i + self.chi2_i), -2 * N.sum(self.chi2_i)

 

    ##### Main Loop #####

    def converge(self, nvec, dQ=10**(-4),niter=N.inf, Lambda_init=None, verbose=False):
        """ looks after the nvec best vectors explaining the denoised data
        variance
        Nvec : number of reconstructed vectors
        Niter : number of iterations
        dQ : convergence criteria
        Lambda_init : initial guess for Lambda
        verbose : if set to True, turns on verbose output mode (slower) """

        self.verbose = verbose

        self.init_lambda(nvec, Lambda_init)

        Lambda_0 = None
        old_log_L = -N.inf
        self.log_L=[]
        self.Q=[]
        

        Q=self.comp_Q()
        self.Q.append(Q+1)
        self.Q.append(Q)
        
        i=0
        #for i in xrange(niter + 1):
        while self.Q[i]-self.Q[i+1]>dQ:

            # E-step
            self.solve_z()

            log_L, chi2_tot = self.log_likelihood(z_solved=True)
            
            # check convergence is all right
            if log_L < old_log_L:
                self.Lambda = Lambda_after_solve
                self.solve_z()
                log_L, chi2_tot = self.log_likelihood(z_solved=True)

                
            print "%2d logL=%f Chi2=%f " % (i, log_L, chi2_tot)
                
            self.log_L.append(log_L)
            if i == niter:
                break

            # M-step
            old_log_L = log_L
            self.solve_Lambda_full()
            QQ=self.comp_Q()
            self.Q.append(QQ)
            i+=1
            Lambda_after_solve = copy.copy(self.Lambda)


        # in some cases ( no noise in input data on a given hyperplane) the solution is not optimal
        # This should not happen given the initial conditions, but I leave the
        # code in case
        """
        save_Lambda = copy.copy(self.Lambda)
        valp,vecp=self.orthogonalize_LZZL()
        self.Lambda= vecp*N.sqrt(valp)
        log_L_zzl,chi2_tot = self.log_likelihood()
        if log_L_zzl > log_L :
            print "%2d/%d logL=%f Chi2=%f "%(i,niter,log_L_zzl,chi2_tot)
            return valp,vecp
        else:
            self.Lambda=save_Lambda
            return self.orthogonalize_Lambda()
        """
        # return self.orthogonalize_Lambda()



if __name__=="__main__":

    import cPickle
    from EMfa import EMPCA as ManuPC
    import pylab as P
    
    dic=cPickle.load(open('pca_all_data.pkl'))

    data=dic['Norm_data'][dic['filter']][:,:2]
    error=dic['Norm_err'][dic['filter']][:,:2]
    vec=dic['vec']
    #vec=N.eye(2)
    A=N.pi/8.
    vec=N.array(([N.cos(A),-N.sin(A)],[N.sin(A),N.cos(A)]))
   
    
    weights=1./error**2
    WEIGHTS=N.zeros((len(data[:,0]),len(data[0]),len(data[0])))
    data_rot=N.zeros(N.shape(data))

    for sn in range(len(data[:,0])):
        data_rot[sn]=N.array(N.dot(vec,N.matrix(data[sn]).T).T)
        WEIGHTS[sn]=N.diag(error[sn]**2)
        WEIGHTS[sn]=N.linalg.inv((N.dot(vec,WEIGHTS[sn]).dot(vec.T)))
        weights[sn]=N.diag(WEIGHTS[sn])
    
    emfa=ManuPC(data,weights)
    emfa.converge(1,niter=1000,center=False,accelerate=False,gradient=False,renorm=False)
    eigenvectors=emfa.Lambda
    projections_in_sub_space=emfa.Z
    

    Emfa=EMfa_covariant(data_rot,WEIGHTS)
    Emfa.converge(1,niter=N.inf)
    Eigenvectors=Emfa.Lambda
    Projections_in_sub_space=Emfa.Z

    
    

    Emfa.Lambda[0]=N.array(N.dot(vec.T,N.matrix(Emfa.Lambda[0]).T).T)
    #Emfa.Lambda[:,1]=N.array(N.dot(vec.T,N.matrix(Emfa.Lambda[:,1]).T).T)
        
    P.scatter(data[:,0],data[:,1],c='b')
    P.plot([0,Emfa.Lambda[0]],[0,Emfa.Lambda[1]],'b')
    #P.plot([0,Emfa.Lambda[0,1]],[0,Emfa.Lambda[1,1]],'b')

    P.scatter(data[:,0],data[:,1],c='r')
    P.plot([0,emfa.Lambda[0]],[0,emfa.Lambda[1]],'r')
    #P.plot([0,emfa.Lambda[0,1]],[0,emfa.Lambda[1,1]],'r')

    P.figure()

    P.plot(-N.array(Emfa.log_L))
    P.plot(-N.array(emfa.log_L))

    P.figure()
    P.plot(Emfa.Q)

