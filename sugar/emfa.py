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
import copy


class EMPCA:
    """
This is the wrapper class to store input data and perform computation
Usage:
solver = EMPCA(data,weights)
    data[nobs,nvar] contains the input data
    weights[nobs,nvar] contains the inverse variance of data
        note that currently the code doesn't support covariance between
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
            weights)) == 2, "weights shall be a 2-dim array, got %i dim" % \
            len(N.shape(data))
        """import data and weights and perform pre-calculations"""
        self.X = N.array(data)  # ensures also a copy is performed in case
        self.nobs, self.nvar = N.shape(self.X)
        self.Psim1_org = N.array(weights)
        self.Psi0 = N.zeros(self.nvar)
        assert N.shape(data) == N.shape(
            weights), 'data and weihgts shall have the same dimension'
        assert N.all(weights >= 0), 'weights shall all be positive'
        # first term of chi2 doesn't depend on any further computation

    def init_lambda(self, nvec, Lambda_init=None):
        """Initialize the lambda matrix and the fit parameters.
           Default is an educated guess based on traditional PCA
           if no Lambda_init is provided"""
        assert nvec <= self.nvar, "nvec shall be lower than nvar=%d" % self.nvar
        assert nvec <= self.nobs, "nvec shall be lower than nobs=%d" % self.nobs
        self.nvec = nvec

        self.Psim1 = 1. / (1. / self.Psim1_org + self.Psi0)
        self.Z = N.zeros((self.nobs, self.nvec))
        self.beta = N.zeros((self.nobs, self.nvec, self.nvar))
        # Esperance of Z^2
        self.Ezz = N.zeros((self.nobs, self.nvec, self.nvec))
        # definition in store_ILPL
        self.ILPL = N.zeros((self.nobs, self.nvec, self.nvec))
        self.ILPLm1 = N.zeros((self.nobs, self.nvec, self.nvec))

        if Lambda_init is None:
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
                self.nvec) + (self.Lambda.T * self.Psim1[i]).dot(self.Lambda)
            # TODO : check possible improvements
            self.ILPLm1[i] = N.linalg.inv(self.ILPL[i])

    def store_LLPm1(self):
        """ LLPm1 is (LLt + Psi)^-1
        WARNING : this assumes ILPL is already stored !
        TODO : this poses storage issue in memory -> shall be deprecated"""
        # taking advantage of Psim1 diagonality and Woodbury matrix inversion
        self.LLPm1 = N.zeros((self.nobs, self.nvar, self.nvar))
        for i in xrange(self.nobs):
            self.LLPm1[i] = N.diag(self.Psim1[i]) - \
                            (self.Lambda.dot(self.ILPLm1[i]).dot(self.Lambda.T)\
                             * self.Psim1[i]).T * self.Psim1[i]

    def get_LLPm1(self, i):
        """ LLPm1 is (LLt + Psi)^-1
        WARNING : this assumes ILPL is already stored ! """
        # taking advantage of Psim1 diagonality and Woodbury matrix inversion
        return N.diag(self.Psim1[i]) - \
               (self.Lambda.dot(self.ILPLm1[i]).dot(self.Lambda.T) * \
                self.Psim1[i]).T * self.Psim1[i]

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
            self.beta[i] = self.ILPLm1[i].dot(self.Lambda.T * self.Psim1[i])
            self.Z[i] = self.beta[i].dot(self.X[i])
            self.Ezz[i] = self.ILPLm1[i] + N.outer(self.Z[i], self.Z[i])

    def center_and_solve_z(self):
        """like solve_z but in addition recenter X
        employing (LLt+Psi)^-1 = Psi^-1 (L beta) as the weights
        This is slow as it has to explore the full nvar space"""
        # it takes advantage of the beta computation in solve_z to streamline
        # the 2 operations

        self.store_ILPL()
        Deltax = N.zeros(self.nvar)
        Weights = N.zeros((self.nvar, self.nvar))
        for i in xrange(self.nobs):
            self.beta[i] = self.ILPLm1[i].dot(self.Lambda.T * self.Psim1[i])
            # speeding-up as Psim1 is diagonal
            w = self.Psim1[i] * \
                (N.eye(self.nvar) - self.Lambda.dot(self.beta[i])).T
            Deltax += w.dot(self.X[i])
            Weights += w
        # cholesky needed (dimensionality of x)
        chof = S.linalg.cho_factor(Weights)
        center = S.linalg.cho_solve(chof, Deltax)
        self.X -= center
        for i in xrange(self.nobs):
            self.Z[i] = self.beta[i].dot(self.X[i])
            self.Ezz[i] = self.ILPLm1[i] + N.outer(self.Z[i], self.Z[i])

    ##### Lambda solving #####

    def solve_Lambda(self):
        """ if all errors were identical for all supernova (i.e. Psi doesn't
        depend on i)
        self.Lambda = self.X.T.dot(self.Z).dot( N.linalg.inv(N.sum(self.Ezz,axis=0)))
         For different errors : solve line by line """
        for j in xrange(self.nvar):
            toinvert = N.tensordot(self.Psim1[:, j], self.Ezz, axes=([0], [0]))
            self.Lambda[j] = N.dot(
                self.Psim1[:, j] * self.X[:, j], self.Z) .dot(N.linalg.inv(toinvert))

    """ solved only on conditional likelihood. Direct method is however
    preferable
    def solve_Psi0(self):
        # solves for unaccountded error : mantatory to help the 0-error case to converge.
        # but only for diagonal elements
        diagxxlz = self.X * ((self.X-self.Lambda.dot(self.Z.T).T))
        for i in xrange(5):
            f = - N.sum(self.Psim1,axis=0) + N.sum( diagxxlz * self.Psim1**2,axis=0)
            df = N.sum(self.Psim1**2,axis=0) - 2*N.sum( diagxxlz * self.Psim1**3,axis=0)
            self.Psi0 += -f/df
            self.Psi0[self.Psi0<0]=0.
            print self.Psi0
            self.Psim1 = 1./(1./ self.Psim1_org + self.Psi0)
            """

    def accelerate_Lambda(self, dLambda,  helpstring="Accelerate"):
        """ check if Lambda + alpha dLambda is a better solution
        and apply it if true"""
        self.store_ILPL()
        # self.store_LLPm1()
        f = 0  # half of the log derivative
        df = 0
        for i in xrange(self.nobs):
            LLPm1 = self.get_LLPm1(i)
            LML = self.Lambda.T.dot(LLPm1).dot(self.Lambda)
            LMd = self.Lambda.T.dot(LLPm1).dot(dLambda)
            dMd = dLambda.T.dot(LLPm1).dot(dLambda)
            XML = self.X[i].dot(LLPm1).dot(self.Lambda)
            XMd = self.X[i].dot(LLPm1).dot(dLambda)
            f += - N.trace(LMd) + XML.dot(XMd.T)
            df += - N.trace(dMd) + N.trace(LML.dot(dMd) + LMd.dot(LMd)) + XMd.dot(XMd.T) - \
                XMd.dot(LML).dot(XMd.T) - 2 * \
                XML.dot(LMd.T).dot(XMd.T) - XML.dot(dMd).dot(XML.T)
        if f * df < 0:
            if self.verbose:
                print helpstring + " : %f" % (-f / df)
            self.Lambda -= dLambda * f / df
        return f, df  # useful for debug purposes


    def normalize_Lambda(self):
        """ Newton-Raphson Likelihood maximization on alpha_i L_i L_i^t
        i.e. solve for L vector norms of columns L_i """
        self.store_ILPL()
        # self.store_LLPm1()
        self.LtLLPm1L = N.zeros((self.nobs, self.nvec, self.nvec))
        self.XtLLPm1L = N.zeros((self.nobs, self.nvec))
        f = N.zeros(self.nvec)
        df = N.zeros((self.nvec, self.nvec))
        for i in xrange(self.nobs):
            LLPm1 = self.get_LLPm1(i)
            self.LtLLPm1L[i] = self.Lambda.T.dot(LLPm1).dot(self.Lambda)
            self.XtLLPm1L[i] = self.X[i].dot(LLPm1).dot(self.Lambda)
            f += - N.diag(self.LtLLPm1L[i]) + self.XtLLPm1L[i] ** 2
            df += self.LtLLPm1L[i] ** 2 - 2 * \
                N.outer(self.XtLLPm1L[i], self.XtLLPm1L[i]) * self.LtLLPm1L[i]
        # df symetric, but not pos-def
        alpha = N.linalg.solve(df, -f)
        prettyalpha = copy.copy(alpha)

        # protecting the square root
        alpha[alpha <= -1] = - N.exp(alpha[alpha <= -1])

        # enforcing move in the "right" direction
        self.Lambda[:, alpha * f > 0] *= N.sqrt(1 + alpha[alpha * f > 0])
        prettyalpha[alpha * f <= 0] = 0
        if self.verbose:
            print prettyalpha
        return f, df, alpha  # for convergence purposes

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

    ##### Psi solving #####

    def solve_Psi0(self):
        """ fit for an additional component in Psi
        This step is mandatory for Factor Analysis, but optional if there is
        already a good guess for Psi
        TODO : this routine is not speed-optimized"""
        # function to minimize

        self.store_ILPL()

        f = 0
        df = 0
        # solving only for the diagonal part
        for i in xrange(self.nobs):
            LLPm1 = self.get_LLPm1(i)
            LLPm1X = N.dot(LLPm1, self.X[i])
            f += N.diag(-LLPm1) + LLPm1X ** 2
            df += LLPm1 ** 2 - 2 * N.outer(LLPm1X, LLPm1X) * LLPm1

        deltaPsi0 = - N.linalg.solve(df, f)
        self.Psi0[deltaPsi0 * f > 0] += deltaPsi0[deltaPsi0 * f > 0]
        self.Psi0[self.Psi0 < 0] = 0.
        if self.verbose:
            print self.Psi0
        self.Psim1 = 1. / (1. / self.Psim1_org + self.Psi0)

    ##### Chi2 and Log-L determination #####

    def chi2(self):
        """ expected chi2 which is minimized by the M-step
        Warning: this chi2 is NOT the term that is minimized by the overall procedure """
        self.chi2_i = N.zeros(self.nobs)
        for i in xrange(self.nobs):
            self.chi2_i = (self.X[i].T * self.Psim1[i]).dot(self.X[i])
            self.chi2_i[
                i] += -2 * (self.X[i].T * self.Psim1[i]).dot(self.Lambda.dot(self.Z[i]))
            self.chi2_i[i] += N.trace(self.ILPL[i].dot(self.Ezz[i]))
        return N.sum(self.chi2_i)

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
                #self.det_i[i] = N.sum(N.log(self.Psim1[i])) /2.
                self.det_i[i] = N.sum(N.log(
                    self.Psim1[i][self.Psim1[i] > 0])) / 2.
                self.det_i[i] -= N.log(N.linalg.det(self.ILPL[i])) / 2.
                # observing that x^T (LLt+Psi)-1 x = x^T Psi-1 ( x - Lz )
                # this assumes z is "solved" (or converged) for L
                self.chi2_i[i] -= (self.X[i] * self.Psim1[i]).dot(
                    self.X[i] - self.Lambda.dot(self.Z[i])) / 2
        else:
            self.store_ILPL()
            for i in xrange(self.nobs):
                # first term : log determinant of psi
                self.det_i[i] = N.sum(N.log(self.Psim1[i])) / 2.
                self.det_i[i] -= N.log(N.linalg.det(self.ILPL[i])) / 2.
                # general case (slower):
                # matrix inversion lemma on (LLt + Psi)-1 ...
                # there is a '-' sign for the second term
                # self.store_ILPL()
                self.chi2_i[i] -= N.sum(self.X[i] ** 2 * self.Psim1[i]) / 2.
                projX = (self.Lambda.T * self.Psim1[i]).dot(self.X[i])
                self.chi2_i[i] += projX.dot(self.ILPLm1[i].dot(projX)) / 2.

                # direct (slow) computation
                #mat = N.diag(1./self.Psim1[i]) + self.Lambda.dot(self.Lambda.T)
                #self.det_i[i] -= N.log(N.linalg.det( mat )) /2.
                #self.det_i[i] -= self.X[i] .dot( N.linalg.inv(mat) ).dot(self.X[i]) /2.
        return N.sum(self.det_i + self.chi2_i), -2 * N.sum(self.chi2_i)

    def grad_log_likelihood(self):
        """ computes the gradient of log_likelihood """
        self.solve_z()
        # self.store_LLPm1()
        gradient = - N.sum(self.beta, axis=0).T
        for i in xrange(self.nobs):
            LLPm1 = self.get_LLPm1(i)
            gradient += N.outer(LLPm1.dot(self.X[i]), self.Z[i])
        return gradient

    ##### Main Loop #####

    def converge(self, nvec, niter=100, Lambda_init=None, center=False,
                 solve_Psi0=False, renorm=True, accelerate=True,
                 gradient=True, verbose=False):
        """ looks after the nvec best vectors explaining the denoised data
        variance
        Nvec : number of reconstructed vectors
        Niter : number of iterations
        Lambda_init : initial guess for Lambda
        Center : if set to True, the centering will be performed at each
        iteration (slows the convergence)
        update_Psi0 : if set to True, Psi0 will be also converged for
        renorm : if set to True (default) normalize Lambda vectors to
        speed-up convergence
        accelerate : if set to True (default) speeds-up convergence by
        extrapolating Lambda in the direction given by successive iterations
        gradient : if set to True (default) adds a step of gradient descent
        in the convergence
        verbose : if set to True, turns on verbose output mode (slower) """

        self.verbose = verbose

        if solve_Psi0:
            self.Psi0 = N.zeros(self.nvar)

        self.init_lambda(nvec, Lambda_init)

        if solve_Psi0:
            self.solve_Psi0()

        Lambda_0 = None
        old_log_L = -N.inf

        for i in xrange(niter + 1):

            # E-step
            if center:
                self.center_and_solve_z()
            else:
                self.solve_z()
            log_L, chi2_tot = self.log_likelihood(z_solved=True)

            # check convergence is all right
            if log_L < old_log_L:
                self.Lambda = Lambda_after_solve
                self.solve_z()
                log_L, chi2_tot = self.log_likelihood(z_solved=True)
            print "%2d/%d logL=%f Chi2=%f " % (i, niter, log_L, chi2_tot)

            if i == niter:
                break

            # M-step
            old_log_L = log_L
            self.solve_Lambda()
            if solve_Psi0:
                self.solve_Psi0()

            Lambda_after_solve = copy.copy(self.Lambda)

            if accelerate and Lambda_0 is not None:
                dLambda = self.Lambda - Lambda_0
                self.accelerate_Lambda(dLambda, "Delta L")
                Lambda_0 = None
            else:
                Lambda_0 = copy.copy(self.Lambda)

            if gradient:
                self.accelerate_Lambda(self.grad_log_likelihood(), "Gradient")

            if renorm:
                self.normalize_Lambda()


