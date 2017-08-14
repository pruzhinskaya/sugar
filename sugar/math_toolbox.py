"""math toolbox for sugar."""

from scipy import linalg
import numpy as np

def svd_inverse(matrix,return_logdet=False):
    """
    svd method using scipy.
    """
    U,s,V = linalg.svd(matrix)
    Filtre = (s>10**-15)
    if np.sum(Filtre)!=len(Filtre):
        print 'Pseudo-inverse decomposition :', len(Filtre)-np.sum(Filtre)

    inv_S = np.diag(1./s[Filtre])
    inv_matrix = np.dot(V.T[:,Filtre],np.dot(inv_S,U.T[Filtre]))

    if return_logdet:
        log_det=np.sum(np.log(s[Filtre]))
        return inv_matrix,log_det
    else:
        return inv_matrix

    
def cholesky_inverse(matrix,return_logdet=False):
    """
    Cholesky method using scipy.
    """
    L = linalg.cholesky(matrix, lower=True)
    inv_L = linalg.inv(L)
    inv_matrix = np.dot(inv_L.T, inv_L)

    if return_logdet:
        log_det=np.sum(2.*np.log(np.diag(L)))
        return inv_matrix,log_det
    else:
        return inv_matrix


def passage(norm_data1,norm_error1,vecteur_propre,sub_space=None):
    """
    project in emfa space.
    """
    norm_data=norm_data1.T
    norm_error=norm_error1.T

    if sub_space==None:
        Y=np.zeros(np.shape(norm_data))
        for sn in range(len(norm_data[0])):
            Y[:,sn]=np.dot(np.dot(np.linalg.inv(np.dot(vecteur_propre.T,np.dot(np.eye(len(norm_data))*(1./norm_error[:,sn]**2),vecteur_propre))),np.dot(vecteur_propre.T,np.eye(len(norm_data))*(1./norm_error[:,sn]**2))),norm_data[:,sn])

    else:
        Y=np.zeros((sub_space,len(norm_data[0])))
        vec_propre_sub_space=np.zeros((len(norm_data),sub_space))
        for vector in range(sub_space):
            vec_propre_sub_space[:,vector]=vecteur_propre[:,vector]
        for sn in range(len(norm_data[0])):
            Y[:,sn]=np.dot(np.dot(np.linalg.inv(np.dot(vec_propre_sub_space.T,np.dot(np.eye(len(norm_data))*(1./norm_error[:,sn]**2),vec_propre_sub_space))),np.dot(vec_propre_sub_space.T,np.eye(len(norm_data))*(1./norm_error[:,sn]**2))),norm_data[:,sn])

    return Y.T



def passage_error(norm_error1,vecteur_propre,sub_space,return_std=False):
    """
    project error. 

    set return std to true if you want just the
    diagonal part.
    """
    norm_error=norm_error1.T

    error_new_base=np.zeros((len(norm_error[0]),sub_space))
    cov_Y=np.zeros((len(norm_error[0]),sub_space,sub_space))

    vec_propre_sub_space=np.zeros((len(norm_error),sub_space))
    for vector in range(sub_space):
        vec_propre_sub_space[:,vector]=vecteur_propre[:,vector]

    for sn in range(len(norm_error[0])):
        cov_Y[sn]=np.linalg.inv(np.dot(vec_propre_sub_space.T,np.dot(np.eye(len(norm_error))*(1./norm_error[:,sn]**2),vec_propre_sub_space)))
        error_new_base[sn]=np.sqrt(np.diag(cov_Y[sn]))

    if return_std:
        return error_new_base
    else:
        return cov_Y
