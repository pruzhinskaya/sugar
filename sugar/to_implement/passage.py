import numpy as N


def passage(Data,eigenvector,Error=None,size_space=None):
    """

    - Data: array_like
    Data have N lines and n collones.
    Where N are the numbers of points/number of supernovae and n the number of components
          
    - eigenvector: array_like
    rotation matrix
    eigenvector have n lines and n collones.
    Where n is the number of components    

    -Error: array_like
    Error of Data
    Data have N lines and n collones.
    Where N are the numbers of points/number of supernovae and n the number of components                                                                   
    - size_space: int
    if you want to reduce the size of the space put the size here
    size_space need to be lower than n.

    Use:

    data_rotate=passage(Data,eigenvector,Error=None,size_space=None)
    
    output --> array_like
    data_rotate have N lines and n collones if size_space is None or size_space collones if size_space is not None


    """


    data=Data.T
    if Error is None: 
        error=N.ones(N.shape(Data.T))
    else:
        error=Error.T

    if size_space is None:
        Y=N.zeros(N.shape(data))
        for sn in range(len(data[0])):
            Y[:,sn]=N.dot(N.dot(N.linalg.inv(N.dot(eigenvector.T,N.dot(N.eye(len(data))*(1./error[:,sn]**2),eigenvector))),N.dot(eigenvector.T,N.eye(len(data))*(1./error[:,sn]**2))),data[:,sn])

    else:
        Y=N.zeros((size_space,len(data[0])))
        vec_propre_sub_space=N.zeros((len(data),size_space))
        for vector in range(size_space):
            vec_propre_sub_space[:,vector]=eigenvector[:,vector]
        for sn in range(len(data[0])):
            Y[:,sn]=N.dot(N.dot(N.linalg.inv(N.dot(vec_propre_sub_space.T,N.dot(N.eye(len(data))*(1./error[:,sn]**2),vec_propre_sub_space))),N.dot(vec_propre_sub_space.T,N.eye(len(data))*(1./error[:,sn]**2))),data[:,sn])

    return Y.T



def passage_error(Error,eigenvector,sub_space=None):
    """

    -Error: array_like
    Error of Data
    Data have N lines and n collones.
    Where N are the numbers of points/number of supernovae and n the number of components                                                                   
    - eigenvector: array_like
    rotation matrix
    eigenvector have n lines and n collones.
    Where n is the number of components    

    
    - size_space: int
    if you want to reduce the size of the space put the size here
    size_space need to be lower than n.

    Use:

    error_rotate=passage_error(Error,eigenvector,sub_space=None)
    
    output --> array_like
    error_rotate have N matrix n*n if sub_space is None or N matrix sub_space*sub_space if sub_space is not None

    """

    if sub_space is None:
        sub_space=len(Error[0])

    error=Error.T

    cov_Y=N.zeros((len(error[0]),sub_space,sub_space))

    vec_propre_sub_space=N.zeros((len(error),sub_space))
    for vector in range(sub_space):
        vec_propre_sub_space[:,vector]=eigenvector[:,vector]

    for sn in range(len(error[0])):
        cov_Y[sn]=N.linalg.inv(N.dot(vec_propre_sub_space.T,N.dot(N.eye(len(error))*(1./error[:,sn]**2),vec_propre_sub_space)))

    return cov_Y




if __name__=='__main__':

    X=N.random.normal(size=100)
    Y=X
    Z=X

    Data=N.array([X,Y,Z]).T
    Error=N.array([N.ones(3)]*100)

    eigval,eigvec=N.linalg.eig(1./(100.-1)*N.dot(Data.T,Data))

    Data_in_new_base=passage(Data,eigvec,Error=Error,size_space=None)
    cov_in_new_base=passage_error(Error,eigvec,sub_space=None)

    Data_in_XY_plane=passage(Data,eigvec,Error=Error,size_space=2)
    cov_in_XY_plane=passage_error(Error,eigvec,sub_space=2)






