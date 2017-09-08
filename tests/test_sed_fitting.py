"""test odr fitting."""

import numpy as np
import os
import sugar


def generate_fake_sed(NSN,plot=False):
    """
    generate fake data to test algorithm.
    """
    np.random.seed(1)
    wavelength = np.linspace(0,100,100)
    scale = np.array(([3,0,0],
                      [0,2,0],
                      [0,0,1]))
    
    x = np.random.multivariate_normal(np.zeros(3),cov=scale,size=NSN)
    x_err_ = np.random.multivariate_normal(np.zeros(3),cov=scale*0.05,size=NSN)

    y = np.zeros((NSN,len(wavelength)))
    y_err = np.zeros((NSN,len(wavelength)))
    x_err = np.zeros((NSN,3,3))

    alpha = np.zeros((3,len(wavelength)))
    alpha[0] = 5. * np.sin(wavelength)
    alpha[1] = 0.1 * wavelength
    alpha[2] = 15. * np.exp(-0.5*((wavelength-50.)/5.)**2)
    
    for i in range(NSN):
        y[i] = x[i,0]*alpha[0] + x[i,1]*alpha[1] + x[i,2]*alpha[2]
        y_err[i] = np.random.normal(scale=np.std(y[i])*0.1,size=len(y[i]))
        y[i] += y_err[i]
        x[i] += x_err_[i]
        x_err[i] = np.eye(3)*x_err_[i]**2
        y_err[i] = np.sqrt(y_err[i]**2)

    if plot:
        import pylab as plt
        plt.figure()
        plt.subplot(2,1,1)
        for i in range(NSN):
            plt.plot(wavelength,y[i],'r',linewidth=3,alpha=0.5)
        plt.subplot(2,1,2)
        plt.plot(wavelength,np.std(y,axis=0),'r',linewidth=3)
    
    return y, y_err, x, x_err,wavelength,alpha


def test_init(plot=False):
    """
    test initialisation of sed_fitting
    """
    nsn = 1000
    y, y_err, x, x_err, wave, alpha_truth = generate_fake_sed(nsn,plot=False)

    y_corrected = np.zeros_like(y)
    alpha = np.zeros_like(alpha_truth)
    m0 = np.zeros_like(wave)
    
    for i in range(len(wave)):
        print i+1,'/',len(wave)
        mlf = sugar.Multilinearfit(x,y[:,i],yerr=y_err[:,i],covx=x_err)
        mlf.Multilinearfit(adddisp=False,PRINT=False)
        y_corrected[:,i] = mlf.y_corrected
        alpha[:,i] = mlf.alpha
        m0[i] = mlf.M0

    if plot:
        import pylab as plt
        plt.figure()
        plt.subplot(2,1,1)
        for i in range(nsn):
            plt.plot(wave,y_corrected[i],'b',linewidth=3,alpha=0.5)
        plt.subplot(2,1,2)
        plt.plot(wave,np.std(y,axis=0),'r',linewidth=3)
        plt.plot(wave,np.std(y_corrected,axis=0),'b',linewidth=3)
                                                        
        plt.figure(figsize=(8,8))
        for i in range(3):
            plt.subplot(4,1,i+1)
            plt.plot(wave,alpha_truth[i],'r',linewidth=5)
            plt.plot(wave,alpha[i],'b',linewidth=3)
        plt.subplot(4,1,4)
        plt.plot(wave,np.zeros_like(wave),'r',linewidth=5)
        plt.plot(wave,m0,'b',linewidth=3)
        
if __name__=='__main__':


    #y, y_err, x, x_err, wave, alpha_truth = generate_fake_sed(100,plot=False)
    #i = 1
    #mlf = sugar.Multilinearfit(x,y[:,i],covx=x_err,yerr=y_err[:,i])
    #mlf.Multilinearfit(adddisp=True,PRINT=False)
             
    
    import pylab as plt
    test_init(plot=True)
