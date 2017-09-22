
import pylab as plt
from scipy.sparse import block_diag
from scipy.sparse import coo_matrix
import numpy as np
import cPickle
import sugar
import cosmogp


dic = cPickle.load(open('../../Desktop/test_gp/gp_output/Prediction_Bin_4.pkl'))

gp = sugar.gaussian_process.load_data_bin_gp()
gp.build_difference_mean(hsiao_empca=True) # --> need to be carefull on the diff construction 
gp.load_data_bin(4)
gp.load_mean_bin(4, hsiao_empca=True)
#

gpr = cosmogp.gaussian_process_nobject(gp.y, gp.time, kernel='RBF1D',
                                       y_err=gp.y_err, diff=gp.diff, Mean_Y=gp.mean,
                                       Time_mean=gp.mean_time, substract_mean=False)

#gpr = cosmogp.gaussian_process_nobject(dic['y'], dic['y_time'], kernel='RBF1D',
#                                       y_err=dic['y_err'], diff=dic['diff'], Mean_Y=dic['mean'],
#                                       Time_mean=dic['mean_time'], substract_mean=False)

gpr.nugget = 0.03

#gpr.find_hyperparameters(hyperparameter_guess=[0.5,8.], nugget=False, svd_method=False)
gpr.hyperparameters = [dic['Hyperparametre']['sigma'], dic['Hyperparametre']['l']]
gpr.get_prediction(new_binning=np.linspace(-12,42,19), COV=True, svd_method=False)



#ptfnew = np.loadtxt('../sugar/data_output/gaussian_process/gp_predict/PTF09dnl.predict')
#ptfold = np.loadtxt('../../Desktop/test_gp/gp_predict/PTF09dnl.predict')


#for i in range(len(ptfnew[0])):
#    plt.figure()
#    plt.plot(ptfnew[:,i]-ptfold[:,i])


#dic = cPickle.load(open('../sugar/data_output/gaussian_process/sed_snia_gaussian_process.pkl'))
#pca = '../sugar/data_output/sugar_paper_output/emfa_3_sigma_clipping.pkl'
#gp = '../sugar/data_output/gaussian_process/gp_predict/'
#gp_old = '../sugar/data_output/Prediction_GP_predict/'

#max_light = '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl'
#ld_old = sugar.make_sugar(pca, max_light, gp_old, filtre=True)
#ld_old.launch_sed_fitting()

#ld = sugar.make_sugar(pca, max_light, gp, filtre=True)
#ld.launch_sed_fitting()

#for i in range(len(ld.sedfit.wy)):
#    plt.plot(ld.sedfit.wy[i].diagonal())
#    if np.sum(ld.sedfit.wy[i].diagonal())<0:
#        print i

