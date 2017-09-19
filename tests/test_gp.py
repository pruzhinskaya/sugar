
from scipy.sparse import block_diag
from scipy.sparse import coo_matrix
import numpy as np
import cPickle
import sugar


dic = cPickle.load(open('../sugar/data_output/gaussian_process/sed_snia_gaussian_process.pkl'))


pca = '../sugar/data_output/sugar_paper_output/emfa_3_sigma_clipping.pkl'
gp = '../sugar/data_output/gaussian_process/gp_predict/'
max_light = '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl'
ld = sugar.make_sugar(pca, max_light, gp, filtre=True)

sn = ld.sn_name[0]

cov = []
for i in range(190):
    cov.append(coo_matrix(dic[sn]['bin%i'%i]['covariance']))

covy = block_diag(cov)

