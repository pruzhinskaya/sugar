"""all command to excute to buil sugar model."""

import sugar
import os

if __name__=='__main__':

    # gaussian process part

    gp = sugar.gp_sed(hsiao_empca=True)
    gp.gaussian_process_regression()
    gp.write_output()

    # emfa on spectral features

    sugar.run_emfa_analysis(output_file=None,
                            sigma_clipping=False)

    # TO DO: implement Rv fitting (done at ccin2p3 for the moment)

    
    # full sed fitting

    path = os.path.dirname(sugar.__file__)
    pca = path + '/data_output/sugar_paper_output/emfa_3_sigma_clipping.pkl'
    gp = path + '/data_output/gaussian_process/gp_predict/'
    max_light = path + '/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl'

    ms = sugar.make_sugar(pca, max_light, gp, filtre=True)
    ms.launch_sed_fitting()
    ms.write_model_output()
