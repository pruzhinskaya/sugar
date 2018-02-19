"""all command to excute to buil sugar model."""

import sugar
import os

class build_model:
    def __init__(self, path_input='data_input/', path_output='data_output/', path_output_gp='data_output/gaussian_process/'):
        """
        run ma poule!
        """
        # gaussian process part

        ##gp = sugar.gp_sed(path_input = path_input, average=True, double_average=False)
        ##gp.gaussian_process_regression()
        ##gp.write_output(path_output = path_output_gp)

        # emfa on spectral features

        ##sugar.run_emfa_analysis(path_input, path_output,
        ##                        sigma_clipping=True)

        # TO DO: implement Rv fitting (done at ccin2p3 for the moment)

    
        # full sed fitting

        #path = os.path.dirname(sugar.__file__)
        #pca = path + '/data_output/sugar_paper_output/emfa_3_sigma_clipping.pkl'
        #gp = path + '/data_output/gaussian_process/gp_predict/'
        #max_light = path + '/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl'

        ms = sugar.make_sugar(path_output=path_output, path_output_gp=path_output_gp, filtre=True)
        ms.launch_sed_fitting()
        ms.write_model_output()
