"""all command to excute to buil sugar model."""

import sugar

if __name__=='__main__':

    # gaussian process part

    gp = sugar.gp_sed(hsiao_empca=True)
    gp.gaussian_process_regression()
    gp.write_output()

    # emfa on spectral features

    sugar.run_emfa_analysis(output_file=None,
                            sigma_clipping=False)

