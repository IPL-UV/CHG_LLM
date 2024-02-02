import argparse 
import os
from ast import literal_eval

import pandas as pd
import numpy as np
from plotnine import * 

def load_predictions(where):
    data = None
    if os.path.isfile(where):
        data = pd.read_csv(where, na_values="")
    else:
        content = os.listdir(where)
        print(content)
        for d in content:
            data1 = None
            if os.path.isdir(os.path.join(where,d)):
                print(d)
                data1 = load_predictions(os.path.join(where, d))
            elif d == "predictions.csv":
                print(d)
                data1 = load_predictions(os.path.join(where, d))
            if data1 is not None:
                if data is None:
                    data = data1
                else:
                    print("merging")
                    data = data.merge(data1, "outer")
    return data

def main():
    parser = argparse.ArgumentParser(description = "Plotting results from evaluations")
    parser.add_argument("results", type=str, help = "Path to  evaluations")
    parser.add_argument("--out", type=str, help = "Path to directory for saving images")
    args = parser.parse_args()

    data = load_predictions(args.results) 
    def sizeZ(xx):
        if pd.isnull(xx):
            return None
        else:
            return len(literal_eval(xx))


    data['size_z'] = [sizeZ(z) for z in   data['z']]
    print(data)

    # calibrations 
    (ggplot(data) + geom_point(aes("no_conf", "avg_rep_no_conf", color = "data"))+theme_bw()).save("plot_no_calibration.pdf")

    (ggplot(data) + geom_point(aes("yes_conf", "avg_rep_yes_conf", color = "data"))+theme_bw()).save("plot_yes_calibration.pdf")


    #  
    (ggplot(data) + geom_violin(aes("answ == pred", "abs(no_conf - yes_conf)", color = "data", fill = "data")) + theme_bw() + scale_x_discrete(labels = ["error", "correct"]) + xlab("") ).save("plot_violin_1_pred.pdf")

    (ggplot(data) + geom_violin(aes("answ == wpred", "abs(sum_rep_no_conf - sum_rep_yes_conf)", color = "data", fill = "data")) + theme_bw() + scale_x_discrete(labels = ["error", "correct"]) + xlab("") ).save("plot_violin_1_wpred.pdf")


    (ggplot(data) + geom_density(aes("abs(no_conf - yes_conf)", color = "answ == pred")) + theme_bw()).save("plot_density_1.pdf")

    (ggplot(data) + geom_bin_2d(aes("answ", "pred", color = "data")) + geom_text(aes("answ", "pred", label="sum(answ==pred)", group = "factor(data) * factor(answ == pred)")) + facet_grid(['data', ""])).save("plot_perfomrance.pdf") 

if __name__ == "__main__":
    main()
