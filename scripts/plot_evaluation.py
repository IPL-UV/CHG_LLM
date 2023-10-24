import argparse 
import os

import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, theme_bw, geom_line, save_as_pdf_pages


def load_predictions(where):
    data = None
    if os.path.isfile(where):
        data = pd.read_csv(where, na_values="--")
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
    print(data)


    plots = []
    ### 
    plots += [ggplot(data) + geom_point(aes("no_conf", "avg_rep_no_conf / 100", color = "data"))+
              theme_bw()]


    plots += [ggplot(data) + geom_point(aes("yes_conf", "avg_rep_yes_conf / 100", color = "data"))+
              theme_bw()]

    save_as_pdf_pages(plots, "plots.pdf")


if __name__ == "__main__":
    main()
