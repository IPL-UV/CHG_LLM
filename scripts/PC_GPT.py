import os
import sys
import yaml
import asyncio 
import logging 

from dotenv import load_dotenv
from pgmpy.base import DAG
import pgmpy
from pybnesian import PC
from pathlib import Path

sys.path.append('.')
sys.path.append('../.')
from gptci import *
import numpy as np
import pandas as pd
import ast
from time import sleep
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio


import argparse

def main():

    # load enviromental variables from .env
    load_dotenv()

    parser = argparse.ArgumentParser(description="Running PC algorithm")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("--out", type=str, default=None, help="if not None, the directory name where to save results [%(default)s]")
    parser.add_argument("--method", type=str, default='vot', help='method used to make decision either vot, wvot or stat')
    parser.add_argument("--null", type=str, default='YES', help='null hypth either YES or NO')
    parser.add_argument("--dryrun", action="store_true", default = False, help="this option will not actually call the api")
    # add argument that is either a file name or None
    parser.add_argument("--pre_stored_file", type=str, default=None, help="if not None, the file name where to load pre-stored results [%(default)s]")

    verbose_PC = True
    args = parser.parse_args()


    with open(args.data) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))

    # Load answers from previous GPT runs
    if args.pre_stored_file:
        pre_stored_file = pd.read_csv(f'{args.pre_stored_file}')
        pre_stored_file['z'].fillna('[]', inplace=True)
        pre_stored_file['z'] = pre_stored_file['z'].apply(lambda x: ast.literal_eval(x))
    else:
        pre_stored_file = None

    gptcit = GPTIndependenceTest(data, pre_stored_file, method = args.method, null = args.null)
    pc = PC()
    graph = pc.estimate(gptcit, verbose=True, allow_bidirected = True)
    print(graph)
    print(graph.nodes()) 

    print(f"arcs: {graph.arcs()}")
    print(f"edges : {graph.edges()}")

    if args.out is not None:
        # Add GT for comparison
        os.makedirs(args.out, exist_ok=True)
        #arcs = [(data['graph'][i]['from'], data['graph'][i]['to']) for i in range(len(data['graph']))]

        #with open(args.out + '/graph_GT.txt', 'w') as f:
        #    f.write(f"arcs: {arcs}")

        with open(args.out + '/graph.txt', 'w') as f:
            f.write("\\begin{tikzpicture}\n")
            f.write("%% nodes\n")
            for n in graph.nodes():
                f.write(f"\\node ({n}) \u007b{n}\u007d;\n")
            f.write("%% arcs\n")
            for a in graph.arcs():
                f.write(f"\draw[->] ({a[0]}) -- ({a[1]});\n")
            f.write("%% edges\n")
            for a in graph.edges():
                f.write(f"\draw[-] ({a[0]}) -- ({a[1]});\n")
            f.write("\\end{tikzpicture}")
        #dag_vis.savefig(args.out + '/graph.png')


if __name__ == "__main__":
    main()
