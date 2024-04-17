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

def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()

    openai.util.logger.setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Running PC algorithm")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=10, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for the model [%(default)s]")
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

    gptcit = GPTIndependenceTest(data, args.model, args.n, args.temperature, method = args.method, null = args.null, dryrun = args.dryrun, verbose=False, pre_stored_file=pre_stored_file)
    pc = PC()
    graph = pc.estimate(gptcit, verbose=True, allow_bidirected = False)
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
            f.write(f"arcs: {graph.arcs()}\nedges: {graph.edges()}")
        #dag_vis.savefig(args.out + '/graph.png')


if __name__ == "__main__":
    main()
