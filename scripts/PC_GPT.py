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

import matplotlib.pyplot as plt



def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    openai.util.logger.setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Evaluating conditional independence test")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=10, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for the model [%(default)s]")
    parser.add_argument("--out", type=str, default=None, help="if not None, the directory name where to save results [%(default)s]")

    verbose_PC = True
    args = parser.parse_args()

    with open(args.data) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))
    
    
    gptcit = GPTIndependenceTest(data, args.model, args.n, args.temperature, verbose=False)
    pc = PC()
    graph = pc.estimate(gptcit, verbose=True)
    print(graph)
    print(graph.nodes()) 
    
    print(f"arcs: {graph.arcs()}")
    print(f"edges : {graph.edges()}")

    if args.out is not None:
        # Add GT for comparison
        os.makedirs(args.out, exist_ok=True)
        arcs = [(data['graph'][i]['from'], data['graph'][i]['to']) for i in range(len(data['graph']))]
        dag_GT = DAG(arcs)
        #print(arcs)
        dag_GT_vis = dag_GT.to_daft()
        dag_GT_vis.savefig(args.out + '/graph_GT.png')
        with open(args.out + '/graph_GT.txt', 'w') as f:
            f.write(str(set(arcs)))

        with open(args.out + '/graph.txt', 'w') as f:
            f.write(str(set(graph.arcs())))
        #dag_vis.savefig(args.out + '/graph.png')


if __name__ == "__main__":
    main()
