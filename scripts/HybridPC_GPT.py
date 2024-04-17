import os
import sys
import yaml
import asyncio 
import logging 

from dotenv import load_dotenv
from pgmpy.base import DAG
import pgmpy
import pybnesian as bn
from pathlib import Path

sys.path.append('.')
sys.path.append('../.')
from gptci import *
import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

import git
from datetime import datetime



def render_output(graph, variables, experiment_path, alpha=0.05):
    # Create a directed graph
    G = nx.DiGraph()

    for node in variables:
        G.add_node(node)

    # Add nodes and edges to the graph
    G.add_edges_from(graph.arcs())
    G.add_edges_from(graph.edges())
    G.add_edges_from([(b, a) for (a, b) in graph.edges()])

    # Draw the graph
    pos = nx.circular_layout(G)  # You can choose different layout algorithms
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, arrowsize=15)

    # Save the plot as an image in the experiment folder
    plot_file_path = experiment_path.joinpath(f'graph_plot_{alpha}.png')
    plt.savefig(plot_file_path)

    # Close the plot to avoid displaying it
    plt.close()

    print(f"Plot saved to: {plot_file_path}")


async def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()

    openai.util.logger.setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Running PC algorithm")
    parser.add_argument("experiment_folder", type=str, help="Path to the folder with all files")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=1, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature for the model [%(default)s]")
    parser.add_argument("--out", type=str, default=None, help="if not None, the directory name where to save results [%(default)s]")
    parser.add_argument("--method", type=str, default='vot', help='method used to make decision either vot, wvot or stat')
    parser.add_argument("--null", type=str, default='YES', help='null hypth either YES or NO')
    parser.add_argument("--dryrun", action="store_true", default = False, help="this option will not actually call the api")
    parser.add_argument("--monthly", action="store_true", default = False, help="this option will base the data-driven results on monthly data")
    parser.add_argument("--alpha", type=float, default = 0.05, help="Significance in the response to decide against the null hypothesis")
    parser.add_argument("--alpha_PC", type=float, default = 0.05, help="Significance level for PC algorithm.")
    parser.add_argument("--max_level", type=int, default = 100, help="Maximum level of CIT to consider from predictions. Return data-driven response for higher.")


    verbose_PC = True
    args = parser.parse_args()

    experiment_path = Path(args.experiment_folder)
    base_path = experiment_path.parent

    with open(base_path.joinpath('horn_africa_food_info.yaml')) as file:
        data_info = yaml.safe_load(file)

    for v in data_info['variables']:
        print("{name}: {description}".format(**v))
    variables = [v['name'] for v in data_info['variables']]

    # enso_parent = list(((v, 'enso') for v in variables if v != 'enso'))
    # spi_parent = list(((v, 'spi') for v in variables if v not in ['enso', 'spi']))
    # block_list = enso_parent + spi_parent
    # gpt_variables = ['sorghum_production']
    # PC_restriction = {'block_list': block_list, 'gpt_variables': gpt_variables}

    # with open(experiment_path.joinpath('horn_africa_food_restriction.yaml'), 'w') as yaml_file:
    #     yaml.dump(PC_restrictions, yaml_file, default_flow_style=False)

    # read relevant files:
    load_dotenv()
    openai.util.logger.setLevel(logging.WARNING)

    with open(experiment_path.joinpath('horn_africa_food_restrictions.yaml')) as file:
        PC_restriction = yaml.load(file, Loader=yaml.FullLoader)

    block_list = PC_restriction['block_list']
    gpt_variables = PC_restriction['gpt_variables']

    with open(experiment_path.joinpath('horn_africa_food_listed.yaml')) as file:
        listed = yaml.load(file, Loader=yaml.FullLoader)

    if args.monthly:
        data = pd.read_csv(base_path.joinpath("horn_africa_food_data_m.csv")).astype(float)
    else:
        data = pd.read_csv(base_path.joinpath("horn_africa_food_data.csv")).astype(float)


    for l in range(1):
        pre_stored_file = pd.read_csv(base_path.joinpath('predictions.csv'), index_col=0)
        pre_stored_file['z'].fillna('[]', inplace=True)
        pre_stored_file['z'] = pre_stored_file['z'].apply(lambda x: ast.literal_eval(x))
        #pre_stored_file['z'] = pre_stored_file['z'].apply(lambda x: str([x]) if (isinstance(x, str) and x!="[]") else x)
        #pre_stored_file['z'] = pre_stored_file['z'].apply(lambda x: ast.literal_eval(x))


        with open(experiment_path.joinpath('horn_africa_food_listed.yaml')) as file:
            listed = yaml.load(file, Loader=yaml.FullLoader)

        # while listed is not empty
        pc = bn.PC()
        data_driven_test = bn.MutualInformation(data)  #(data)   #MutualInformation(data)

        gptcit = HybridGPTIndependenceTest(data_info, pre_stored_file = pre_stored_file, gpt_variables = gpt_variables, 
                                        data_driven_test=data_driven_test, method=args.method, null=args.null, dryrun=args.dryrun,
                                        alpha=args.alpha, max_level=args.max_level)
        graph = pc.estimate(gptcit, verbose=True, allow_bidirected = False, arc_blacklist=block_list, alpha=args.alpha_PC) #
        render_output(graph, variables, experiment_path, args.alpha)

        # Example list
        print("---------------------------")
        print(f"To be tested: {gptcit.test_list}")
        print(f"Current level: {gptcit.current_level}")

        # Write the list to the YAML file
        with open(experiment_path.joinpath('horn_africa_food_listed.yaml'), 'w') as yaml_file:
            yaml.dump(gptcit.test_list, yaml_file, default_flow_style=False)

        with open(experiment_path.joinpath('horn_africa_food_listed.yaml')) as file:
            listed = yaml.load(file, Loader=yaml.FullLoader)


        print("---------------------------")
        print("listed ci statements in the data file")
        cis = []
        for ci in gptcit.test_list:
            ci.update({"type":"listed"})
            cis.append(ci)
            cis.append({'x':ci['y'], 'y':ci['x'], 'z':ci['z'], 'type':'listed'})


        ### tmstamp and git hash 
        tmstp = str(datetime.now())
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        args.tmstp = tmstp
        args.repo = repo
        args.sha = sha

        # get results 
        results = await gpt_cis(cis, data_info,
                                model=args.model,
                                n=args.n,
                                temperature=args.temperature, 
                                tdelay = 30,
                                dryrun = args.dryrun, 
                                verbose = False)

        ## append results to cis 
        if not args.dryrun:
            for i in range(len(cis)):
                result = results[i][0] #generate_random_dict(n) # = 

                cis[i].update(result)
                cis[i].update({"sha" : sha, "tmstmp" : tmstp, 
                                "model" : "gpt-3.5-turbo", "temperature" : 0.6}) 

                ######### prepare final results
                if isinstance(cis[i]['z'], str):
                    cis[i]['z'] = [cis[i]['z']]

                cis[i]['z'] = [cis[i]['z']]
                cisdf = pd.DataFrame(cis[i])
                pre_stored_file = pd.concat([pre_stored_file, cisdf], ignore_index=True)
                pre_stored_file.to_csv(base_path.joinpath('predictions.csv'))

        if experiment_path.joinpath('raw.yaml').exists():
            # If the file exists, open it in append mode
            with open(experiment_path.joinpath('raw.yaml'), 'a') as outfile:
                yaml.dump_all(results, outfile, default_flow_style=False)
        else:
            # If the file doesn't exist, create a new file
            with open(experiment_path.joinpath('raw.yaml'), 'w') as outfile:
                # Write the data to the new file
                yaml.dump_all(results, outfile, default_flow_style=False)

        with open(experiment_path.joinpath('args.yaml'), 'w') as outfile:
            yaml.dump(args, outfile)

        # Implies that all necessary results had been stored and we found the final graph.
        if gptcit.current_level == -1:
            return None

if __name__ == "__main__":
    asyncio.run(main())