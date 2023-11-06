import os
import sys
import yaml
import asyncio 
import logging 

from dotenv import load_dotenv
from pgmpy.base import DAG
import pgmpy

sys.path.append('.')
from gptci import *
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

def get_vars(edges):
    return  list(set([e['from'] for e in edges] + [e['to'] for e in edges]))

def get_dag(edges):
    all_variables = list(set([e['from'] for e in edges] + [e['to'] for e in edges]))
    dag = DAG([(e['from'], e['to']) for e in edges])
    return dag
 
def get_cis(edges, translate=False):
    dag = get_dag(edges)
    cis = dag.get_independencies()
    if translate:
        cis2 = cis.get_assertions()
        cis = []
        for ci in cis2:
            xs = list(ci.event1)
            ys = list(ci.event2)
            for x, y in zip(xs, ys):
                cis = cis + [{"x": x,
                    "y": y,
                    "z": list(ci.event3),
                    "answ": "YES",
                    "type": "valid"}]
    return cis 

def get_assertion(x, y, z, *args):
    return pgmpy.independencies.IndependenceAssertion(x, y, z)

def sample_ci(variables, min_cond_set = 0, max_cond_set = None):
    # Sample the first variable
    var1 = random.choice(variables)
    variables.remove(var1)

    # Sample the second variable
    var2 = random.choice(variables)
    variables.remove(var2)

    if max_cond_set is None:
        max_cond_set = len(variables)
    # Sample a random number of elements from the remaining ones
    num_elements = random.randint(min_cond_set, max_cond_set)
    conditioning_set = random.sample(variables, num_elements)

    return {"x": var1, "y": var2, "z" : conditioning_set}


def sample_valid_cis(variables):
    return None
    ## TODO

def sample_cis(data, k = 1, min_cond_set = 0, max_cond_set = None):
    edges = data['graph']
    dag = get_dag(edges)
    cis = k * [None]
    all_indep = dag.get_independencies()
    for i in range(k):
        variables = get_vars(edges)
        ci = sample_ci(variables, min_cond_set, max_cond_set)
        if all_indep.contains(get_assertion(**ci)):
            ci['answ'] = "YES"
        else:
            ci['answ'] = "NO" 
        cis[i] = ci
    return cis
    
async def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    openai.util.logger.setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Evaluating conditional independence test")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("--listed", action="store_true", help="evaluate listed cis")
    parser.add_argument("--random", type=int, default=0, help="number of random cis from true graph [%(default)s]")
    parser.add_argument("--valid", type=int, default=0, help="number of valid cis from true graph [%(default)s]")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=10, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for the model [%(default)s]")
    parser.add_argument("--maxcond", type=int, default=None, help="maximum conditioning set [%(default)s]")
    parser.add_argument("--out", type=str, default=None, help="if not None, the directory name where to save results [%(default)s]")
    parser.add_argument("--dryrun", action="store_true", default = False, help="this option will not actually call the api")
    parser.add_argument("--verbose", action="store_true", default = False, help="verbose?")

    args = parser.parse_args()
    data_file = args.data 
    with open(data_file) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))
    
    cis = []

    if args.listed:
        print("---------------------------")
        print("listed ci statements in the data file")
        listed_cis = data['ci-statements'] 
        for ci in listed_cis:
            ci.update({"type":"listed"})
        cis = cis + listed_cis

    if args.valid > 0:
        if data['graph'] is None:
            print('no graph provided')
        else:
            valid_cis = get_cis(data['graph'], translate = True) 
            cis = cis + random.sample(valid_cis, k = min(args.valid, len(valid_cis)))
        

    if args.random > 0:
        if data['graph'] is None:
            print("no graph provided, it is not possible to sample cis")
        else:
            print(f"generate {args.random} random ci statements")
            sampled_cis = sample_cis(data, int(args.random), max_cond_set = args.maxcond)
            for ci in sampled_cis:
                ci.update({"type":"random"})
            cis = cis + sampled_cis

    print(cis)

    if args.dryrun:
        tdelay = 0
    else:
        tdelay = 10
    results = await gpt_cis(cis, data,
            model=args.model,
            n=args.n,
            temperature=args.temperature, tdelay = tdelay, dryrun = args.dryrun, verbose = args.verbose)
    
    ## append results to cis 
    for i in range(len(cis)):
        cis[i].update(results[i][0])

    ######### prepare final results
    cisdf = pd.DataFrame(cis)

    ## store name of data
    cisdf["data"] = os.path.basename(data_file).split(".")[0] 

    acc = accuracy_score(cisdf['answ'], cisdf['pred'])
    print(f'accuracy : {acc} \n')

    print(cisdf.to_markdown())


    if args.out is not None:
        bn = os.path.basename(data_file).split(".")[0]
        dr = os.path.join(args.out, bn)
        os.makedirs(dr, exist_ok=True)
        fn = os.path.join(dr, "{0}.csv") 
        cisdf.to_csv(fn.format("predictions"))

if __name__ == "__main__":
    asyncio.run(main())
