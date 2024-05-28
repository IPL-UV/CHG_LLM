import os
import sys
import yaml
import asyncio 
import logging 
from datetime import datetime
from itertools import chain, combinations

from dotenv import load_dotenv
from pgmpy.base import DAG
import pgmpy

sys.path.append('.')
from gptci import *
import random
import numpy as np
import pandas as pd
import git

from sklearn.metrics import accuracy_score


from openai import AsyncOpenAI, AsyncAzureOpenAI


# https://docs.python.org/3/library/itertools.html
def powerset(iterable, m = None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if m is None:
        m = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(m+1))


def get_vars(edges):
    return  list(set([e['from'] for e in edges] + [e['to'] for e in edges]))


def get_dag(edges):
    dag = DAG([(e['from'], e['to']) for e in edges])
    return dag


def get_all_cis(data, max_cond_set = None):
    variables = [v['name'] for v in data['variables']]
    if data.get('graph') is not None:
        edges = data['graph']
        dag = get_dag(edges)
        all_indep = dag.get_independencies()
        print(all_indep)
        print("got indeps")
    cis = []
    for i in range(1,len(variables)):
        print(i)
        for j  in range(i):
            tmp = set(variables.copy())
            tmp.remove(variables[i])
            tmp.remove(variables[j])
            rest = list(tmp)
            pws = list(powerset(rest, max_cond_set))
            for condset in pws:
                ci = { "x": variables[i],
                    "y": variables[j],
                    "z": list(condset)}
                answ = "UNK"

                if data.get('graph') is not None:
                    if not dag.is_dconnected(variables[i], variables[j], condset):
                    #if all_indep.contains(get_assertion(**ci)):
                        answ = "YES"
                    else:
                        answ = "NO"
                ci['answ'] = answ
                cis = cis + [ci] + [{
                    "x": variables[j],
                    "y": variables[i],
                    "z": list(condset),
                    "answ": answ}]

    return cis



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

    if data.get('graph') is not None:
        edges = data['graph']
        dag = get_dag(edges)
        all_indep = dag.get_independencies()

    cis = k * [None]
    for i in range(k):
        variables = [v['name'] for v in data['variables']]
        ci = sample_ci(variables, min_cond_set, max_cond_set)

        if data.get('graph') is not None:
            if all_indep.contains(get_assertion(**ci)):
                ci['answ'] = "YES"
            else:
                ci['answ'] = "NO" 
        else:
            ci['answ'] = "UNK"
        cis[i] = ci
    return cis

async def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluating conditional independence test")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("--listed", action="store_true", help="evaluate listed cis")
    parser.add_argument("--random", type=int, default=0, help="number of random cis from true graph [%(default)s]")
    parser.add_argument("--valid", type=int, default=0, help="number of valid cis from true graph [%(default)s]")
    parser.add_argument("--all", action = "store_true", default = False, help="if all statements should be tested [CAREFUL CAN BE A LOT of $$]")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=10, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for the model [%(default)s]")
    parser.add_argument("--maxcond", type=int, default=None, help="maximum conditioning set [%(default)s]")
    parser.add_argument("--out", type=str, default=None, help="if not None, the directory name where to save results [%(default)s]")
    parser.add_argument("--dryrun", action="store_true", default = False, help="this option will not actually call the api")
    parser.add_argument("--verbose", action="store_true", default = False, help="verbose?")
    parser.add_argument("--azure", action="store_true", default = False, help="use azure?")

    args = parser.parse_args()

    if args.azure:
        client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
    else:    
        client = AsyncOpenAI(
                    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                    )

    #client.util.logger.setLevel(logging.WARNING)
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
        if data.get("graph") is None:
            print("no graph provided, it is not possible to validate cis")

        print(f"generate {args.random} random ci statements")
        sampled_cis = sample_cis(data, int(args.random), max_cond_set = args.maxcond)
        for ci in sampled_cis:
            ci.update({"type":"random"})
        cis = cis + sampled_cis

    if args.all:
        if data.get("graph") is None:
            print("no graph provided, it is not possible to check which CIs are valid, running anyway...")
        all_cis = get_all_cis(data, max_cond_set = args.maxcond)
        cis = cis + all_cis



    ## TODO take away duplicates???
    #print(cis)

    if args.dryrun:
        tdelay = 0
    else:
        tdelay = 10

    ### tmstamp and git hash 
    tmstp = str(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    print("now call api")
    ## get results 
    results = await gpt_cis(client, cis, data,
            model=args.model,
            n=args.n,
            temperature=args.temperature, tdelay = tdelay,
            dryrun = args.dryrun, verbose = args.verbose)

    ## append results to cis 
    for i in range(len(cis)):

        cis[i].update(results[i][0])
        cis[i].update({"sha" : sha, "tmstmp" : tmstp, 
                       "model" : args.model, "temperature" : args.temperature}) 

    ######### prepare final results
    cisdf = pd.DataFrame(cis)

    ## store name of data
    cisdf["data"] = os.path.basename(data_file).split(".")[0] 

    #acc = accuracy_score(cisdf['answ'], cisdf['pred'])
    #print(f'accuracy : {acc} \n')

    print(cisdf.to_markdown())


    if args.out is not None:
        bn = os.path.basename(data_file).split(".")[0]
        dr = os.path.join(args.out, bn, tmstp)
        os.makedirs(dr, exist_ok=True)
        fn = os.path.join(dr, "{0}.csv") 
        cisdf.to_csv(fn.format("predictions")) 
        with open(os.path.join(dr,'raw.yml'), 'w') as outfile:
            yaml.dump(results, outfile)

if __name__ == "__main__":
    asyncio.run(main())
