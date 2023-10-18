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

from sklearn.metrics import accuracy_score

def get_vars(edges):
    return  list(set([e['from'] for e in edges] + [e['to'] for e in edges]))

def get_dag(edges):
    all_variables = list(set([e['from'] for e in edges] + [e['to'] for e in edges]))
    dag = DAG([(e['from'], e['to']) for e in edges])
    return dag
 
def get_cis(edges):
    dag = get_dag(edges)
    return dag.get_independencies() 

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


def sample_cis(data, k = 1, min_cond_set = 0, max_cond_set = None):
    edges = data['graph']
    dag = get_dag(edges)
    cis = k * [None]
    for i in range(k):
        variables = get_vars(edges)
        ci = sample_ci(variables, min_cond_set, max_cond_set)
        if dag.get_independencies().contains(get_assertion(**ci)):
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
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use [%(default)s]")
    parser.add_argument("--n", type=int, default=10, help="number of answer requested from model [%(default)s]")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for the model [%(default)s]")

    args = parser.parse_args()
    data_file = args.data 
    with open(data_file) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))
    
    if args.listed:
        print("---------------------------")
        print("listed ci statements in the data file")
        cis = data['ci-statements'] 
    elif args.random > 0:
        if data['graph'] is None:
            return None
        else:
            print(f"generate {args.random} random ci statements")
            cis = sample_cis(data, int(args.random))

    results = await gpt_cis(cis, data,
                                     model=args.model,
                                     n=args.n,
                                     temperature=args.temperature)
    y_pred = [res[0][0] for res in results]
    y_true = [ci['answ'] for ci in cis]
    print("    ci-statement      |  true  |  pred  |")
    for i in range(len(cis)):
        x = cis[i]['x']
        y = cis[i]['y']
        z = cis[i]['z']
        print(f"{x} ind {y} given {z} |{y_true[i].center(8)}|{y_pred[i].center(8)}|") 
    acc = accuracy_score(y_true, y_pred)
    print(f'accuracy : {acc}')



if __name__ == "__main__":
    asyncio.run(main())
