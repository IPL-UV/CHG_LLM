import os
import sys
import yaml
from openai import OpenAI, AzureOpenAI

from dotenv import load_dotenv
from pgmpy.base import DAG
import random

sys.path.append('.')
from gptci import *

# load enviromental variables from .env
load_dotenv()

def sample_CI_set(all_variables):
    # Sample the first variable
    var1 = random.choice(all_variables)
    all_variables.remove(var1)

    # Sample the second variable
    var2 = random.choice(all_variables)
    all_variables.remove(var2)

    # Sample a random number of elements from the remaining ones
    num_elements = random.randint(0, len(all_variables))
    conditioning_set = random.sample(all_variables, num_elements)

    return var1, var2, conditioning_set


def graph_CI_test(var1, var2, conditioning_set, dag):
    all_CI = dag.get_independencies().__dict__['independencies']

    for CI in all_CI:
        if var1 in list(CI.__dict__['event1']):
            if var2 in list(CI.__dict__['event2']):
                if sorted(conditioning_set) == sorted(list(CI.__dict__['event3'])):
                    return True
        if var2 in list(CI.__dict__['event1']):
            if var1 in list(CI.__dict__['event2']):
                if sorted(conditioning_set) == sorted(list(CI.__dict__['event3'])):
                    return True
    return False


def sample_CI_statements(data):
    all_variables = list(set([data['graph'][i]['from'] for i in range(len(data['graph']))] + [data['graph'][i]['to'] for i in range(len(data['graph']))]))
    dag = DAG([(data['graph'][i]['from'], data['graph'][i]['to']) for i in range(len(data['graph']))])

    var1, var2, conditioning_set = sample_CI_set(all_variables)
    statement_valid = graph_CI_test(var1, var2, conditioning_set, dag)

    return var1, var2, conditioning_set, statement_valid


def ci_test(var1, var2, additional_vars, yaml_file):
    # Load data from the YAML file
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))

    print("---------------")

    print(f"asking if {var1} and {var2} are independent given {additional_vars}")
    client = OpenAI(
             api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
            )
    out = gpt_ci(client, var1, var2, additional_vars, data, verbose = True)

    return out

def random_ci_test(yaml_file):
    # Load data from the YAML file
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    var1, var2, additional_vars, statement_valid = sample_CI_statements(data)

    for v in data['variables']:
        print("{name}: {description}".format(**v))

    print("---------------")
    print(f"Ground Truth: {statement_valid}")
    print("---------------")

    print(f"asking if {var1} and {var2} are independent given {additional_vars}")
    client = OpenAI(
             api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
            )
    out = gpt_ci(client, var1, var2, additional_vars, data, verbose = True)


    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Perform conditional independence test")
    parser.add_argument("data", type=str, help="Path to the YAML data file")
    parser.add_argument("var1", type=str, help="Name of the first variable")
    parser.add_argument("var2", type=str, help="Name of the second variable")
    parser.add_argument("additional_vars", nargs="*", type=str, help="Additional variables as a list")
    parser.add_argument("--random", type=bool, help="If test should be sampled randomly.")

    args = parser.parse_args()
    if args.random:
        result = random_ci_test(args.data)
    else:
        result = ci_test(args.var1, args.var2, args.additional_vars, args.data)
    print(result)

if __name__ == "__main__":
    main()
