import os
import sys
import yaml

from dotenv import load_dotenv
from pgmpy.base import DAG

sys.path.append('.')
from gptci import *
import random

def evaluate_all_cis(data, ):


def main():
    import argparse
    import openai
    # load enviromental variables from .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description="Evaluating conditional independence test")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML data file")

    args = parser.parse_args()
    yaml_file = args.yaml_file 
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))
    
    result = ci_test(args.var1, args.var2, args.additional_vars, args.yaml_file)
    print(result)

if __name__ == "__main__":
    main()
