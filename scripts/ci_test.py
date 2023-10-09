import os
import sys
import yaml
import openai

from dotenv import load_dotenv

sys.path.append('.')
from gptci import *

# load enviromental variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ci_test(var1, var2, additional_vars, yaml_file):
    # Load data from the YAML file
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))

    print("---------------")

    print(f"asking if {var1} and {var2} are independent given {additional_vars}")
    out = gpt_ci(var1, var2, (additional_vars,), data)
    
    return out

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform conditional independence test")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML data file")
    parser.add_argument("var1", type=str, help="Name of the first variable")
    parser.add_argument("var2", type=str, help="Name of the second variable")
    parser.add_argument("additional_vars", nargs="*", type=str, help="Additional variables as a list")

    args = parser.parse_args()
    
    result = ci_test(args.var1, args.var2, args.additional_vars, args.yaml_file)
    print(result)

if __name__ == "__main__":
    main()
