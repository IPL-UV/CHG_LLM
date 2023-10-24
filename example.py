import asyncio
import os
import yaml
import openai

from dotenv import load_dotenv

from gptci import *

# load enviromental variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def main():
 
    # load data 
    
    with open('data/smoking.yaml') as file:
        data = yaml.safe_load(file)
    
    for v in data['variables']:
        print("{name}: {description}".format(**v))
    
    print("--------------- \n \n")
    
    print("asking if SMK and LC are independent \n\n") 
    out = gpt_ci_sync("SMK", "LC", None, data, n = 10, tryagain = True, verbose = True)
    print("\n")
    print(f"voted answer: {out[0]} \n")
    print(f"parsed answers:")
    print(out[1])
    print("\n") 
    print("Model output:")
    for o in out[2]:
        print(o + "\n\n")
   
    print("--------------- \n \n")
    print("asking if SMK and PN are independent given LC \n \n") 
    out2 = await gpt_ci("SMK", "PN", ["LC"], data, n = 10, verbose = True)
    
    print("\n")
    print(f"voted answer: {out2[0]} \n")
    print(f"parsed answers:")
    print(out2[1])
    print("\n") 
    print("Model output:")
    for o in out2[2]:
        print(o + "\n\n")


    print("trying concurrent task async")
    resa = gpt_cis(data['ci-statements'], data, n = 2)
    print(resa)



asyncio.run(main())
