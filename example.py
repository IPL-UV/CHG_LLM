import asyncio
import os
import yaml
from openai import AsyncOpenAI

from dotenv import load_dotenv

from gptci import *

# load enviromental variables from .env
load_dotenv()

client = AsyncOpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
async def main():

    # load data 

    with open('data/smoking.yaml') as file:
        data = yaml.safe_load(file)

    for v in data['variables']:
        print("{name}: {description}".format(**v))

    print("--------------- \n \n")

    print("asking if SMK and LC are independent \n\n") 
    out = await gpt_ci(client, "SMK", "LC", None, data, n = 4, tryagain = True, verbose = True)
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
    out2 = await gpt_ci(client, "SMK", "PN", ["LC"], data, n = 4, verbose = True)

    print("\n")
    print(f"voted answer: {out2[0]} \n")
    print(f"parsed answers:")
    print(out2[1])
    print("\n") 
    print("Model output:")
    for o in out2[2]:
        print(o + "\n\n")


    print("trying concurrent task async")
    resa = await gpt_cis(client, data['ci-statements'], data, n = 2)
    print(resa)



asyncio.run(main())
