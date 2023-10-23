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
    print("asking if FH indep of LC;  FH indep of SMK and FH indep of PN")
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(gpt_ci("FH", "LC", None, data, n = 20))
        task2 = tg.create_task(gpt_ci("FH", "SMK", None, data, n = 20))
        task3 = tg.create_task(gpt_ci("FH", "PN", None, data, n = 20))

    print(f"3 tasks have completed now:\n FH indep of LC? {task1.result()[0]} \n FH indep of SMK? {task2.result()[0]} \n FH indep of PN? {task3.result()[0]}")


asyncio.run(main())
