import os
import yaml
import openai

from dotenv import load_dotenv

from gptci import *

# load enviromental variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# load data 

with open('data/smoking.yaml') as file:
    data = yaml.safe_load(file)

for v in data['variables']:
    print("{name}: {description}".format(**v))

print("---------------")

print("asking if SMK and LC are independent") 
out = gpt_ci("SMK", "LC", None, data)

print(out)


print("asking if SMK and PN are independent given LC") 
out2 = gpt_ci("SMK", "PN", ("LC",), data)

print(out2)
