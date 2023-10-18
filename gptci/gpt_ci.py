import asyncio
import openai 
import re
import numpy as np

PRS0 = "You are a helpful expert willing to answer questions."
PRS1 = "You are a helpful expert in {field} willing to answer questions."

INST0 = ("When asked to provide estimates and "
        "confidence using prior knowledge "
         "you will answer with your best guess based "
         "on the knowledge you have access to.")

INST1 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "Your answer should not be based on data or observations, "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain provide a valid answer and uncertainty.\n"
        "Answer only in the required format.\n")

INSTCAUSAL = ("You will be asked to provide your estimate and confidence "
        "on the direct causal link  between two variables "
        "(eventually conditioned on a set of variables)."
        "Your answer should not be based on data or observations "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain provide a valid answer and uncertainty."
        "Answer only in the required format. ")

INST2 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "First argue and dicuss the reasoning behind the answer, "
        "and finally provide the requested answer in the required format")

QINDEP = "is {x} independent of {y} ?"
QCINDEP = "is {x} independent of {y} given {z} ?"

INDEP = "{x} is independent of {y}"
CINDEP = "{x} is independent of {y} given {z}"

DEP = "{x} is not independent of {y}"
CDEP = "{x} is not independent of {y} given {z}"



CAUSAL = "are {x} and {y} causally linked ?"
CCAUSAL = "are {x} and {y} causally linked if controlling for {z} ?"

RSPTMPL0 = ("Provide the answer between brackets as YES/NO"
           "with percentage uncertainty between parenthesis.\n"
           "For example [NO (90%)]")

RSPTMPL1 = ("After explaining your reasoning, "
            "provide the answer between brackets as YES/NO, "
           "with percentage uncertainty between parenthesis.\n"
           "Where YES stands for \"{ci}\" and NO stands for \"{noci}\".\n" 
           "For example [NO (50%)] or [YES (50%)].")

# TODO allow spaces between brackets e.g. [ NO (90%) ]
NO = '\[NO \(\d{1,3}\%\)\]'
YES = '\[YES \(\d{1,3}\%\)\]'


def voting(x):
    x = [xx[0] for xx in x]
    nNO = x.count("NO")
    nYES = x.count("YES")
    if nNO > nYES:
        out = "NO"
    if nNO < nYES:
        out = "YES"
    if nNO == nYES:
        out = None
    return out, nNO, nYES, len(x)

def parse_response(response):

    ## locate [NO (xx%)]
    no = re.search(NO, response)
    ## locate [YES (xx%)]
    yes = re.search(YES, response)

    if yes is not None:
        answ = "YES"
        conf = re.search('\(\d{1,3}\%\)', yes[0])[0][1:-2]
    elif no is not None:
        answ = "NO"
        conf = re.search('\(\d{1,3}\%\)', no[0])[0][1:-2]
    else: 
        answ = None
        conf = None
    return answ, conf

def get_persona(data=None):
    if data is None:
        return PRS0
    else:
        if data['field'] is None:
            return PRS0
        else:
            return PRS1.format(**data)

# not used
def get_context(data=None):
    if data is None:
        return ""
    if data['context'] is None:
        return ""
    return data['context']

# this function generate the variables description 
# It also add the context field at the beginning 
def get_var_descritpions(data=None, x=None,y=None,z=None):
    if data is None:
        return "Consider the following variables: {x}, {y} and {z}."
    vs = [v['name'] for v in data['variables']]
    if x is not None and y is not None:
        vs = [x,y]
    if z is not None:
        vs = vs + z
    out = ("{context}\n"
           "Consider the following variables:\n").format(**data)
    for v in data['variables']:
        if v['name'] in vs:
            out = out + "- {name}: {description}\n".format(**v) 
    return out

# this function generate the cis in question format 
def get_qci(x,y,z):
    if z is None:
        return QINDEP.format(x = x,y = y)
    else:
        return QCINDEP.format(x = x, y = y, z = z)

# this function generate the ci statement
def get_ci(x,y,z):
    if z is None:
        return INDEP.format(x = x,y = y)
    else:
        return CINDEP.format(x = x, y = y, z = z)

# this function generate the negated cis  
def get_noci(x,y,z):
    if z is None:
        return DEP.format(x = x,y = y)
    else:
        return CDEP.format(x = x, y = y, z = z)

# this function get the causal statement 
def get_causal(x,y,z):
    if z is None:
        return CAUSAL.format(x = x,y = y)
    else:
        return CCAUSAL.format(x = x, y = y, z = z)


"""CI testing with GPT query

This function query gpt for a (conditional) independence testing statement.
NOW USING ASYNC

Patameters
----------
x : str 
 The name of the first variable
y : str 
 The name of the second variable
z : list of str, optional
 An eventual conditioning set
data : dict 
 The data associate with the problem,
 at minimum it should contain variable descriptions
temperature: float, default = None (it will default to the API defualt 1)
 The temperature parameter for the language model 
model: str, default = "gpt-3.5-turbo"
 The name of the openai model to be called
n: int, default = 1
 The number of answer requested to the model, if n > 1 
 voting is applied to obtain an answer
instruction: str, default = INST1 
 The instruction on the task 
response_template: str
 The response instuction with template 
verbose: bool, defautl = False
 If True the used prompt is printed
"""
async def gpt_ci(x, y, z=None, data=None,
           model="gpt-3.5-turbo", temperature=None, n = 1,
           instruction = INST1, response_template = RSPTMPL1,
           verbose = False):

    persona = get_persona(data)
    vdescription = get_var_descritpions(data,x,y,z)
    qci = get_qci(x,y,z)
    ci = get_ci(x,y,z)
    noci = get_noci(x,y,z)
    if verbose:
        print(f"system: {persona}")
        print(f"system: {instruction}")
        print(f"system: {response_template.format(ci = ci, noci = noci)}")
        print(f"user: {vdescription}\n{qci}")
    try:
        response = await openai.ChatCompletion.acreate(
                model=model,
                temperature=temperature,
                n = n,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "system", "content": instruction},
                    {"role": "system", "content": response_template.format(ci=ci, noci=noci)},
                    {"role": "user", "content": vdescription + "\n" + qci}
                    ])
    except Exception as inst:
        print(inst)
        return None

    results = [res['message']['content'] for res in response['choices']] 
    parsed = [parse_response(res) for res in results] 
    voted = voting(parsed)

    return voted, parsed, results

### similar to ci testing but asking causal question....
### not well done
# TODO either improve it, move it in another file or delete it
# TODO implement the changes in gpt_ci
def gpt_causal(x, y, z=None, data=None, temperature=None, model="gpt-3.5-turbo",
           n = 1, instruction = INSTCAUSAL, response_template = RSPTMPL1, verbose = False):

    persona = get_persona(data)
    context = get_context(data)
    vdescription = get_var_descritpions(data,x,y,z)
    ci = get_causal(x,y,z)
    if verbose:
        print(f"persona: {persona}")
        print(f"instruction: {instruction}")
        print(f"vdescription: {vdescription}")
        print(f"ci: {ci}")
        print(f"template: {response_template}")
    try:
        response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                n = n,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": vdescription},
                    {"role": "user", "content": ci},
                    {"role": "system", "content": response_template}
                    ])
    except Exception as inst:
        return None

    results = [res['message']['content'] for res in response['choices']] 
    parsed = [parse_response(res) for res in results] 
    voted, nNO, nYES = voting(parsed)

    return voted, parsed, results

"""
This function query gpt for a list of CI statements.
Here we use the old completion api

Patameters
----------
cis : list with dicts with 
   x : str 
    The name of the first variable
   y : str 
    The name of the second variable
   z : list of str, optional
    An eventual conditioning set
data : dict 
 The data associate with the problem,
 at a aminimum it should contain variable descriptions
temperature: float, default = 0.6
 The temperature parameter for the language model 
model: str, default = "gpt-3.5-turbo"
 The name of the openai model to be called
instruction: str, default = INST1 
 The instruction on the task 
response_template: str
 The response instuction with template 
"""
def gpt_ci_list(cis, data=None, temperature=None, model="gpt-3.5-turbo-instruct",
           instruction = INST1, response_template = RSPTMPL0, verbose = False):

    persona = get_persona(data)
    context = get_context(data)
    prompts = []
    for ci in cis:
        x = ci['x']
        y = ci['y']
        z = ci['z']
        vdescription = get_var_descritpions(data, x, y, z)
        ci = get_qci(x,y,z)
        prompt = persona + "\n"
        prompt = prompt + instruction + "\n"
        prompt = prompt + vdescription + "\n"
        prompt = prompt + qci + "\n"
        prompt = prompt + response_template 
        if verbose:
            print(prompt)
        prompts = prompts + [prompt]

    try:
        response = openai.Completion.create(
                model=model,
                temperature=temperature,
                prompt = prompts,
                max_tokens = 100)
    except Exception as inst:
        print(inst)
        return None

    results = [""] * len(prompts)
    for choice in response.choices:
        results[choice.index] = choice.text 
    
    return [(parse_response(res), res) for res in results]
