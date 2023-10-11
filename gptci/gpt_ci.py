import openai 
import re

PRS0 = "You are a helpful expert willing to answer questions."
PRS1 = "You are a helpful expert in {field} willing to answer questions."

INST0 = ("When asked to provide estimates and "
        "confidence using prior knowledge "
         "you will answer with your best guess based "
         "on the knowledge you have access to.")

INST1 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
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

INDEP = "is {x} independent of {y} ?"
CINDEP = "is {x} independent of {y} given {z} ?"

RSPTMPL0 = ("Provide the answer between brackets as YES/NO"
           "with percentage uncertainty between parenthesis."
           " For example \n [NO (90%)]")

RSPTMPL1 = ("After expalining your reasoning, "
            "please provide the answer between brackets as YES/NO,"
           "with percentage uncertainty between parenthesis."
           " For example \n [NO (90%)]")

NO = '\[NO \(\d{1,3}\%\)\]'
YES = '\[YES \(\d{1,3}\%\)\]'


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
    return answ, conf, response

def get_persona(data=None):
    if data is None:
        return PRS0
    else:
        return PRS1.format(**data)
def get_context(data=None):
    if data is None:
        return ""
    return data['context']

def get_var_descritpions(data, x=None,y=None,z=None):
    vs = [v['name'] for v in data['variables']]
    if x is not None and y is not None:
        vs = [x,y]
    if z is not None:
        vs = vs + z
    out = ("Given {context}.\n"
           "Consider the following variables:\n").format(**data)
    for v in data['variables']:
        if v['name'] in vs:
            out = out + "- {name}: {description}\n".format(**v) 
    return out

def get_ci(x,y,z):
    if z is None:
        return INDEP.format(x = x,y = y)
    else:
        return CINDEP.format(x = x, y = y, z = z)


"""CI testing with GPT query

This function query gpt for a (conditional) independence testing statement.

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
def gpt_ci(x, y, z=None, data=None, temperature=None, model="gpt-3.5-turbo",
           instruction = INST1, response_template = RSPTMPL1, verbose = False):

    persona = get_persona(data)
    context = get_context(data)
    vdescription = get_var_descritpions(data,x,y,z)
    ci = get_ci(x,y,z)
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
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": vdescription},
                    {"role": "user", "content": ci},
                    {"role": "system", "content": response_template}
                    ])
    except Exception as inst:
        return None

    out = response['choices'][0]['message']['content']
    result = parse_response(out)
    return result 

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
        ci = get_ci(x,y,z)
        prompt = persona + "\n"
        prompt = prompt + instruction + "\n"
        prompt = prompt + vdescription + "\n"
        prompt = prompt + ci + "\n"
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
        results[choice.index] = choice.text#[len(prompts[choice.index]):] 
    
    return [parse_response(res) for res in results]
