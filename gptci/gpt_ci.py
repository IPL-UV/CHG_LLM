import openai 

PRS0 = "You are a helpful expert willing to answer questions."
PRS1 = "You are a helpful expert in {field} willing to answer questions."

INST0 = ("When asked to provide estimates and "
        "confidence using prior knowledge "
         "you will answer with your best guess based "
         "on the knowledge you have access to.")

INST1 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables)."
        "Answer only in the required format.")


INST2 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "First argue and dicuss the reasoning behind the answer, "
        "and finally provide the requested answer in the required format")

INDEP = "{x} is independent of {y}"
CINDEP = "{x} is independent of {y} given {z}"

RSPTMPL0 = ("Provide the answer as YES/NO "
           "with percentage uncertainty between brackets."
           " For example \n NO [90%]")

RSPTMPL1 = ("After expalining your reasoning, please provide the answer as YES/NO "
           "with percentage uncertainty between brackets."
           " For example \n NO [90%]")



def parse_response(response, x, y, z, data):
    ## TODO 
    out = response['choices'][0]['message']['content']
    return out

def get_persona(data=None):
    if data is None:
        return PRS0
    else:
        return PRS1.format(**data)

def get_context(data=None):
    data['context']

def get_var_descritpions(x,y,z, data):
    vs = (x,y)
    if z is not None:
        vs = vs + z
    out = """Given {context}.\n
             Consider the following variables:\n
             """.format(**data)
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
z : touple of str, optional
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
           instruction = INST2, response_template = RSPTMPL1):

    persona = get_persona(data)
    context = get_context(data)
    vdescription = get_var_descritpions(x,y,z,data)
    ci = get_ci(x,y,z)
    # TODO pass temperature parameter
    try:
        response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": vdescription},
                    {"role": "user", "content": ci},
                    {"role": "system", "content": response_template}
                    ])
    except Exception as inst:
        return None

    result = parse_response(response,x,y,z,data)
    return result 

