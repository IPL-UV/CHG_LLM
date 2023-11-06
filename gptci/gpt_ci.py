import asyncio
import openai 
import re
import numpy as np
from random import random
from pybnesian import IndependenceTest
from time import sleep
import pandas as pd

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
        "Even when unsure or uncertain, provide a valid answer and uncertainty.\n"
        "Answer only in the required format.\n")

INSTCAUSAL = ("You will be asked to provide your estimate and confidence "
        "on the direct causal link  between two variables "
        "(eventually conditioned on a set of variables)."
        "Your answer should not be based on data or observations "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain, provide a valid answer and uncertainty."
        "Answer only in the required format. ")

INST2 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "First argue and dicuss the reasoning behind the answer, "
        "and finally provide the requested answer in the required format")

INST3 = ("You will be asked to provide your best guess and your uncertainty "
        "on the statistical independence between two variables "
        "potentially conditioned on a set of variables.\n"
        "Your answer should not be based on data or observations, "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain, provide your best guess (YES or NO) "
        "and the probability that your guess is correct.\n"
        "Answer only in the required format.\n")



QINDEP = "is {x} independent of {y} ?"
QCINDEP = "is {x} independent of {y} given {z} ?"

INDEP = "{x} is independent of {y}"
CINDEP = "{x} is independent of {y} given {z}"

DEP = "{x} is not independent of {y}"
CDEP = "{x} is not independent of {y} given {z}"

DEP3 = "{x} and {y} are dependent"
CDEP3 = "{x} and {y} are dependent given {z}"



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

RSPTMPL2 = ("Work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "The answer must be provided in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "For example [NO (50%)] or [YES (50%)].")

#NO = '\[NO \(\d{1,3}\%\)\]'
NO = '\[\s*NO.*\]'
#YES = '\[YES \(\d{1,3}\%\)\]'
YES = '\[\s*YES.*\]'



def voting(x):
    answs = [xx[0] for xx in x]
    confs = [xx[1] for xx in x]
    nNO = answs.count("NO")
    nYES = answs.count("YES") 
    n = len(x)
    
    ## voting
    if nNO > nYES:
        out = "NO"
    if nNO < nYES:
        out = "YES"
    if nNO == nYES:
        out = "Uncertain"
    
    ## masking None values
    confmask = [c is not None for c in confs] 
    nomask = [a == "NO" and c is not None for a, c in zip(answs, confs)]
    yesmask = [a == "YES" and c is not None for a, c in zip(answs, confs)]
    
 
    ## cleaned values
    confs_c = np.array(confs)[confmask]
    confs_no = np.array(confs)[nomask]
    confs_yes = np.array(confs)[yesmask]


    ## sums of reported confidence
    sumconfno =  np.sum(confs_no)
    sumconfyes = np.sum(confs_yes) 

    ## wighted voting
    if sumconfno > sumconfyes:
        wout = "NO"
    if sumconfno < sumconfyes:
        wout = "YES"
    if sumconfno == sumconfyes:
        wout = "Uncertain"
    
    avgconf = None 
    avgconfno = None 
    avgconfyes = None 
    stdconf = None 
    stdconfno = None 
    stdconfyes = None
    medconf = None 
    medconfno = None 
    medconfyes = None
    q25conf = None
    q75conf = None
    q25confno = None
    q75confno = None
    q25confyes = None
    q75confyes = None
    if len(confs_c) > 0:
        avgconf = np.average(confs_c)
        stdconf =    np.std(confs_c)
        medconf =    np.median(confs_c)
        q25conf = np.quantile(confs_c, 0.25)
        q75conf = np.quantile(confs_c, 0.75)
    if len(confs_no) > 0:
        avgconfno = np.average(confs_no)
        stdconfno =  np.std(confs_no)
        medconfno =  np.median(confs_no)
        q25confno = np.quantile(confs_no, 0.25)
        q75confno = np.quantile(confs_no, 0.75)
    if len(confs_yes) > 0:
        avgconfyes = np.average(confs_yes)
        stdconfyes = np.std(confs_yes)
        medconfyes = np.median(confs_yes)
        q25confyes = np.quantile(confs_yes, 0.25)
        q75confyes = np.quantile(confs_yes, 0.75)


    noconf = nNO / n 
    yesconf = nYES / n 
    return {"pred" : out,
            "wpred": wout,
            "n_no" : nNO, "n_yes" : nYES, "n" : n,
            "no_conf": noconf,
            "yes_conf": yesconf,
            "sum_rep_no_conf":  sumconfno,
            "sum_rep_yes_conf": sumconfyes,
            "avg_rep_conf" :    avgconf,
            "avg_rep_no_conf":  avgconfno, 
            "avg_rep_yes_conf": avgconfyes,
            "std_rep_conf" :    stdconf,
            "std_rep_no_conf":  stdconfno, 
            "std_rep_yes_conf": stdconfyes,
            "med_rep_conf" :    medconf,
            "med_rep_no_conf":  medconfno, 
            "med_rep_yes_conf": medconfyes,
            "q25_rep_conf" :    q25conf,
            "q25_rep_no_conf":  q25confno, 
            "q25_rep_yes_conf": q25confyes,
            "q75_rep_conf" :    q75conf,
            "q75_rep_no_conf":  q75confno, 
            "q75_rep_yes_conf": q75confyes
            } 
            

def parse_response(response):

    ## locate [NO (xx%)]
    no = re.search(NO, response)
    ## locate [YES (xx%)]
    yes = re.search(YES, response)

    conf = None
    if yes is not None:
        answ = "YES"
        conf_s = re.search('\(\d{1,3}\%\)', yes[0])
        if conf_s is not None:
            conf = float(conf_s[0][1:-2]) / 100
    elif no is not None:
        answ = "NO"
        conf_s = re.search('\(\d{1,3}\%\)', no[0])
        if conf_s is not None:
            conf = float(conf_s[0][1:-2]) / 100
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
        if type(x) is not list:
            x = [x]
        if type(y) is not list:
            y = [y]
        vs = x + y
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
 The numbergg of answer requested to the model, if n > 1 
 voting is applied to obtain an answer
instruction: str, default = INST1 
 The instruction on the task 
response_template: str
 The response instuction with template 
verbose: bool, default = False
 If True the used prompt is printed
tryagain: bool, default = False,
 if True, try the api call again after exponential delay 
tdelay: double 
 dealy in seconds before rescheduling the task 
dryrun: bool, default = False
 if True a test-run will be performed, without actual calls to the api 
"""
async def gpt_ci(x, y, z=None, data=None,
           model="gpt-3.5-turbo", temperature=None, n = 1,
           instruction = INST3, response_template = RSPTMPL2,
           verbose = False, tryagain = False, tdelay = 1, dryrun = False):

    # if x,y are list and length 1 reduce them
    if type(x) is list:
        if len(x) == 1:
            x = x[0]
    if type(y) is list:
        if len(y) == 1:
            y = y[0]
    persona = get_persona(data)
    vdescription = get_var_descritpions(data,x,y,z)
    qci = get_qci(x,y,z)
    ci = get_ci(x,y,z)
    noci = get_noci(x,y,z)
    prompt = f"system: {persona} \n system: {instruction} \n" 
    prompt = prompt +  f"system: {response_template.format(ci = ci, noci = noci)}\n"
    prompt = prompt + f"user: {vdescription}\n{qci}" 
    if verbose:
        print(prompt)
    try:
        if dryrun:
            if random() > 0.5:
                response = {"choices": [{"message": {"content":"[NO (0%)]"}}] * n }
                results = [res['message']['content'] for res in response['choices']] 
                parsed = [parse_response(res) for res in results] 
                voted = voting(parsed)
                return voted, parsed, results, prompt
            else: 
                raise Exception("test exception")
        else:
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

            results = [res['message']['content'] for res in response['choices']] 
            parsed = [parse_response(res) for res in results] 
            voted = voting(parsed)
            return voted, parsed, results, prompt

    except Exception as inst:
        print("error from server (likely)")
        print(inst)
        if tryagain:
            print(f"rescheduling task in {tdelay} seconds ...")
            await asyncio.sleep(tdelay)
            res = await gpt_ci(x,y,z,data,model,temperature,n,
                instruction,response_template,verbose,tryagain, tdelay*2, dryrun = dryrun)
            return res
        else:
            return None

def gpt_ci_sync(x, y, z=None, data=None,
           model="gpt-3.5-turbo", temperature=None, n = 1,
           instruction = INST3, response_template = RSPTMPL2,
           verbose = False, tryagain = False, tdelay = 1, dryrun = False):

    # if x,y are list and length 1 reduce them
    if type(x) is list:
        if len(x) == 1:
            x = x[0]
    if type(y) is list:
        if len(y) == 1:
            y = y[0]
    persona = get_persona(data)
    vdescription = get_var_descritpions(data,x,y,z)
    qci = get_qci(x,y,z)
    ci = get_ci(x,y,z)
    noci = get_noci(x,y,z)
    prompt = f"system: {persona} \n system: {instruction} \n" 
    prompt = prompt +  f"system: {response_template.format(ci = ci, noci = noci)}\n"
    prompt = prompt + f"user: {vdescription}\n{qci}" 
    if verbose:
        print(prompt)
    try:
        if dryrun:
            if random() > 0.5:
                response = {"choices": [{"message": {"content":"[NO (0%)]"}}] * n }
                results = [res['message']['content'] for res in response['choices']] 
                parsed = [parse_response(res) for res in results] 
                voted = voting(parsed)
                return voted, parsed, results, prompt
            else: 
                raise Exception("test exception")
        else:
            response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    n = n,
                    messages=[
                        {"role": "system", "content": persona},
                        {"role": "system", "content": instruction},
                        {"role": "system", "content": response_template.format(ci=ci, noci=noci)},
                        {"role": "user", "content": vdescription + "\n" + qci}
                        ])

            results = [res['message']['content'] for res in response['choices']] 
            parsed = [parse_response(res) for res in results] 
            voted = voting(parsed)
            return voted, parsed, results, prompt

    except Exception as inst:
        print("error from server (likely)")
        print(inst)
        if tryagain:
            print(f"rescheduling task in {tdelay} seconds ...")
            sleep(tdelay)
            res = gpt_ci_sync(x,y,z,data,model,temperature,n,
                instruction,response_template,verbose,tryagain, tdelay*2, dryrun = dryrun)
            return res
        else:
            return None


## async reqests for multiple cis
async def gpt_cis(cis, data,
                  model = "gpt-3.5-turbo", n = 1, temperature = None, 
                  instruction = INST3, response_template = RSPTMPL2,
                  tdelay = 60, dryrun = False, verbose = False):

    tasks = set()
    for i in range(len(cis)):
        x = cis[i]['x']
        y = cis[i]['y']
        z = cis[i]['z']
        task = asyncio.create_task(gpt_ci(x, y, z, data = data,
                                          model = model, 
                                          temperature = temperature, 
                                          n = n,
                                          instruction = instruction,
                                          response_template = response_template,
                                          tryagain = True, tdelay = tdelay,
                                          dryrun = dryrun, verbose = verbose),
                                   name = i) 
        tasks.add(task)
        await asyncio.sleep(0.01) ## wait 1/100 seconds between requests at least
    await asyncio.gather(*tasks)

    print(f"total task executed: {len(tasks)}")
    results = [None] * len(cis)
     
    # extract results in correct order
    for task in tasks:
        if task.done() and task.result() is not None:
            i = int(task.get_name())
            results[i] = task.result()

    return results  


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

class GPTIndependenceTest(IndependenceTest):
    def __init__(self, data, model, n, temperature, verbose = False, pre_stored_file=None):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.data = data
        self.model = model
        self.n = n
        self.temperature = temperature
        self.verbose = verbose
        # extract variable names from data dictionary
        self.variables = [var['name'] for var in self.data['variables']]
        self.pre_stored_file = pre_stored_file
            
    def num_variables(self):
        return len(self.variables)

    def variable_names(self):
        return self.variables

    def has_variables(self, vars):
        return set(vars).issubset(set(self.variables))

    def name(self, index):
        return self.variables[index]

    def pvalue(self, x, y, z):
        # Make sure that z is a list
        if isinstance(z, str):
            z = [z]
        if z is None:
            z = []
        
        # check if self.pre_stored_file is not None
        if not self.pre_stored_file is None:            
            # TODO: Should we average over both options X,Y and Y,X?
            rowXY = self.pre_stored_file.loc[(self.pre_stored_file['x'] == x) & (self.pre_stored_file['y'] == y) & (np.array([set(Z) for Z in self.pre_stored_file['z']]) ==set(z))]
            rowYX = self.pre_stored_file.loc[(self.pre_stored_file['y'] == x) & (self.pre_stored_file['x'] == y) & (np.array([set(Z) for Z in self.pre_stored_file['z']]) ==set(z))]
            # union of both dataframes
            row = pd.concat([rowXY, rowYX])

            if len(row) > 1:
                print(f"Warning: more than one row found in pre-stored file for statement {x} indep {y} given {z}. Average output reponse.")
                # TODO: Discuss what to prefer in case of 0.5
                if row['n_no'].sum() >= row['n_yes'].sum():
                    # NO wins voting, not independent, significant evidence against conditional independence
                    return 0
                else:
                    # YES wins voting, independent, or rather no significant evidence against conditional independence
                    return 1

            if len(row) == 1:
                if (row['pred'].values)[0] == 'NO':
                    return 0
                else:
                    return 1
            print(f"Warning: No row found in pre-stored file for statement {x} indep {y} given {z}. Ask Chat GPT.")

        # If there are no pre-stored results, ask Chat GPT.
        results = gpt_ci_sync(x, y, z, self.data,
                                model=self.model,
                                n=self.n,
                                temperature=self.temperature,
                                verbose=self.verbose,
                                instruction = INST3, 
                                response_template = RSPTMPL2,
                                tryagain=True,
                                dryrun= False)
        if results[0]['pred'] == 'NO':
            return 0
        else:
            return 1
