import asyncio

import re
import numpy as np
from random import random
from pybnesian import IndependenceTest
from time import sleep
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from scipy.stats import norm

PRS0 = "You are a helpful expert willing to answer questions."
PRS1 = "You are a helpful expert in {field} willing to answer questions."

INST0 = ("When asked to provide estimates and "
        "confidence using prior knowledge "
        "you will answer with your best guess based "
        "on the knowledge you have access to.")

INSTCAUSAL = ("You will be asked to provide your estimate and confidence "
        "on the direct causal link  between two variables "
        "(eventually conditioned on a set of variables)."
        "Your answer should not be based on data or observations "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain, provide a valid answer and uncertainty."
        "Answer only in the required format. ")

INST1 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "Your answer should not be based on data or observations, "
        "but only on the available knowledge.\n"
        "Even when unsure or uncertain, provide a valid answer and uncertainty.\n"
        "Answer only in the required format.\n")

INST2 = ("You will be asked to provide your estimate and confidence "
        "on statistical independence between two variables "
        "(eventually conditioned on a set of variables).\n"
        "First argue and dicuss the reasoning behind the answer, "
        "and finally provide the requested answer in the required format")

INST3 = ("You will be asked to provide your best guess and your uncertainty "
        "on the statistical independence between two variables "
        "potentially conditioned on a set of variables.\n"
        "Your answer should not be based on data or observations, "
        "but only on knowledge. The knowledge should go beyond the given context.\n"
        "Even when unsure or uncertain, provide your best guess (YES or NO) "
        "and the probability that your guess is correct.\n"
        "Answer only in the required format.\n")

cond = ""#'conditionally '

QINDEP = "is {x} independent of {y} ?"
QCINDEP = "is {x} conditionally independent of {y} conditioned on {z} ?"

INDEP = "{x} is independent of {y}"
CINDEP = "{x} is conditionally independent of {y} conditioned on {z}"

DEP = "{x} is not independent of {y}"
CDEP = "{x} is not conditionally independent of {y} conditioned on {z}"

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


RSPTMPL2 = ( "First, take a step back: Think about the bigger picture and name all colliders,"
            " the children of colliders, common causes or mediators that "
            "need to be taken into account to answer the question. "
            "Then work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "For example [NO (50%)] or [YES (50%)].")

RSPTMPL = ("Work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "Here are three examples:\n"
            "First example:\n"
            "To determine if ice cream sales (X) is conditionally independent of drowning incidents (Y)"
            "given [temperature (Z)], we need to consider if knowing Z makes X and Y carry information about the"
            "occurance of each other. There is a known correlation between ice cream sales and "
            "drowning incidents. However, we have reason to believe that temperature is a confounder "
            "that explains both X and Y. For example, higher temperatures may lead to both "
            "increased ice cream sales and more people going to the beach, "
            "increasing the likelihood of drowning incidents. "
            "This means that, for a given temperature, the occurrence or non-occurrence of drowning "
            "incidents likely doesn't provide any additional information about ice cream sales beyond "
            "what is already known from the temperature."
            "Based on this reasoning, we can conclude:\n\n[YES (85%)]\n\n"
            "Second example:\n"
            "To determine if ice cream sales (X) is independent of drowning incidents (Y), "
            "we need to consider if X and Y carry information about the"
            "occurance of each other. There is a known correlation between ice cream sales and "
            "drowning incidents."
            "ice cream sales tend to increase at the same time as drowning incidents."
            "Based on this reasoning, we can conclude:\n\n[NO (90%)]\n\n"
            "Third example:\n"
            "To determine if hours of sleep (X) is conditionally independent of academic performance (Y) "
            "given [extracurricular activities (Z)], we need to consider if knowing Z makes X and Y carry information about the"
            "occurance of each other. There is no known or only very weak correlation between X and "
            "Y. However, Students who perform well academically tend to engage in more extracurricular activities."
            "Students who get more sleep tend to engage in fewer extracurricular activities."
            "Z is influenced by both X and Y. If we take a student with a lot of extracurricular activities "
            "a good academic performance probably goes hand in hand with few hours of sleep. Conditioning on Z therefore "
            "leads to a spurious correlation between X and Y. Given Z there seems to be an information flow between X and Y."
            "Based on this reasoning, we can conclude:\n\n[NO (80%)]"
            )

RSPTMPL = ("Work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer and that you respect the "
            "principles conditional independence testing in causality."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "For example [NO (50%)] or [YES (50%)].\n"
            "Principles of conditional independence testing:\n"
            "1. Identify and account for all variables that are direct causes" 
            "or common causes of the variables being tested for conditional independence\n"
            "2. Utilize a reliable causal inference method, such as the backdoor criterion" 
            "or frontdoor criterion, to ensure proper adjustment for confounding variables" 
            "and establish conditional independence relationships.\n"
            "3. Be cautious of collider variables, as conditioning on them may"
            "induce spurious associations, and consider the underlying causal structure" 
            "when interpreting the results of conditional independence tests in the context of a causal graph.\n"
            )

RSPTMPL = ("Work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "For example [NO (50%)] or [YES (50%)].")

RSPTMPL4 = ("Work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "Here are three examples:\n"
            "First example:\n"
            "To determine if ice cream sales (X) is conditionally independent of drowning incidents (Y)"
            "given [temperature (Z)], we need to consider if knowing Z makes X and Y carry information about the"
            "occurance of each other. There is a known correlation between ice cream sales and "
            "drowning incidents. However, we have reason to believe that temperature is a confounder "
            "that explains both X and Y. For example, higher temperatures may lead to both "
            "increased ice cream sales and more people going to the beach, "
            "increasing the likelihood of drowning incidents. "
            "This means that, for a given temperature, the occurrence or non-occurrence of drowning "
            "incidents likely doesn't provide any additional information about ice cream sales beyond "
            "what is already known from the temperature."
            "Based on this reasoning, we can conclude:\n\n[YES (85%)]\n\n"
            "Second example:\n"
            "To determine if ice cream sales (X) is independent of drowning incidents (Y), "
            "we need to consider if X and Y carry information about the"
            "occurance of each other. There is a known correlation between ice cream sales and "
            "drowning incidents."
            "ice cream sales tend to increase at the same time as drowning incidents."
            "Based on this reasoning, we can conclude:\n\n[NO (90%)]\n\n"
            "Third example:\n"
            "To determine if hours of sleep (X) is conditionally independent of academic performance (Y) "
            "given [extracurricular activities (Z)], we need to consider if knowing Z makes X and Y carry information about the"
            "occurance of each other. There is no known or only very weak correlation between X and "
            "Y. However, Students who perform well academically tend to engage in more extracurricular activities."
            "Students who get more sleep tend to engage in fewer extracurricular activities."
            "Z is influenced by both X and Y. If we take a student with a lot of extracurricular activities "
            "a good academic performance probably goes hand in hand with few hours of sleep. Conditioning on Z therefore "
            "leads to a spurious correlation between X and Y. Given Z there seems to be an information flow between X and Y."
            "Based on this reasoning, we can conclude:\n\n[NO (80%)]"
            )

RSPTMPL2 = ( "First, take a step back: Think about the bigger picture and name all colliders,"
            " the children of colliders, common causes or mediators that "
            "need to be taken into account to answer the question. "
            "Then work out the answer in a step-by-step way to be as "
            "sure as possible that you have the right answer."
            "After explaining your reasoning, "
            "provide the answer in the following form: "
            "[<ANSWER> (<PROBABILITY>)] where ANSWER is either YES or NO "
            "and PROBABILITY is a percentage between 0% and 100%."
            "YES stands for \"{ci}\" and NO stands for \"{noci}\".\n"
            "For example [NO (50%)] or [YES (50%)].")

#NO = '\[NO \(\d{1,3}\%\)\]'
NO = '\[\s*NO.*\]'
#YES = '\[YES \(\d{1,3}\%\)\]'
YES = '\[\s*YES.*\]'



def voting(x):
    answs = [xx["answ"] for xx in x]
    confs = [xx["conf"] for xx in x]
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

    answy, pvaly = test_prop(nNO, nYES, n, null = "YES", alpha = 0.05)
    answn, pvaln = test_prop(nNO, nYES, n, null = "NO", alpha = 0.05)
    return {"pred" : out,
            "wpred": wout,
            "stat_yes": answy,
            "pval_yes": pvaly,
            "stat_no": answn,
            "pval_no": pvaln,
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
    return {"answ" : answ, "conf" : conf}

def get_persona(data=None):
    if data is None:
        return PRS0
    else:
        if data.get('field') is None:
            return PRS0
        else:
            return PRS1.format(**data)

# not used
def get_context(data=None):
    if data is None:
        return ""
    return data['context']

# this function generate the variables description 
# It also add the context field at the beginning 
def get_var_descriptions(data=None, x=None,y=None,z=None):
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
        if len(z) > 0:
            vs = vs + z

    out = ("{context}\n"
            "Consider the following variables:\n").format(**data)
    for v in data['variables']:
        #out = out + "- {name}: {description}\n".format(**v) 
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
async def gpt_ci(client, x, y, z=None, data=None,
           model="gpt-3.5-turbo", temperature=None, n = 1,
           instruction = INST1, response_template = RSPTMPL1,
           verbose = False, tryagain = False, tdelay = 1, dryrun = False, out = None):

    # if z is emtpy just put None
    if z is not None:
        if len(z) == 0:
            z = None
    # if x,y are list and length 1 reduce them
    if type(x) is list:
        if len(x) == 1:
            x = x[0]
    if type(y) is list:
        if len(y) == 1:
            y = y[0]
    persona = get_persona(data)
    vdescription = get_var_descriptions(data,x,y,z)
    qci = get_qci(x,y,z)
    ci = get_ci(x,y,z)
    noci = get_noci(x,y,z)
    prompt = f"system: {persona} \n system: {instruction} \n" 
    prompt = prompt + f"user: {vdescription}\n{qci}" 
    prompt = prompt +  f"system: {response_template.format(ci = ci, noci = noci)}\n"
    if verbose:
        print(prompt)
    try:
        if dryrun:
            if random() > 0:
                response = {"choices": [{"message": {"content":"[NO (0%)]"}}] * n }
                results = [res['message']['content'] for res in response["choices"]] 
                parsed = [parse_response(res) for res in results] 
                voted = voting(parsed)
                return voted, parsed, results, prompt, f'{x},{y}|{z}'
            else: 
                raise Exception("test exception")
        else:
            response = await client.chat.completions.create(model=model,
            temperature=temperature,
            n = n,
            messages=[
                {"role": "system", "content": persona},
                {"role": "system", "content": instruction},
                {"role": "user", "content": vdescription + "\n" + qci},
                {"role": "system", "content": response_template.format(ci=ci, noci=noci)}
                ])

            results = [res.message.content for res in response.choices] 
            parsed = [parse_response(res) for res in results] 
            voted = voting(parsed)
            return voted, parsed, results, prompt, f'{x},{y}|{z}'

    except Exception as inst:
        print("error from server (likely)")
        print(inst)
        if tryagain:
            td = min(100, tdelay) * random()
            print(f"rescheduling task in {td} seconds ...")
            await asyncio.sleep(td)
            res = await gpt_ci(client,x,y,z,data,model,temperature,n,
                instruction,response_template,verbose,tryagain, tdelay*1.5, dryrun = dryrun)
            return res
        else:
            return None


## async reqests for multiple cis
async def gpt_cis(client, cis, data,
                  model = "gpt-3.5-turbo", n = 1, temperature = None, 
                  instruction = INST1, response_template = RSPTMPL1,
                  tdelay = 60, dryrun = False, verbose = False):

    tasks = set()
    for i in range(len(cis)):
        x = cis[i]['x']
        y = cis[i]['y']
        z = cis[i]['z']
        task = asyncio.create_task(gpt_ci(client, x, y, z, data = data,
                                          model = model, 
                                          temperature = temperature, 
                                          n = n,
                                          instruction = instruction,
                                          response_template = response_template,
                                          tryagain = True, tdelay = tdelay,
                                          dryrun = dryrun, verbose = verbose),
                                   name = i) 
        tasks.add(task)
        await asyncio.sleep(0.1) ## wait 1/10 seconds between requests at least
    await tqdm_asyncio.gather(*tasks)
    #await asyncio.gather(*tasks)

    print(f"total task executed: {len(tasks)}")
    results = [None] * len(cis)

    # extract results in correct order
    for task in tasks:
        if task.done() and task.result() is not None:
            i = int(task.get_name())
            results[i] = task.result()

    return results  


def test_prop(n_no, n_yes, n, null = "YES", alpha = 0.05):
    p_no = n_no / n # prop of no
    p_yes = n_yes / n # prop of yes
    p = (n_no + n_yes) / (2*n) # pooled prop 
    se = np.sqrt( p * (1 - p) * 2 / n) # standard error 
    if null == "YES":  
        answ = "YES"
        pval = norm.sf((p_no - p_yes) / se) # p_no - p_yes >> 0 is extreme for null == "YES" 
        if pval <= alpha:
            answ = "NO"
    if null == "NO":
        answ = "NO"
        pval = norm.sf((p_yes - p_no) / se) # p_yes - p_no >> 0 is extreme for null == "NO" 
        if pval <= alpha:
            answ = "YES"
    return answ, pval

# method can be vot (voting), wvot (weighted voting), stat (hyp test on proportions)
class GPTIndependenceTest(IndependenceTest):
    def __init__(self, data, pre_stored_file, method = "vot", null = "YES"):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.data = data
        self.pre_stored_file = pre_stored_file
        self.method = method
        self.null = null
        # extract variable names from data dictionary
        self.variables = [var['name'] for var in self.data['variables']]

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

            if len(row) >= 1:
                if len(row) > 1:
                    print(f"Warning: more than one row found in pre-stored file for statement {x} indep {y} given {z}. Average output reponse.")
                # TODO: Discuss what to prefer in case of 0.5

                n_no = row['n_no'].sum()
                n_yes = row['n_yes'].sum()
                nn = row['n'].sum()

                if self.method == "stat":
                    answ, pval = test_prop(n_no, n_yes, nn, null = self.null, alpha = 0.01)
                    if answ == "NO":
                        return 0
                    if answ == "YES":
                        return 1

                if self.method == "vot":
                    if n_no > n_yes:
                        # NO wins voting, not independent, significant evidence against conditional independence
                        return 0
                    if n_yes >  n_no:
                        # YES wins voting, independent, or rather no significant evidence against conditional independence
                        return 1
                    if n_yes == n_no:
                        if self.null == "YES":
                            return 1
                        if self.null == "NO":
                            return 0

                if self.method == "wvot":
                    return 0
                    ##TODO

        print(f"Warning: No row found in pre-stored file for statement {x} indep {y} given {z}. return 1")
        return 1

class HybridGPTIndependenceTest(IndependenceTest):
    def __init__(self, data_info, pre_stored_file, gpt_variables = None, data_driven_test=None, method = "vot", null = "YES", test_list=None, dryrun=False, alpha=0.05, max_level=100):
        # IMPORTANT: Always call the parent class to initialize the C++ object.
        IndependenceTest.__init__(self)
        self.data_info = data_info
        self.null = null
        self.data_driven_test = data_driven_test
        self.gpt_variables = gpt_variables
        self.test_list = []
        self.method=method
        self.dryrun=dryrun
        self.alpha=alpha
        self.max_level=max_level

        # extract variable names from data dictionary
        self.variables = [var['name'] for var in self.data_info['variables']]
        self.pre_stored_file = pre_stored_file
        self.current_level = -1 # value for level-wise increase with gpt queries

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

        # check if one of the strings in x+y+z is in gpt_variables
        if len(set(z+[x]+[y]) & set(self.gpt_variables)) > 0 and len(z) <= self.max_level:       

            rowXY = self.pre_stored_file.loc[(self.pre_stored_file['x'] == x) & (self.pre_stored_file['y'] == y) & (np.array([set(Z) for Z in self.pre_stored_file['z']]) ==set(z))]
            rowYX = self.pre_stored_file.loc[(self.pre_stored_file['y'] == x) & (self.pre_stored_file['x'] == y) & (np.array([set(Z) for Z in self.pre_stored_file['z']]) ==set(z))]
            # union of both dataframes
            row = pd.concat([rowXY, rowYX])

            if len(row) >= 1:
                if len(row) > 1:
                    print(f"Warning: more than one row found in pre-stored file for statement {x} indep {y} given {z}. Average output reponse.")

                n_no = row['n_no'].sum()
                n_yes = row['n_yes'].sum()
                nn = row['n'].sum()

                if self.method == "stat":
                    answ, pval = test_prop(n_no, n_yes, nn, null = self.null, alpha = self.alpha)
                    if answ == "NO":
                        return 0
                    if answ == "YES":
                        return 1

                if self.method == "vot":
                    if n_no > n_yes:
                        # NO wins voting, not independent, significant evidence against conditional independence
                        return 0
                    if n_yes >  n_no:
                        # YES wins voting, independent, or rather no significant evidence against conditional independence
                        return 1
                    if n_yes == n_no:
                        if self.null == "YES":
                            return 1
                        if self.null == "NO":
                            return 0

                if self.method == "wvot":
                    return 0
                    ##TODO

            if row.empty:
                if self.dryrun:
                    return 0
                elif self.current_level != -1 and len(z) > self.current_level:
                    return 1
                else:
                    self.test_list.append({'x': x, 'y': y, "z": z})     
                    self.current_level = len(z)
                    return 0
        else:
            return self.data_driven_test.pvalue(x,y,z)

