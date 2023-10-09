# Causal discovery with GPT 


## content 

- `data` folder with descriptions of causal discovery problems and eventually datasets
- `gptci` python module that implement CI testing via GPT queries
- `.env.example` example of needed environemntal variables 


## how to 

- `python -m venv env`
- `source env/bin/activate` 
- `pip install -r requirements.txt` 
- `cp .env.example .env` generate the `.env` file 
- add your openai API key into the `.env` file (.env is already ignored in the `.gitignore` file but doubel check you are not pushing it) 
- `python example.py`  to run a simple example 
- For running a specific CI test var1, var2| (var3, var4,...) 
`python scripts\ci_test.py var1 var2 [var3] [var4] ...`



