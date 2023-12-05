# Operation Simulation 

used repos: ma-gym

python version:  3.9.2

## Setup:


      
      git clone git@github.com:yvoievid/intro-to-data-science.git
      cd ./intro-to-data-science
      python -m venv .
      pip install -r requirements.txt

## Run

      python main.py


## Examples of commands:

After running locally you should be able to reach the api with following link

### Simulation:
  
http://127.0.0.1:105/command/simulate
```
{
    "command":"ATTACK",
    "group": "alfa",
    "flang": "RIGHT",
    "weather": "summer",
    "strategy": "SAFE"
}
```
### Just Run

http://127.0.0.1:105/command/dryrun
```
{
    "command":"ATTACK",
    "group": "alfa",
    "flang": "RIGHT",
    "weather": "summer",
    "strategy": "SAFE"
}
```


## Keyboard Commands While simulating

| Key  | Action  | 
|---|---|
| ESCAPE  |  Quit the game  | 
| SPACE  |   Reset simulation to the initial position | 
