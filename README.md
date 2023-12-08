# Team 1: Reinforcement Learning Application of Military Operation Research
Done by team members:
- Yurii Voievidka
- Artur Shevtsov

## Code parts breakdown:

**Implemented by Artur Shevtsov**: 
- A Monte-Carlo simulation algorithm that returns a probability of alliance unit group defeating an enemy group. The logic is located in the following files: 
    - Classes architecture, main attributes and methods are located in: `/operation_simulation/models`: `unit.py`, `soldier.py`, `tank.py`
    - Helper functions for working with various data formats: everything in `/operation_simulation/helpers` directory
    - Main logic of the Monte-Carlo simulation algo: `/operation_simulation/models/local_battle`

**Implemented by Yurii Voievidka**:

## Operation Simulation 

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
    "flank": "RIGHT",
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
    "flank": "RIGHT",
    "weather": "summer",
    "strategy": "SAFE"
}
```


## Keyboard Commands While simulating

| Key  | Action  | 
|---|---|
| ESCAPE  |  Quit the game  | 
| SPACE  |   Reset simulation to the initial position | 
