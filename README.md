# Team 1: Reinforcement Learning Application of Military Operation Research
Done by team members:
- Yurii Voievidka
- Artur Shevtsov

## Code parts breakdown:

**Implemented by Artur Shevtsov**: 
- A Monte-Carlo simulation algorithm that returns a probability of alliance unit group defeating an enemy group. The logic is located in the following files: 
    - Classes architecture, main attributes and methods are located in: 
   `operation_simulation/models`: `unit.py`, `soldier.py`, `tank.py`
he   
    - Helper functions for working with various data formats: everything in `/operation_simulation/helpers` directory
    - Main logic of the Monte-Carlo simulation algo: `operation_simulation/cross_units_battle/local_battle`
    - Displaying simulation outcomes in PyGame visualization window: `operation_simulation/envs/safe_path.py`

**Implemented by Yurii Voievidka**:

The Reinforcement Learning simulation that shows the whole path of the chosen agent 

In this work I implemented other not that much significant files, like `main.py`, `setup.py`, `helpers/utils.py` that just framework for the Logic is located in following files:
- `simulation.py`- contains the actors setup, the gymnasium environment setup and main pygame process that constantly listen the api and checks the inference from the customer
- `api.py` - Flask API that listens to inference from the LLM input from user
- `operation_simulation/envs/safe_path.py` - Gymnasium Environment class that defines the init, reset, step and render method to retrieve the reward to the agent in simulation.py file
- `operation_simulation/models` - all nessessary models to form unit groups to be used in Gymnasium environment and Local Battle simulation
- `operation_simulation/constants` - all nessessary constants to unify the inference variables in the project
- `operation_simulation/assets` - map for simualtion battleground compreessed using K-Means algorythm
  

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

Query params
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

Query params
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
