from operation_simulation.models.unit import Unit
from operation_simulation.models.soldier import Soldier
from operation_simulation.models.tank import Tank
from operation_simulation.helpers.units import set_units, attack_units, count_group_units
import numpy as np

def local_battle(alliance, enemies):
    '''
        Simulates a battle between two armies

        Input:
            alliance, enemies - numpy arrays of battling units
        
        Output:
            outcome - string with the outcome of the battle
            alliance_state - numpy array of alliance units after the battle 
                (empty array if alliance lost, and array with alive units with updated states otherwise)
    '''
    outcome = None
    while alliance.size > 0 and enemies.size > 0:
        first, second = (alliance, enemies) if np.random.rand() > 0.5 else (enemies, alliance)
        attack_units(first, second)
        alliance = alliance[[unit.is_alive for unit in alliance]]
        enemies = enemies[[unit.is_alive for unit in enemies]]
    
    alliance_state = alliance
    outcome = "Alliance won!" if alliance.size > 0 else "Enemy won."
    return outcome, alliance_state


def simulate_local_battle(alliance_powers, enemy_powers, n_simulations):
    '''
        Simulates a battle between two armies `n_simulations` times

        Input:
            alliance_powers, enemy_powers - dictionaries with keys as unit types and values as their count
            n_simulations - number of simulations to run the battle
    '''
    wins = 0
    for i in range(n_simulations):
        alliance, enemy = set_units(alliance_powers), set_units(enemy_powers)
        outcome = local_battle(alliance, enemy)
        if outcome[0] == "Alliance won!":
            wins += 1
            
    prob_winning = wins/n_simulations
    return prob_winning

def group_battles(alliance, enemy_group, group_name):
    alliance, enemies = count_group_units(alliance), count_group_units(enemy_group)
    p = simulate_local_battle(alliance_powers = alliance, enemy_powers=enemies, n_simulations=100)
    return {f"{group_name}": p}

