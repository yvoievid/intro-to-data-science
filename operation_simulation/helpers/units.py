from ..models import Soldier, Tank
import numpy as np

is_tank = np.vectorize(lambda x: isinstance(x, Tank))
is_soldier = np.vectorize(lambda x: isinstance(x, Soldier))

def count_group_units(unit_group):
    '''
        Counts number of tanks and soliders in the unit group

        Input:
            unit_group - object of class UnitGroup
        
        Output:
            dictionary with keys as unit types and values as their count

    '''
    tanks = np.sum(is_tank(unit_group))
    soliders = np.sum(is_soldier(unit_group))
    return {'tanks': tanks, 'soldiers': soliders}

def set_units(items):
    '''
        Creates an array of units with account on the input dictionary

        Input:
            items - dictionary with keys as unit types and values as their count

        Output:
            units - numpy array of units
    '''
    unit_types = {'soliders': Soldier, 'tanks': Tank}
    units = np.array([])

    for unit_type, count in items.items():
        unit_class = unit_types.get(unit_type.lower())
        if unit_class is not None:
            new_units = np.array([unit_class() for _ in range(count)])
            units = np.concatenate((units, new_units))

    np.random.shuffle(units)

    return units

def has_active_units(units):
    '''
        Checks whether there are any active units in the array

        Input:
            units - numpy array of elements of class Unit
        
        Output:
            bool - True if there are active units, False otherwise
    '''
    alive = np.array([unit.is_alive for unit in units])
    return False if not np.any(units) else len(units[alive]) > 0

def filter_active_units(units):
    '''
        Filters out inactive units from the array

        Input:
            units - numpy array of elements of class Unit
        
        Output:
            units - numpy array of units with health > 0
    '''
    alive = np.array([unit.is_alive for unit in units], dtype='bool')
    return units[alive]

def attack_units(attackers, defenders):
    '''
        Attack units and update state of defenders objects

        Input:
            attackers - numpy array of attacking units
            defenders - numpy array of defending units
    '''
    for attacker in attackers:
        if np.any(defenders):
            attacker.attack_enemy(defenders)
            defenders = filter_active_units(defenders)
        else:
            break