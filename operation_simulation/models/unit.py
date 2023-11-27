from gymnasium import spaces
from dataclasses import dataclass,field
from typing import List

@dataclass
class Unit:
    name: str = ""
    size: int = 1
    is_alive: bool = True
    health: int = 25
    damage: int = 5
    speed: int = 1
    # alpha: float = 0.1
    # gamma: float = 0.6
    # epsilon: float = 0.1
    # lr: float = 0.1
    # agent specific parameters
    # up, down, left, right
    action_space: spaces.Space = spaces.Discrete(4)

    def draw(self):
        pass

    def update_health(self, value):
        '''
            Updates health of a unit on which the method is called

            Input:
                value - damage inflicted by an enemy
        '''
        self.health = self.health - value
        if self.health < 0:
            self.is_alive = False

    def inflict_damage(self, enemies):
        '''
            Inflicts damage to an enemy with account on the following random variables:
                `theta` - ability of an enemy to escape the attack 
                `eta` - ability to variate power of the attack
            
            Input:
                enemies - numpy array of enemies to attack
            
            Output:
                damage - the value of damage inflicted to an enemy
        '''
        theta, eta = np.random.rand(), np.random.uniform(0, 1.75) 
        return theta*(eta*self.damage)


