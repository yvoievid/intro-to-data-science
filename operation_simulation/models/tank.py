from .unit import Unit
from dataclasses import dataclass
import numpy as np

@dataclass
class Tank(Unit):
    fuel: int = 100
    speed: int = 4
    power: int = 10
    cover_area: int = 3
    health: int = 45
    damage: int = 15
    
    def attack_enemy(self, enemies):
        size = 3 if len(enemies) >= 3 else len(enemies)
        ids = np.random.choice(a=len(enemies), size=size, replace=False)
        for i in ids:
            target = enemies[i]
            damage = self.inflict_damage(target)
            target.update_health(damage)