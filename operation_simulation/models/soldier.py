from .unit import Unit
from dataclasses import dataclass

@dataclass
class Soldier(Unit):
    ammo: int = 30
    cover_area: int = 1

    def attack_enemy(self, enemies):
        target = enemies[np.random.randint(len(enemies))]
        damage = self.inflict_damage(target)
        target.update_health(damage)