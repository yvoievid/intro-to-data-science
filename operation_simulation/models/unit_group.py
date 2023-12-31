from dataclasses import dataclass, field

@dataclass
class UnitGroup:
    units: list
    q_table: list = field(default_factory=list)
    speed: int = 1
    name: str = ""
    state: int = 0
    position: list = field(default_factory=list)
    
    def get_cover_area(self):
        units_cover_areas = [unit.cover_area for unit in self.units]
        return max(units_cover_areas)
    
    def calculate_state(self):
        return self.position[0] * self.position[1] 