from dataclasses import dataclass
from operation_simulation.constants import Commands, Flanks, GroupNames, Strategies, Weather

@dataclass
class Inference: 
    command: str = Commands.ATTACK
    flank: str = Flanks.CENTER
    group: str = GroupNames.ALPHA
    weather: str = Weather.SUMMER
    strategy: str = Strategies.AGGRESSIVE
    train: bool = False
    simulate: bool = False
    dryrun: bool = False