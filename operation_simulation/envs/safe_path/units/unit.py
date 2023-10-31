from abc import ABC, abstractmethod

class Unit(ABC):
    def __init__(self):
        self.position = [0,0]