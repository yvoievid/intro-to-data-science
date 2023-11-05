from abc import ABC, abstractmethod

class Unit(ABC):
    def __init__(self, position, name, step, size):
        self._position = position
        self._name = name
        self._step = step
        self._size = size
        self._type = None
    
    @property
    def step(self):
        return self._step
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
        
    @property
    def size(self):
        return self._size
    
    @property
    def name(self):
        return self._name
    
    def draw(self):
        pass


