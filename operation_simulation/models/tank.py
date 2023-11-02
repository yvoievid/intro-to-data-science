from .unit import Unit

class Tank(Unit):
    def __init__(self, position, name,  step, size):
        super().__init__(position, name, step, size)
