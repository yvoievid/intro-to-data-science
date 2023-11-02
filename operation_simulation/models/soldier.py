from .unit import Unit

class Soldier(Unit):
    def __init__(self, position, name="Recrut",  step=1, size=1):
        super().__init__(position, name, step, size)
    
    @staticmethod
    def get_soldier_step():
        return 1