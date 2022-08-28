
class RandomPool:
    def __init__(self, seed: int, num_states: int):
        self.seed: int = seed
        self.num_states: int = num_states

class Random_XorShift64_Pool(RandomPool):
    def __init__(self, seed: int, num_states: int):
        super().__init__(seed, num_states)

class Random_XorShift1024_Pool(RandomPool):
    def __init__(self, seed: int, num_states: int):
        super().__init__(seed, num_states)

def rand(dtype: type):
    """
    Generate a random number

    :param dtype: the data type of the number to be generated
    """

    pass