import random

from config import BOX_COUNT


class Game:

    def __init__(self, seed=None):

        self.rng = random.Random(seed)

        self.number = self.rng.randint(0, BOX_COUNT - 1)
        self.turns = 0
        self.finished = False

    def guess(self, position):

        self.turns += 1

        if position == self.number:
            self.finished = True
            return True

        if self.rng.randint(1, 2) == 1:
            self.number = (self.number + 1) % BOX_COUNT
        else:
            self.number = (self.number - 1) % BOX_COUNT

        return False