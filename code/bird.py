import numpy as np

from settings import BIRD_JUMP_SPEED, IS_BIRD_RANDOM_POS, BIRD_INITIAL_RANGE
from settings import BIRD_Y_POS, BIRD_X_POS, GRAVITY, PIPE_VELOCITY


class Bird:

    def __init__(self, images):
        self.image = images
        self.frame_no = 0
        self.cycle = True
        self.v = 0
        self.g = GRAVITY
        self.y = np.random.randint(*BIRD_INITIAL_RANGE) if IS_BIRD_RANDOM_POS else BIRD_Y_POS
        self.x = BIRD_X_POS

    def draw(self, surface):
        self.y += self.v
        self.v += GRAVITY
        if self.frame_no < 0:
            self.cycle = True
            self.frame_no = 1
        elif self.frame_no > 30:
            self.cycle = False
            self.frame_no = 29

        surface.blit(self.image[int(self.frame_no/10)], (self.x, self.y))

        if self.cycle:
            self.frame_no += 1
        else:
            self.frame_no -= 1

    def jump(self):
        self.v = -BIRD_JUMP_SPEED

    def reset(self):
        self.y = np.random.randint(*BIRD_INITIAL_RANGE)
        self.v = PIPE_VELOCITY
        self.cycle = True
        self.frame_no = 0