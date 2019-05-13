import pygame

from numpy import random

from settings import PIPE_VELOCITY, PIPE_WIDTH, PIPE_GAP, PIPE_HEIGHT
from settings import WINDOW_SIZE, BIRD_SCALE


class Pipe:

    def __init__(self, image, x=None):
        self.imagel = image
        self.imageu = image
        self.x = WINDOW_SIZE[0] if x is None else x
        self.height = random.randint(*PIPE_HEIGHT)
        self.imageu = pygame.transform.flip(self.imageu, False, True)
        self.imagel = pygame.transform.scale(self.imagel, (PIPE_WIDTH, WINDOW_SIZE[1] - self.height - PIPE_GAP))
        self.imageu = pygame.transform.scale(self.imageu, (PIPE_WIDTH, self.height))

    def draw(self, surface):
        self.x -= PIPE_VELOCITY
        if self.x + PIPE_WIDTH > 0:
            surface.blit(self.imageu, (self.x, 0))
            surface.blit(self.imagel, (self.x, self.height + PIPE_GAP))

    def is_seen(self):
        return self.x + PIPE_WIDTH > 0

    def is_touching(self, bird):
        if bird.x + BIRD_SCALE[0] < self.x or bird.x > self.x + PIPE_WIDTH:
            return False

        if bird.y > self.height and bird.y + BIRD_SCALE[1] < self.height + PIPE_GAP:
            return False

        return True
