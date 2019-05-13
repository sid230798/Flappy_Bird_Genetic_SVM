import pygame
import numpy as np

from bird import Bird
from pipes import Pipe
from NeuralModel import Neural

from settings import BIRD_IMAGES, PIPE_IMAGE, BACKGROUND, PIPE_VELOCITY, TEXT_CENTER
from settings import BIRD_SCALE, POPULATION, WINDOW_SIZE, FPS, PIPE_WIDTH, FONT_COLOR
from settings import NO_OF_GENERATIONS, DATA_COLLECTION_FREQ,  PIPE_FREQ, FONT, FONT_SIZE
from settings import SVM_DATA_ONES, SVM_DATA_ZEROS


class Flappy:

    def __init__(self, model="gan"):
        pygame.init()
        global POPULATION
        self.display = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Smart Flappy Bird")
        self.clock = pygame.time.Clock()
        self.birds = []
        self.pipes = []
        self.score = 0
        if model != "gan":
            POPULATION = 1
        self.fitness = [0 for _ in range(POPULATION)]
        self.bird_alive = [True for _ in range(POPULATION)]
        self.pipe_image = None
        self.background = None
        self.font = None
        self.model = model
        self.set_background()


    def init(self):
        bird_frames = Flappy.get_frames(BIRD_IMAGES)
        for i in range(POPULATION):
            self.birds.append(Bird(bird_frames))

        pipe_image = pygame.image.load(PIPE_IMAGE)
        self.pipe_image = pipe_image
        self.font = pygame.font.Font(FONT, FONT_SIZE)

    def reset(self):
        self.pipes = []
        self.fitness = [0 for _ in range(POPULATION)]
        self.bird_alive = [True for _ in range(POPULATION)]
        for i in range(POPULATION):
            self.birds[i].reset()

        self.score = 0

    def smart_run(self, clf, mean=0, std=1):

        file0 = open(SVM_DATA_ZEROS, "a")
        file1 = open(SVM_DATA_ONES, "a")

        close = False

        no_of_gen = NO_OF_GENERATIONS
        if self.model.lower() != "gan":
            no_of_gen = 1

        for j in range(no_of_gen):

            crashed = False
            self.pipes.append(Pipe(self.pipe_image, x=200))

            freq_pipe = PIPE_FREQ
            is_add = freq_pipe - 40

            data_freq = DATA_COLLECTION_FREQ
            is_collect = data_freq

            while (not crashed) and (not close):
                action = -1
                data = self.get_sample(Is=True, mean=mean, std=std)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        close = True

                for i in range(POPULATION):
                    if not self.bird_alive[i]:
                        continue
                    sample = self.get_sample(bird_index=i, mean=mean, std=std)
                    action = clf[i].predict(sample)[0]
                    if action == 1:
                        if self.model.lower() == "gan-collect":
                            self.collect_data(file1, data, action)
                        self.birds[i].jump()

                if is_collect % data_freq == 0 and self.model.lower() == "gan-collect":
                    if action == -1:
                        self.collect_data(file0, data, action)
                is_collect = (is_collect + 1) % data_freq

                if is_add % freq_pipe is 0:
                    self.pipes.append(Pipe(self.pipe_image))
                is_add = (is_add + 1) % freq_pipe

                self.draw()

                if self.check_collision():
                    crashed = True

            if self.model.lower() == "gan":
                neural_list, fitness = self.sorted_clf(clf)
                clf = Neural.create_new_generation(neural_list)
            else:
                fitness = self.fitness[0]
            print("Generation = {}, fitness = {}".format(j, fitness))

            self.reset()

            if close:
                break

        pygame.quit()

        file0.close()
        file1.close()

        return clf[0]

    def sorted_clf(self, clf):
        fitness_clf = dict()
        for i in range(POPULATION):
            fitness_clf[clf[i]] = self.fitness[i]

        sorted_neural = sorted(fitness_clf.items(), key=lambda kv: kv[1])
        sorted_neural = sorted_neural[-1::-1]
        clf_sorted = []
        for i in range(POPULATION):
            clf_sorted.append(sorted_neural[i][0])

        return clf_sorted, sorted_neural[0][1]

    def check_collision(self):
        pipes = []
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH <= 0:
                continue
            pipes.append(pipe)
        if len(pipes) == 1:
            pipes = [pipes[0].x, 800]
        else:
            pipes = [pipes[0].x, pipes[1].x]
        not_alive = 0
        for i in range(POPULATION):
            if self.bird_alive[i]:
                if self.is_collision(i):
                    self.bird_alive[i] = False
                    for_pipe_1 = pipes[0] - self.birds[i].x
                    for_pipe_2 = pipes[1] - self.birds[i].x
                    nearest = for_pipe_1 if abs(for_pipe_1) < abs(for_pipe_2) else for_pipe_2
                    self.fitness[i] -= nearest
                    not_alive += 1
            else:
                not_alive += 1

        return not_alive == POPULATION

    def get_sample(self, mean=0, std=1, Is=False, bird_index=0):
        pipes = []
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH + 10 < self.birds[bird_index].x:
                continue
            pipes.append(pipe)
        v = self.birds[bird_index].v
        if len(pipes) >= 1:
            dx, dy = (pipes[0].x - self.birds[bird_index].x), (self.birds[bird_index].y - pipes[0].height - 100)
        else:
            dx, dy = 380, self.birds[bird_index].y - 100

        h = WINDOW_SIZE[1] - self.birds[bird_index].y
        sample = "{} {} {} {}".format(v, h, dx, dy)
        if Is:
            return sample
        sample = np.array([float(x) for x in sample.split()]).reshape(1, -1)
        if self.model.lower()[:3] != "gan":
            if self.model.lower() == "linearsvm":
                sample = np.hstack([sample, sample**2, sample**3])
            sample = (sample - mean) / std
            return sample
        return sample.T

    def predict(self, clf, mean, std):
        sample = self.get_sample(mean, std)
        action_p = clf.predict(sample)[0]
        # print("action = {}".format(action_p))
        return action_p

    def collect_data(self, file, data, action):
        line = "{} {}\n".format(data, action)
        file.write(line)

    def is_collision(self, bird_index=0):
        if self.birds[bird_index].y + BIRD_SCALE[1] > WINDOW_SIZE[1]:
            return True

        if self.birds[bird_index].y < 0:
            return True

        for pipe in self.pipes:
            if pipe.is_touching(self.birds[bird_index]):
                return True
        else:
            return False

    def draw(self):
        """
        draw background, bird and pipes, update display
        """
        self.display.blit(self.background, (0, 0))
        for i in range(POPULATION):
            if self.bird_alive[i]:
                self.fitness[i] += PIPE_VELOCITY
                self.birds[i].draw(self.display)
        self.draw_pipes()
        self.score_display()
        pygame.display.update()
        self.clock.tick(FPS)

    def draw_pipes(self):
        """
        draw pipes on the display and remove zombie pipes
        """
        no_of_pipes = len(self.pipes)
        zombie_pipes = []
        for i in range(no_of_pipes):
            if not self.pipes[i].is_seen():
                zombie_pipes.append(i)
                self.score += 1
                continue
            self.pipes[i].draw(self.display)

        for index in zombie_pipes:
            self.pipes.pop(index)

    def score_display(self):
        text_surface = self.font.render(str(self.score), True, FONT_COLOR)
        text_rect = text_surface.get_rect()
        text_rect.center =TEXT_CENTER
        self.display.blit(text_surface, text_rect)

    @staticmethod
    def get_frames(images):
        """
        :param images: bird frame images list
        :param scale: bird image scale
        :return: pygame.Surface list for images
        """
        frames = []
        for image in images:
            frame = pygame.image.load(image)
            frame = pygame.transform.scale(frame, BIRD_SCALE)
            frames.append(frame)

        return frames

    def set_background(self):
        """
        :param background: background image address
        """
        background = pygame.image.load(BACKGROUND)
        background = pygame.transform.scale(background, WINDOW_SIZE)
        self.background = background
