###########################################
#   _____    _____    ___ _
#  |   __|  |  _  |  |   | |
#  |  |  |  |     |  | | | |
#  |_____|  |__|__|  |_|___|
#
###########################################
POPULATION = 10

MUTATION_RATE = 0.3

CROSSOVER_RATE = 0.8

SELECTION_PERCENTAGE = 0.4

MUTATION_PERCENTAGE = 0.6

CROSSOVER_PERCENTAGE = 0.1

INPUT_LAYER = 4

HIDDEN_LAYER = 6

OUTPUT_LAYER = 1

NO_OF_GENERATIONS = 500
#############################################

DATA_PATH = "data/"

WINDOW_SIZE = (380, 600)
GRAVITY = 0.3
BACKGROUND = "res/img/background.png"
FPS = 180

FONT = "res/fonts/FreeSansBold.ttf"
FONT_SIZE = 40
FONT_COLOR = (255, 255, 255)
TEXT_CENTER = (int(WINDOW_SIZE[0]/2), 140)

BIRD_JUMP_SPEED = 6
BIRD_SCALE = (50, 40)
BIRD_Y_POS = WINDOW_SIZE[1]/2
BIRD_X_POS = 40
IS_BIRD_RANDOM_POS = False
BIRD_INITIAL_RANGE = [50, WINDOW_SIZE[1] - 20 - BIRD_SCALE[1]]

BIRD_IMAGES = ["res/img/frame-{}.png".format(i) for i in range(1, 5)]

PIPE_WIDTH = 60
PIPE_VELOCITY = 2
PIPE_GAP = 180
IS_PIPE_VERTICAL_VELOCITY = False
PIPE_HEIGHT = [50, WINDOW_SIZE[1] - 50 - PIPE_GAP]
PIPE_IMAGE = "res/img/pipe.png"
PIPE_FREQ = 120

SVM_DATA_ONES = "data/ones.txt"
SVM_DATA_ZEROS = "data/zeros.txt"

DATA_COLLECTION_FREQ = 10
