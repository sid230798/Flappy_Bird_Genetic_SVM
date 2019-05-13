import os
import argparse
import pickle

from Flappy import Flappy
from model import get_model


def get_arguments():
    parser = argparse.ArgumentParser(description='provide model by command line flags --model')
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    return args.model


def main():
    model_name = get_arguments()
    if model_name.upper() == "GAN":
        print("GAN model")
        neural_list = get_model(model_name)
        print("running {}".format(model_name))
        game = Flappy(model_name.lower())
        game.init()
        best_neural = game.smart_run(neural_list)
        with open("model/gan_1.model", "wb") as f:
            pickle.dump(best_neural, f)
    elif model_name.lower() == "gan-collect":
        model, mean, std = get_model(model_name)
        game = Flappy(model_name.lower())
        game.init()
        game.smart_run(model, mean=mean, std=std)
    else:
        model, mean, std = get_model(model_name)
        print("running {}".format(model_name))
        game = Flappy(model_name.lower())
        game.init()
        game.smart_run(model, mean=mean, std=std)


if __name__ == '__main__':
    PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PATH)
    print("In Directory {}".format(PATH))
    main()
