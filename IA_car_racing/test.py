import argparse
import os
import time
import torch
import numpy as np
import pickle

import car_racing
from model import BehavioralCloningModel
from util import data_transform, test_seeds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="models\model.pkl", help="File path with model to test"
)
args = parser.parse_args()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)


def save_result(result, path_to_save="results"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_to_save)
    file_name = "results_{}.pkl".format(time.strftime("%d%m%Y_%H%M", time.localtime()))
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(result, f)


def test(model, env, rendering=True, max_timesteps=3000):
    result = {}
    result["model"] = model
    result["maps"] = test_seeds
    for seed in test_seeds:
        result[seed] = {}
    for seed in test_seeds:
        env.seed(seed)
        episode_reward = 0
        step = 0
        state = env.reset()
        while True:
            state = np.array(state)
            state = data_transform(state)
            state = state.unsqueeze(0)
            model.eval()
            model_action = model(state)
            a = model_action.detach().numpy()[0]

            state, r, done, info = env.step(a)
            episode_reward += r
            step += 1

            if rendering:
                env.render()

            if done or step > max_timesteps:
                break
        print("Track reward: {}".format(episode_reward))
        result[seed]["reward"] = episode_reward
    save_result(result)


if __name__ == "__main__":
    model = BehavioralCloningModel()
    model.load_state_dict(torch.load(model_path))
    env = car_racing.CarRacing()
    env.render()
    test(model, env)
