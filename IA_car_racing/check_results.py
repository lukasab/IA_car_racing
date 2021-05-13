import os
import pickle

def check_results(path):
    filenames = os.listdir(path)
    filenames = ["results\\"+filename for filename in filenames if filename.endswith(".pkl" )]
    results = []
    for file in filenames:
        with open(file, "rb") as f:
            data = pickle.load(f)
            data["file"] = file
            results.append(data)

    for result in results:
        print(result["file"])
        mean_reward = 0
        for seed in result["maps"]:
            print("Map {}, reward: {}".format(seed, result[seed]["reward"]))
            mean_reward += result[seed]["reward"]
        mean_reward = mean_reward/len(result["maps"])
        print("Mean reward: {}". format(mean_reward))

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"results")
    check_results(path)