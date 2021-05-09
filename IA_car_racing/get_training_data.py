import pickle
import os
import numpy as np
import time
import car_racing
from pyglet.window import key

if __name__ == "__main__":

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart, exit_env, pause
        if k == 0xFF0D: # Enter
            restart = True
        if k == 0xFF1B:  # escape
            exit_env = True
        if k == 0x020:  # space
            pause = not pause
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    def save_data(data, path_to_save='data'):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_to_save)
        file_name = "data_{}.pkl".format(time.strftime("%d%m%Y_%H%M", time.localtime()))
        with open(os.path.join(path,file_name), "wb" ) as f:
            pickle.dump(data, f)

    env = car_racing.CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    
    data = []

    prev_s = env.reset()
    total_reward = 0.0
    restart = False
    exit_env = False
    pause = False
    steps = 0
    while True:
        action_to_save = np.copy(a)
        s, r, done, info = env.step(a)
        data.append((prev_s, action_to_save))
        prev_s = s
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        isopen = env.render()

        if done or restart or isopen == False:
            prev_s = env.reset()
        
        while pause:
            env.render()
            time.sleep(0.1)
        
        if exit_env:
            break

    env.close()
    save_data(data)
