import retro
import random


def main():
    env = retro.make(game='SuperMarioWorld-Snes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(movRightOrJump())
        env.render()
        print("Reward ", rew, "Done: ", done, "info: ", info)
        if(info['lives'] == 3):
            done= True
        if done:
            obs = env.reset()
    env.close()

def movRightOrJump():
    if(random.random()>=0.5):
        return [1,0,0,0,0,0,0,1,0,0,0,0]
    else:
        return [0,0,0,0,0,0,0,1,0,0,0,0]
if __name__ == "__main__":
    main()
