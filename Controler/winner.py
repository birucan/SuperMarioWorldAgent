import retro
import numpy as np
import cv2
import neat
import pickle
import math

env =  retro.make(game='SuperMarioWorld-Snes',state='YoshiIsland2.state')




config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neatconfig')
# create population
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
# add reporters so you can get some nice stats
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# save a check point file every 10 iterations
p.add_reporter(neat.Checkpointer(10))

with open('winner.pkl', 'rb') as inputG:
    genome=pickle.load(inputG)
screenArray=[]

image= env.reset()
actionVariable= env.action_space.sample()

resX, resY, resC = env.observation_space.shape

newResX= math.floor(resX/8)
newResY= math.floor(resY/8)

nNeat = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
current_max_fitness = 0
fitness_current = 0
frame=0
counter=0
marioX=0
marioY=0
maxX=0
done=False
endTimer=0
timer=0
lives=0
curLives=5
marioStillX=0
stillTimer=0
oldTime=0

## TEMP:
maxX=0




while(not done):

    frame+=1
    image = cv2.resize(image,(newResX,newResY))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = np.reshape(image,(newResX, newResY))

    cv2.imshow('main', image)
    cv2.waitKey(1)

    env.render()

    screenArray= np.ndarray.flatten(image)

    nnOutput =  nNeat.activate(screenArray)

    image, rew, done, info = env.step(nnOutput)





    marioX=info['marioX']
    marioY=info['marioY']
    endTimer=info['endTimer']
    lives=info['lives']
    timer= (info['t1']*100)+(info['t2']*10)+(info['t3'])

    ## TEMP fitness:
    if((marioX)>maxX):
        fitness_current=marioX
        maxX=marioX
        genome.fitness = fitness_current

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1


    if(endTimer!=0):
        genome.fitness = fitness_current
        fitness_current+=timer*10

        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0

        done= True
        #print("Genome: ", genome_id, ", Fitness Achieved: ", fitness_current)



    if(counter>=500):
        genome.fitness = fitness_current
        done= True
        #print("Genome: ", genome_id, ", Fitness Achieved: ", fitness_current)

    '''
    if(marioX!=marioStillX):
        stillTimer=0
    else:
        stillTimer+=1


    print(stillTimer)
    if(stillTimer>=500 ):
        done= True
        print("Genome: ", genome_id, ", Fitness Achieved: ", fitnessCurrent)
        stillTimer=0

    marioStillX=marioX
    '''

    if(lives!=curLives):
        if(curLives!=0):
            print("lost life")
    curLives=lives

    genome.fitness = fitness_current
#vw.release()
