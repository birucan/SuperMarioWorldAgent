import retro
import numpy as np
import cv2
import neat
import pickle
import math

env =  retro.make(game='SuperMarioWorld-Snes',state='YoshiIsland2.state')

#actionVariable= env.action_space.sample()

#resX, resY, resC = env.observation_space.shape

#print(env.observation_space.shape)

def evaluateGenome(genomes, config):

    screenArray=[]

    for genome_id, genome in genomes:


        image= env.reset()
        actionVariable= env.action_space.sample()

        resX, resY, resC = env.observation_space.shape

        newResX= math.floor(resX/8)
        newResY= math.floor(resY/8)

        nNeat = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        curMaxFitness=0
        fitnessCurrent=0
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
            image = np.reshape(image,(newResX, newResY))

            env.render()

            screenArray= np.ndarray.flatten(image)

            nnOutput =  nNeat.activate(screenArray)

            image,rew,done,info = env.step(nnOutput)

            marioX=info['marioX']
            marioY=info['marioY']
            endTimer=info['endTimer']
            lives=info['lives']
            timer= (info['t1']*100)+(info['t2']*10)+(info['t3'])

            ## TEMP fitness:
            if((marioX)>maxX):
                fitnessCurrent+=1+info['t1']
                maxX=marioX

            if fitnessCurrent > curMaxFitness:
                curMaxFitness = fitnessCurrent
                counter = 0
            else:
                counter += 1


            if(endTimer!=0):
                fitnessCurrent+=500
                if fitnessCurrent > curMaxFitness:
                    curMaxFitness = fitnessCurrent
                    counter = 0

                done= True
                print("Genome: ", genome_id, ", Fitness Achieved: ", fitnessCurrent)


            if(marioX!=marioStillX):
                stillTimer=0
            else:
                stillTimer+=1


            #print(stillTimer)
            if(stillTimer>=500 ):
                done= True
                print("Genome: ", genome_id, ", Fitness Achieved: ", fitnessCurrent)
                stillTimer=0

            marioStillX=marioX

            if(lives!=curLives):
                if(curLives!=0):
                    print("Genome: ", genome_id, ", Fitness Achieved: ", fitnessCurrent)
            curLives=lives





config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neatconfig')
# create population
p = neat.Population(config)

# add reporters so you can get some nice stats
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# save a check point file every 10 iterations
p.add_reporter(neat.Checkpointer(10))

# if you have already trained some, and want to restore, uncomment this line and change the 'neat-checkpoint-10'
# to whichever file you want to use (they'll all be named neat-checkpoint-somenumber)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-10')

# this line runs the previous eval_genomes function. Once done, the best is set to winner
winner = p.run(evaluateGenome)

#saves a pickle file of the winning genome.
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
