import retro
import numpy as np
import cv2
import neat
import pickle
import math
import random



def evaluateGenome(genomes, config):
    env =  retro.make(game='SuperMarioWorld-Snes',state=levelStack[random.randint(0,1)])
    screenArray=[]
        
    for genome_id, genome in genomes:
        image= env.reset()
        actionVariable= env.action_space.sample()

        resX, resY, resC = env.observation_space.shape

        newResX= math.floor(resX/8)
        newResY= math.floor(resY/8)

        chimeraNeat = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame=0
        counter=0
        marioX=0
        marioY=0
        maxX=0
        maxY=370
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


            env.render()


            screenArray= np.ndarray.flatten(image)

            chimeraOutput =  chimeraNeat.activate(screenArray)
            
            innerGenomeOutput = nnExecuter(chimeraOutput, screenArray)

            #print( env.step(nnOutput))
            if(innerGenomeOutput is not None):
                image, rew, done, info = env.step(innerGenomeOutput)
            else:
                #print("igo is null?")
                image, rew, done, info = env.step([0,0,0,0,0,0,0,0,0,0,0,0])

            marioX=info['marioX']
            marioY=info['marioY']
            endTimer=info['endTimer']
            lives=info['lives']
            timer= (info['t1']*100)+(info['t2']*10)+(info['t3'])

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
                print("Genome: ", genome_id, ", Fitness Achieved: ", fitness_current)

            if(counter>=300):
                genome.fitness = fitness_current
                done= True
                print("Genome: ", genome_id, ", Fitness Achieved: ", fitness_current)
            
            genome.fitness = fitness_current







#Temp solution
def nnExecuter(nnOutputRaw, screenArray):
    nnOutput = [0,0,0]
    count = 0
    for a in nnOutputRaw:        
        if a>0.5:
            nnOutput[count]=1.0
        else:          
            nnOutput[count]=0.0
        count+=1
    #print(nnOutput)

    if(nnOutput==[0.0,0.0,0.0]):
        #print("none")
        return [0,0,0,0,0,0,0,0,0,0,0,0]

    if(nnOutput==[0.0,0.0,1.0] or nnOutput==[0.0,1.0,1.0]):
        #print("nn 0")
        with open(('./0/0.pkl'), 'rb') as inputG:
            innerGenome=pickle.load(inputG)
        nn = neat.nn.recurrent.RecurrentNetwork.create(innerGenome, innerConfig)
        return nn.activate(screenArray)
    
    if(nnOutput==[1.0,1.0,1.0] or nnOutput==[1.0,1.0,0.0]):
        #print("nn 1")
        with open(('./1/1.pkl'), 'rb') as inputG:
            innerGenome=pickle.load(inputG)
        nn = neat.nn.recurrent.RecurrentNetwork.create(innerGenome, innerConfig)
        return nn.activate(screenArray)
    
    if(nnOutput==[1.0,0.0,0.0] or nnOutput==[0.0,1.0,0.0]):
        #print("nn 2")        
        with open(('./2/2.pkl'), 'rb') as inputG:
            innerGenome=pickle.load(inputG)
        nn = neat.nn.recurrent.RecurrentNetwork.create(innerGenome, innerConfig)
        return nn.activate(screenArray)

    





levelStack = ["YoshiIsland2.state", "YoshiIsland3.state"]
tempNNStack = ["./1/1.pkl","./0/0.pkl","./2/2.pkl"]

chimeraConfig = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'chimeraConfig')

innerConfig = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'innerConfig')

p = neat.Population(chimeraConfig)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


p.add_reporter(neat.Checkpointer(25, 1500, "chimera-checkpoint"))


#winner = p.run(evaluateGenome)
p.run(evaluateGenome)

'''
with open('chimeraWinner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
'''