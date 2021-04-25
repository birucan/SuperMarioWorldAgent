import retro
import numpy as np
import cv2
import neat
import pickle
import math



class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make(game='SuperMarioWorld-Snes',state='YoshiIsland2.state')

        self.env.reset()

        image, _, _, _ = self.env.step(self.env.action_space.sample())

        newResX = int(image.shape[0]/8)
        newResY = int(image.shape[1]/8)


        nNeat = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness=0
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

            #frame+=1
            image = cv2.resize(image,(newResX,newResY))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = np.reshape(image,(newResX, newResY))
            '''
            cv2.imshow('main', image)
            cv2.waitKey(1)
            '''
            #self.env.render()

            screenArray= np.ndarray.flatten(image)
            screenArray = np.interp(screenArray, (0,254),(-1, +1))
            actions = nNeat.activate(screenArray)

            #nnOutput =  nNeat.activate(screenArray)

            image, rew, done, info = self.env.step(actions)

            marioX=info['marioX']
            marioY=info['marioY']
            #print(info)
            endTimer=info['endTimer']
            lives=info['lives']
            timer= (info['t1']*100)+(info['t2']*10)+(info['t3'])

            ## TEMP fitness:
            if((marioX)>maxX):
                fitness=marioX
                maxX=marioX
                counter=0
            else:
                counter +=1


            '''
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            '''

            if(endTimer!=0):
                fitness+=timer*10
                done= True
                #print("Genome: ", self.genome.genome_id, ", Fitness Achieved: ", fitness_current)


            if(counter>=300):
                done= True
                #print("Genome: ",self.genome.genome_id, ", Fitness Achieved: ", fitness_current)

            '''
            if(lives!=curLives):
                if(curLives!=0):
                    print("Genome: ", self.genome.genome_id, ", Fitness Achieved: ", fitness_current)
            '''
            curLives=lives
        print(fitness)
        return fitness

def eval_genomes(genome, config):

    worky = Worker(genome, config)
    return worky.work()









config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neatconfig')
# create population
p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('CurrentY1')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(10, eval_genomes)



# this line runs the previous eval_genomes function. Once done, the best is set to winner
winner = p.run(pe.evaluate)

#saves a pickle file of the winning genome.
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
