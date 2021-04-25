import retro
import numpy as np
import cv2
import neat
import pickle
import math



#actionVariable= env.action_space.sample()

#resX, resY, resC = env.observation_space.shape

#print(env.observation_space.shape)

class Worker(object):
    def __init__(self, genome, config):
        self.genome =  genome
        self.config = config

    def work(self):

        screenArray=[]

        self.env =  retro.make(game='SuperMarioWorld-Snes',state='DonutPlains1.state')


        image= self.env.reset()
        actionVariable= self.env.action_space.sample()
        resX, resY, resC = self.env.observation_space.shape

        newResX= math.floor(resX/8)
        newResY= math.floor(resY/8)

        nNeat = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        fitness_current= 0
        maxFitness = 0
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
            #image = np.reshape(image,(newResX, newResY))
            '''
            cv2.imshow('main', image)
            cv2.waitKey(1)
            '''
            #self.env.render()
            screenArray= np.ndarray.flatten(image)

            nnOutput =  nNeat.activate(screenArray)

            image, rew, done, info = self.env.step(nnOutput)

            marioX=info['marioX']
            marioY=info['marioY']
            endTimer=info['endTimer']
            lives=info['lives']
            timer= (info['t1']*100)+(info['t2']*10)+(info['t3'])

            #pitafall death
            '''
            if(marioY>400):
                #fitness_current-=300
                if(maxX<=500 and maxY>112):
                    fitness_current -= maxX
                    done= True
                    break
                else:
                    print("else")
                    fitness_current-=300
                    done= True

                break
            '''

            ## TEMP fitness:
            if((marioX)>maxX):
                fitness_current=marioX
                counter =0
                maxX=marioX
                ##genome.fitness = fitness_current
            else:
                counter+=1
            #height TODO only if in level
            '''
            if((marioY)<maxY):
                fitness_current+=1
                #counter =0
                maxY=marioY
                ##genome.fitness = fitness_current
            '''


            if(fitness_current>maxFitness):
                maxFitness=fitness_current
                counter=0
            elif(fitness_current==maxFitness):
                counter+=1





            if(endTimer!=0):
                fitness_current+=timer*10

                counter = 0

                done= True
                pass #print( ", Fitness Achieved: ", fitness_current)




            if(counter>=500):
                done= True
                pass #print(" Fitness Achieved: ", fitness_current)

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
                    pass #print(" Fitness Achieved: ", fitness_current)
            curLives=lives

        return fitness_current




def evaluateGenome(genomes, config):

    nWorker =  Worker(genomes, config)
    return nWorker.work()










config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neatconfig')
# create population
p = neat.Population(config)
#Change to cores - 1 or 2 of max threads on cpu
threads = 12
pe = neat.ParallelEvaluator(threads, evaluateGenome)

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-25')
# add reporters so you can get some nice stats
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# save a check point file every iteration
p.add_reporter(neat.Checkpointer(10))


# this line runs the previous eval_genomes function. Once done, the best is set to winner
winner = p.run(pe.evaluate)

#saves a pickle file of the winning genome.
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
