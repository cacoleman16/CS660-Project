import os
import neat
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import pickle
from envWrapper import make_env
import pickle
import multiprocessing as mp
import numpy as np




class Train:
    def __init__(self, generations, parallel=8, level="1-1"):
        self.generations = generations
        self.lock = mp.Lock()
        self.par = parallel
        self.level = level

    def _get_actions(self, actions):

        return np.random.choice(np.flatnonzero(actions == actions.max()))

    def init_values(self):
        self.counter = 0
        self.score = 0
        self.xpos = 0
        self.xpos_max = 0
        self.stage = 0

    def update_fitness(self, info, fitness_current):

        if info['status'] != 'small':
            bonus = 100
        else:
            bonus = 0

        if self.stage == info['stage'] & self.xpos <= info['x_pos']:
            fitness_current -= 50

        self.score = info['score']
        self.xpos = info['x_pos']
        self.stage = info['stage']
        if self.xpos > self.xpos_max:
            fitness_current += 10
            self.xpos_max = self.xpos

        fitness_current += info['score'] + info['coins']

        return fitness_current

    def _fitness_func_no_parallel(self, genomes, config):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = make_env(env)
        idx, genomes = zip(*genomes)
        for genome in genomes:
            try:
                state = env.reset()
                net = neat.nn.recurrent.RecurrentNetwork.create(
                    genomes, config)
                self.init_values()
                current_max_fitness = 0
                fitness = 0
                state = self.env.reset()
                done = False
                old = 0
                while not done:
                    env.render()
                    nnOutput = net.activate(state.flatten())
                    actions = np.array(nnOutput)
                    action = self._get_actions(actions)
                    nextState, reward, done, info = env.step(action)
                    fitness = self.update_fitness(info, fitness)
                    state = nextState

                    self.counter += 1
                    if self.counter % 50 == 0:
                        if (old == self.xpos) & (self.level == info['stage']):
                            break
                        else:
                            old = self.xpos

                genome.fitness = fitness
                env.close()
            except KeyboardInterrupt:
                env.close()
                exit()

    def _fitness_func(self, genome, config, o):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = make_env(env)
        # env.configure(lock=self.lock)
        try:
            state = env.reset()
            self.init_values()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            done = False
            fitness = 0
            old = 0
            while not done:
                if self.counter > 5000:
                    env.render()
                nnOutput = net.activate(state.flatten())
                actions = np.array(nnOutput)
                action = self._get_actions(actions)
                nextState, reward, done, info = env.step(action)
                fitness = self.update_fitness(info, fitness)
                state = nextState
                self.counter += 1
                if self.counter % 50 == 0:
                    if (old == self.xpos) & (self.level == info['stage']):
                        break
                    else:
                        old = self.xpos

            if fitness >= 10000:
                pickle.dump(genome, open("finisher.pkl", "wb"))
                env.close()
                print("Done")
                exit()
            o.put(fitness)
            print(" Not Done")
            env.close()
        except KeyboardInterrupt:
            print(" keyboard interruptttttt")
            env.close()
            exit()

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.par):
            output = mp.Queue()

            processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in
                         genomes[i:i + self.par]]

            [p.start() for p in processes]
            [p.join() for p in processes]

            results = [output.get() for p in processes]

            for n, r in enumerate(results):
                genomes[i + n].fitness = r

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        print("loaded checkpoint...")
        winner = p.run(self._eval_genomes, n)
        win = p.best_genome
        pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open('real_winner.pkl', 'wb'))


    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)


if __name__ == "__main__":
    t = Train(1000)
    t.main()
