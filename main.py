from experiments.experiment import Experiment
from experiments.random_experiment import RandomExperiment

from tqdm import tqdm

experiment = Experiment(6)
experiment.run(10)
experiment.print()

n = 4

for i in tqdm(range(10000)):
    experiment = RandomExperiment(n, i)
    experiment.run(10)