from experiments.experiment import Experiment
from experiments.random_experiment import RandomExperiment

from tqdm.auto import tqdm

experiment = Experiment(16)
experiment.run(5)
experiment.print()

n = 2

for i in tqdm(range(100000)):
    break
    experiment = RandomExperiment(n, i)
    experiment.run(10)