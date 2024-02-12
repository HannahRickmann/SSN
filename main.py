from experiments.experiment import Experiment

from tqdm.auto import tqdm

max_int = 10
length = 13224708

experiment = Experiment(50)
#experiment.construct_data(max_int)

experiment.read_constructed_data(max_int, length)

for id in tqdm(range(length)):
    experiment.choose_experiment(id)
    experiment.run(5)

for i in tqdm(range(100000)):
    break
    experiment = RandomExperiment(n, i)
    experiment.run(10)