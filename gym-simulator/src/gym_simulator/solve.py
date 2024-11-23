import dataclasses

from tqdm import tqdm
from dataset_generator.core.gen_dataset import generate_dataset
from gym_simulator.algorithms.cp_sat import CpSatScheduler
from gym_simulator.args import TESTING_DS_ARGS
from gym_simulator.environments.static import StaticCloudSimEnvironment

dataset_args = TESTING_DS_ARGS

total_makespan = 0
for i in tqdm(range(10)):
    env_config = {
        "seed": i,
        "simulator_mode": "internal",
        "simulator_kwargs": {"dataset_args": dataclasses.asdict(TESTING_DS_ARGS)},
    }
    scheduler = CpSatScheduler(timeout=60)
    env = StaticCloudSimEnvironment(env_config)

    (tasks, vms), _ = env.reset(seed=env_config["seed"])
    vm_assignments = scheduler.schedule(tasks, vms)
    _, _, terminated, truncated, info = env.step(vm_assignments)
    assert terminated or truncated

    env.close()

    total_makespan += scheduler._makespan

print(total_makespan / 10)
