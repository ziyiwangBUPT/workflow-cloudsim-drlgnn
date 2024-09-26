import tyro
import dataclasses

from gym_simulator.algorithms.round_robin import round_robin
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    render_mode: str | None = None
    """render mode"""
    host_count: int = 4
    """number of hosts"""
    vm_count: int = 10
    """number of VMs"""
    workflow_count: int = 20
    """number of workflows"""
    task_limit: int = 30
    """maximum number of tasks"""


def main(args: Args):
    env = StaticCloudSimEnvironment(
        env_config={
            "host_count": args.host_count,
            "vm_count": args.vm_count,
            "workflow_count": args.workflow_count,
            "task_limit": args.task_limit,
            "simulator_mode": "embedded",
            "simulator_kwargs": {"simulator_jar_path": args.simulator},
            "render_mode": args.render_mode,
        },
    )

    # Since this is static, the step will be only called once
    # So there will not be loops
    # But we are iterating N times to make sure results are consistent
    rewards: list[float] = []
    for i in range(10):
        print(f"Step {i}")
        (tasks, vms), _ = env.reset()
        action = round_robin(tasks, vms)
        _, reward, terminated, truncated, _ = env.step(action)
        assert terminated or truncated, "Static environment should terminate after one step"
        rewards.append(reward)

    print(f"Average reward: {sum(rewards) / len(rewards)}")
    print(f"Variance: {sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)}")
    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
