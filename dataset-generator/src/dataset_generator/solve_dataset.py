import dataclasses
import json
import tyro

from dataset_generator.core.models import Dataset, Solution
from dataset_generator.solvers.solver import solve
from dataset_generator.visualizers.printers import print_solution


@dataclasses.dataclass
class Args:
    method: str = "sat"
    """Method to solve the dataset"""


def main(args: Args):
    dataset_str = input()
    dataset_dict = json.loads(dataset_str)
    dataset = Dataset.from_json(dataset_dict)

    # Solution
    vm_assignments = solve(args.method, dataset)
    print_solution(dataset.workflows, vm_assignments)

    solution = Solution(dataset=dataset, vm_assignments=vm_assignments)
    json_data = json.dumps(dataclasses.asdict(solution))
    print(json_data)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
