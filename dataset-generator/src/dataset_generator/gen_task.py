import random
import math

from scipy import stats


def generate_poisson_delay(lam: float) -> float:
    """
    Generate a random delay between workflows in a Poisson process.
    The delay is exponentially distributed with parameter lambda.
    """

    return stats.expon.rvs(scale=1 / lam)


def generate_task_length(method: str, low: float, high: float) -> float:
    """
    Generate a random task length based on the specified method. <br/>
    Available methods: uniform, normal, left_skewed, right_skewed <br/>
    """

    if method == "uniform":
        return stats.uniform.rvs(loc=low, scale=high - low)

    # 99.7% of the data is within 3 standard deviations (by empirical rule)
    # so we set the standard deviation to be 1/6 of the range
    mean = (low + high) / 2
    std = (high - low) / 6
    if method == "normal":
        return stats.norm.rvs(loc=mean, scale=std)
    elif method == "left_skewed":
        return stats.skewnorm.rvs(-5, loc=mean, scale=std)
    elif method == "right_skewed":
        return stats.skewnorm.rvs(5, loc=mean, scale=std)

    raise ValueError(f"Invalid method: {method}")


def generate_dag(n: int, p: float | None = None) -> dict[int, set[int]]:
    """
    Generate a random Directed Acyclic Graph (DAG) using the G(n, p) model.
    The resulting graph is represented as an adjacency list. <br/>
    The resulting graph has n + 1 nodes, with node 0 being the starting node. <br/>
    If p is None, it is set to log(n + eps) / n where n is the number of generated nodes.
    """

    if n == 1:
        return {0: set()}

    if p is None:
        p = math.log(n + 0.1) / n

    nodes: dict[int, set[int]] = {i: set() for i in range(n + 1)}
    start_nodes: set[int] = set(range(1, n + 1))

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if random.random() < p:
                nodes[i].add(j)
                start_nodes.discard(j)

    for i in start_nodes:
        nodes[0].add(i)

    return nodes
