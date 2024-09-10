import click
import random
import numpy as np

from dataset_generator.gen_vm import generate_hosts, generate_vms, allocate_vms


@click.command()
@click.option("--seed", default=0, help="Random seed", type=int)
@click.option("--host_count", default=100, help="Number of hosts")
@click.option("--vm_count", default=1000, help="Number of VMs")
@click.option("--max_cores", default=2, help="Maximum number of cores per VM")
@click.option("--min_cpu_speed_mips", default=1000, help="Minimum CPU speed in MIPS")
@click.option("--max_cpu_speed_mips", default=2000, help="Maximum CPU speed in MIPS")
def main(seed: int, host_count: int, vm_count: int, max_cores: int, min_cpu_speed_mips: int, max_cpu_speed_mips: int):
    random.seed(seed)
    np.random.seed(seed)

    hosts = generate_hosts(host_count)
    vms = generate_vms(vm_count, max_cores, min_cpu_speed_mips, max_cpu_speed_mips)
    vm_allocations = allocate_vms(vms, hosts)
    for vm_allocation in vm_allocations:
        print(vm_allocation)


if __name__ == "__main__":
    main()
