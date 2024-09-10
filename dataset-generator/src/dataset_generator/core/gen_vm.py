import json
import random

from dataset_generator.core.models import Host, Vm, VmAllocation


def generate_hosts(n: int) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    Uses the host specifications from data/host_specs.json.
    """

    hosts: list[Host] = []
    with open("data/host_specs.json") as f:
        available_hosts: list = json.load(f)

    for i in range(n):
        spec = random.choice(available_hosts)
        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(spec["cpu_speed_gips"] * 1e3),
                memory_mb=int(spec["memory_gb"] * 1024),
                disk_mb=int(spec["disk_tb"] * 1e6),
                bandwidth_mbps=int(spec["bandwidth_gbps"] * 1024),
                power_idle_watt=int(spec["power_idle_watt"]),
                power_peak_watt=int(spec["power_peak_watt"]),
            )
        )
    return hosts


def generate_vms(n: int, max_cores: int, min_cpu_speed_mips: int, max_cpu_speed_mips: int) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    vms = []
    for i in range(n):
        cores = random.randint(1, max_cores)
        cpu_speed = random.randint(min_cpu_speed_mips, max_cpu_speed_mips)
        vms.append(Vm(i, cores, cpu_speed, memory_mb=512, disk_mb=1024, bandwidth_mbps=50, vmm="Xen"))
    return vms


def allocate_vms(vms: list[Vm], hosts: list[Host]) -> list[VmAllocation]:
    """
    Allocate VMs to hosts randomly.
    """

    vm_allocations: list[VmAllocation] = []
    for vm in vms:
        host = random.choice(hosts)
        vm_allocations.append(VmAllocation(vm.id, host.id))
    return vm_allocations
