import json

import numpy as np

from scheduler.config.settings import HOST_SPECS_PATH
from scheduler.config.carbon_intensity import FIXED_NUM_HOSTS, CARBON_INTENSITY_DATA
from scheduler.dataset_generator.core.models import Host, Vm


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(n: int, rng: np.random.RandomState) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    
    注意：为了支持碳强度特征，现在强制生成固定数量（4个）的Host。
    每个Host对应一个碳强度曲线。
    
    Args:
        n: 请求的Host数量（将被忽略，始终生成4个Host）
        rng: 随机数生成器
        
    Returns:
        list[Host]: 固定4个Host的列表
    """
    # 强制使用固定数量的Host
    actual_n = FIXED_NUM_HOSTS
    
    if n != actual_n:
        print(f"[警告] 请求生成 {n} 个Host，但为支持碳强度特征，强制生成 {actual_n} 个Host")
    
    with open(HOST_SPECS_PATH) as f:
        available_hosts: list = json.load(f)

    hosts: list[Host] = []
    for i in range(actual_n):
        # 为每个Host选择规格（可以使用不同的规格）
        spec = available_hosts[rng.randint(0, len(available_hosts))]
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
                carbon_intensity_curve=CARBON_INTENSITY_DATA[i],  # 新增：分配碳强度曲线
            )
        )
    return hosts


# Generating and Allocating VMs
# ----------------------------------------------------------------------------------------------------------------------


def generate_vms(
    n: int, max_memory_gb: int, min_cpu_speed_mips: int, max_cpu_speed_mips: int, rng: np.random.RandomState
) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    vms = []
    for i in range(n):
        ram_mb = rng.randint(1, max_memory_gb + 1) * 1024
        cpu_speed = rng.randint(min_cpu_speed_mips, max_cpu_speed_mips + 1)
        host_id = -1  # Unallocated
        vms.append(Vm(i, host_id, cpu_speed, memory_mb=ram_mb, disk_mb=1024, bandwidth_mbps=50, vmm="Xen"))
    return vms


def allocate_vms(vms: list[Vm], hosts: list[Host], rng: np.random.RandomState):
    """
    Allocate VMs to hosts randomly.
    """

    for vm in vms:
        host = hosts[rng.randint(0, len(hosts))]
        vm.host_id = host.id
