"""
碳强度数据配置模块

模仿 ecmws-experiments 项目的电价数据结构，
为 4 个数据中心（Host）提供 24 小时碳强度数据。

碳强度的单位：gCO2/kWh（每千瓦时的二氧化碳克数）
碳成本 = 能耗 * 碳强度
"""

import numpy as np


# 硬编码的 4 个 Host 的碳强度数据（24 小时）
# 数据来源：参考 ecmws-experiments/utils/given_data.py 的电价数据结构
# 注意：这里暂时使用与ecmws相同的数值，实际应用时可以调整为真实的碳强度值
def gen_carbon_intensity_data(num_hosts: int = 4) -> list[list[float]]:
    """
    生成碳强度数据（24小时，4个Host）
    
    Returns:
        list[list[float]]: 碳强度数组，shape = (num_hosts, 24)
                           每个Host有24小时的碳强度值
    """
    # 定义特定时间点的碳强度值（参考ecmws的电价数据）
    time_ticks = [int(x.split(':')[0]) for x in
                  ['0:00', '2:00', '6:00', '7:00', '11:00', '12:00', '14:00', '16:00', 
                   '17:00', '19:00', '20:00', '22:00', '23:00']]
    
    # 4个Host的碳强度曲线（可以理解为不同地区的数据中心）
    # Host 1: 高碳强度区域（煤电为主）
    host1 = [0.15, 0.10, 0.11, 0.11, 0.19, 0.22, 0.19, 0.11, 0.12, 0.07, 0.08, 0.10, 0.14]
    
    # Host 2: 低碳强度区域（水电/核电为主）
    host2 = [0.07, 0.08, 0.09, 0.07, 0.13, 0.14, 0.13, 0.13, 0.12, 0.12, 0.09, 0.09, 0.09]
    
    # Host 3: 中等碳强度区域（混合能源）
    host3 = [0.10, 0.09, 0.11, 0.11, 0.09, 0.09, 0.10, 0.10, 0.15, 0.15, 0.15, 0.13, 0.12]
    
    # Host 4: 高碳强度区域（天然气为主）
    host4 = [0.13, 0.13, 0.15, 0.15, 0.17, 0.19, 0.21, 0.21, 0.21, 0.18, 0.18, 0.17, 0.13]
    
    carbon_intensity = [host1, host2, host3, host4]
    
    # 插值填充24小时的完整数据
    new_carbon_intensity = [[0.0 for _ in range(24)] for _ in range(4)]
    for i, tick in enumerate(time_ticks):
        if tick > 0:
            for x in range(time_ticks[i - 1], tick):
                for k in range(4):
                    new_carbon_intensity[k][x] = carbon_intensity[k][i - 1]
        for k in range(4):
            new_carbon_intensity[k][tick] = carbon_intensity[k][i]
    
    # 如果需要超过4个Host，可以随机复制
    if num_hosts > 4:
        import random
        base_patterns = new_carbon_intensity[:4]
        for _ in range(num_hosts - 4):
            new_carbon_intensity.append(random.choice(base_patterns))
    
    return new_carbon_intensity[:num_hosts]


# 全局碳强度数据（4个Host的24小时碳强度）
CARBON_INTENSITY_DATA = gen_carbon_intensity_data(num_hosts=4)

# 固定Host数量为4
FIXED_NUM_HOSTS = 4


def get_carbon_intensity_at_time(host_id: int, time_seconds: float) -> float:
    """
    获取指定Host在指定时间的碳强度值
    
    Args:
        host_id: Host的ID (0-3)
        time_seconds: 时间（秒），相对于工作流到达时间
        
    Returns:
        float: 碳强度值
    """
    if host_id < 0 or host_id >= FIXED_NUM_HOSTS:
        raise ValueError(f"Invalid host_id: {host_id}. Must be in range [0, {FIXED_NUM_HOSTS-1}]")
    
    # 将秒转换为小时，并取模24实现循环
    hour = int(time_seconds // 3600) % 24
    return CARBON_INTENSITY_DATA[host_id][hour]


def get_carbon_intensity_features(host_id: int, start_time: float, end_time: float, 
                                   num_hours_ahead: int = 6) -> list[float]:
    """
    获取指定Host在指定时间段的碳强度特征
    类似于 ecmws 的 flatten_normalized_electricity_prices
    
    Args:
        host_id: Host的ID
        start_time: 任务开始时间（秒）
        end_time: 任务结束时间（秒）
        num_hours_ahead: 向前看的小时数（默认6小时）
        
    Returns:
        list[float]: 碳强度特征向量（已归一化）
    """
    start_hour = int(start_time // 3600) % 24
    
    # 获取当前及未来num_hours_ahead小时的碳强度
    hours = [(start_hour + i) % 24 for i in range(num_hours_ahead)]
    features = [CARBON_INTENSITY_DATA[host_id][h] for h in hours]
    
    # 归一化（类似ecmws的处理方式）
    features_array = np.array(features)
    min_val = features_array.min()
    max_val = features_array.max()
    delta = max_val - min_val
    
    if delta <= 1e-4:
        return features
    else:
        return ((features_array - min_val) / delta).tolist()


def calculate_carbon_cost(energy_consumption: float, host_id: int, 
                          start_time: float, end_time: float) -> float:
    """
    计算碳成本
    
    Args:
        energy_consumption: 能耗（Joules或其他单位）
        host_id: Host的ID
        start_time: 任务开始时间（秒）
        end_time: 任务结束时间（秒）
        
    Returns:
        float: 碳成本
    """
    # 计算平均碳强度（简化处理：使用起始时间的碳强度）
    avg_carbon_intensity = get_carbon_intensity_at_time(host_id, start_time)
    
    # 碳成本 = 能耗 * 碳强度
    carbon_cost = energy_consumption * avg_carbon_intensity
    
    return carbon_cost

