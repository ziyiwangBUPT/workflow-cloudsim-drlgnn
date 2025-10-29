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
    host1 = [1500, 1000, 1100, 1100, 1900, 2200, 1900, 1100, 1200, 700, 800, 1000, 1400]
    
    # Host 2: 低碳强度区域（水电/核电为主）
    host2 = [700, 800, 900, 700, 1300, 1400, 1300, 1300, 1200, 1200, 900, 900, 900]
    
    # Host 3: 中等碳强度区域（混合能源）
    host3 = [1000, 900, 1100, 1100, 900, 900, 1000, 1000, 1500, 1500, 1500, 1300, 1200]
    
    # Host 4: 高碳强度区域（天然气为主）
    host4 = [1300, 1300, 1500, 1500, 1700, 1900, 2100, 2100, 2100, 1800, 1800, 1700, 1300]
    
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


def get_average_carbon_intensity(host_id: int, start_time: float, end_time: float) -> float:
    """
    计算指定Host在指定时间段的平均碳强度
    
    Args:
        host_id: Host的ID (0-3)
        start_time: 任务开始时间（秒）
        end_time: 任务结束时间（秒）
        
    Returns:
        float: 平均碳强度值
    """
    if host_id < 0 or host_id >= FIXED_NUM_HOSTS:
        raise ValueError(f"Invalid host_id: {host_id}. Must be in range [0, {FIXED_NUM_HOSTS-1}]")
    
    if start_time >= end_time:
        # 如果任务执行时间为0，使用开始时间的碳强度
        return get_carbon_intensity_at_time(host_id, start_time)
    
    # 将时间转换为小时
    start_hour = start_time / 3600.0
    end_hour = end_time / 3600.0
    
    # 计算跨越的小时数
    duration_hours = end_hour - start_hour
    
    # 如果任务在一个小时内完成，使用开始时间的碳强度
    if duration_hours < 1.0:
        return get_carbon_intensity_at_time(host_id, start_time)
    
    # 如果任务跨越多个小时，计算加权平均
    total_weighted_intensity = 0.0
    total_weight = 0.0
    
    # 第一个小时的部分时间
    current_hour = int(start_hour) % 24
    next_hour_time = (int(start_hour) + 1) * 3600
    first_duration = min(next_hour_time - start_time, end_time - start_time)
    total_weighted_intensity += CARBON_INTENSITY_DATA[host_id][current_hour] * first_duration
    total_weight += first_duration
    
    # 中间的完整小时
    current_time = next_hour_time
    while current_time + 3600 <= end_time:
        current_hour = int(current_time / 3600) % 24
        total_weighted_intensity += CARBON_INTENSITY_DATA[host_id][current_hour] * 3600
        total_weight += 3600
        current_time += 3600
    
    # 最后一个小时的部分时间
    if current_time < end_time:
        current_hour = int(current_time / 3600) % 24
        last_duration = end_time - current_time
        total_weighted_intensity += CARBON_INTENSITY_DATA[host_id][current_hour] * last_duration
        total_weight += last_duration
    
    return total_weighted_intensity / total_weight if total_weight > 0 else get_carbon_intensity_at_time(host_id, start_time)


def calculate_carbon_cost(energy_joules: float, host_id: int, 
                          start_time: float, end_time: float) -> float:
    """
    计算碳成本
    
    Args:
        energy_joules: 能耗（Joules）
        host_id: Host的ID
        start_time: 任务开始时间（秒）
        end_time: 任务结束时间（秒）
        
    Returns:
        float: 碳成本（gCO2）
    """
    # 将能耗从 Joules 转换为 kWh
    # 1 kWh = 3,600,000 Joules
    energy_kwh = energy_joules / 3_600_000.0
    
    # 计算平均碳强度（gCO2/kWh）
    avg_carbon_intensity = get_average_carbon_intensity(host_id, start_time, end_time)
    
    # 碳成本 = 能耗(kWh) * 碳强度(gCO2/kWh) = gCO2
    carbon_cost = energy_kwh * avg_carbon_intensity
    
    return carbon_cost

