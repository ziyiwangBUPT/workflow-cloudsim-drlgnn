"""
生成示例数据用于测试可视化脚本
当你还没有运行完整实验时，可以使用此脚本生成模拟数据来测试plot_results.py
"""
import pandas as pd
import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    output_csv: str = "sample_experiment_results.csv"
    """输出CSV文件路径"""
    num_samples: int = 10
    """每个设置的样本数量"""


def generate_sample_data(args: Args):
    """
    生成模拟的实验数据
    """
    np.random.seed(42)
    
    algorithms = ["Random", "Round Robin", "Min-Min", "Max-Min", "HEFT", "Proposed"]
    results = []
    
    # 实验1: 工作流数量
    workflow_counts = [10, 15, 20, 25]
    for workflow_count in workflow_counts:
        for seed_id in range(args.num_samples):
            for algorithm in algorithms:
                # 模拟碳排放数据 - Proposed算法表现最好
                if algorithm == "Proposed":
                    base_emission = 15000 + workflow_count * 800
                    noise = np.random.normal(0, 500)
                elif algorithm == "HEFT":
                    base_emission = 18000 + workflow_count * 900
                    noise = np.random.normal(0, 600)
                elif algorithm == "Min-Min":
                    base_emission = 20000 + workflow_count * 950
                    noise = np.random.normal(0, 700)
                elif algorithm == "Max-Min":
                    base_emission = 21000 + workflow_count * 1000
                    noise = np.random.normal(0, 800)
                elif algorithm == "Round Robin":
                    base_emission = 23000 + workflow_count * 1050
                    noise = np.random.normal(0, 900)
                else:  # Random
                    base_emission = 25000 + workflow_count * 1100
                    noise = np.random.normal(0, 1000)
                
                carbon_emission = base_emission + noise
                
                results.append({
                    "SeedId": seed_id,
                    "SettingId": workflow_count,
                    "ExperimentType": "workflow_count",
                    "XValue": workflow_count,
                    "Rho": 0.2,
                    "Algorithm": algorithm,
                    "Makespan": carbon_emission * 0.05 + np.random.normal(0, 100),
                    "CarbonEmission_gCO2": carbon_emission,
                    "Time": np.random.uniform(0.1, 5.0),
                    "WorkflowDeadlineSatisfactionRate": np.random.uniform(0.7, 1.0),
                })
    
    # 实验2: 任务数量
    task_counts = [150, 225, 300, 375]
    for task_count in task_counts:
        for seed_id in range(args.num_samples):
            for algorithm in algorithms:
                # 模拟碳排放数据
                if algorithm == "Proposed":
                    base_emission = 10000 + task_count * 70
                    noise = np.random.normal(0, 600)
                elif algorithm == "HEFT":
                    base_emission = 12000 + task_count * 80
                    noise = np.random.normal(0, 700)
                elif algorithm == "Min-Min":
                    base_emission = 14000 + task_count * 85
                    noise = np.random.normal(0, 800)
                elif algorithm == "Max-Min":
                    base_emission = 15000 + task_count * 90
                    noise = np.random.normal(0, 900)
                elif algorithm == "Round Robin":
                    base_emission = 17000 + task_count * 95
                    noise = np.random.normal(0, 1000)
                else:  # Random
                    base_emission = 19000 + task_count * 100
                    noise = np.random.normal(0, 1100)
                
                carbon_emission = base_emission + noise
                
                results.append({
                    "SeedId": seed_id,
                    "SettingId": 100 + task_count,
                    "ExperimentType": "task_count",
                    "XValue": task_count,
                    "Rho": 0.2,
                    "Algorithm": algorithm,
                    "Makespan": carbon_emission * 0.05 + np.random.normal(0, 100),
                    "CarbonEmission_gCO2": carbon_emission,
                    "Time": np.random.uniform(0.1, 5.0),
                    "WorkflowDeadlineSatisfactionRate": np.random.uniform(0.7, 1.0),
                })
    
    # 实验3: rho值
    rho_values = [0.2, 0.4, 0.6, 0.8]
    for rho in rho_values:
        for seed_id in range(args.num_samples):
            for algorithm in algorithms:
                # 模拟碳排放数据 - rho越大，碳排放可能略有变化
                rho_factor = 1.0 + (rho - 0.2) * 0.1
                if algorithm == "Proposed":
                    base_emission = 20000 * rho_factor
                    noise = np.random.normal(0, 800)
                elif algorithm == "HEFT":
                    base_emission = 22000 * rho_factor
                    noise = np.random.normal(0, 900)
                elif algorithm == "Min-Min":
                    base_emission = 23000 * rho_factor
                    noise = np.random.normal(0, 1000)
                elif algorithm == "Max-Min":
                    base_emission = 24000 * rho_factor
                    noise = np.random.normal(0, 1100)
                elif algorithm == "Round Robin":
                    base_emission = 25000 * rho_factor
                    noise = np.random.normal(0, 1200)
                else:  # Random
                    base_emission = 26000 * rho_factor
                    noise = np.random.normal(0, 1300)
                
                carbon_emission = base_emission + noise
                
                results.append({
                    "SeedId": seed_id,
                    "SettingId": 1000 + int(rho * 10),
                    "ExperimentType": "rho",
                    "XValue": rho,
                    "Rho": rho,
                    "Algorithm": algorithm,
                    "Makespan": carbon_emission * 0.05 + np.random.normal(0, 100),
                    "CarbonEmission_gCO2": carbon_emission,
                    "Time": np.random.uniform(0.1, 5.0),
                    "WorkflowDeadlineSatisfactionRate": np.random.uniform(0.7, 1.0),
                })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 计算RPD（相对性能偏差）
    # 对每个实验实例（SeedId, ExperimentType, XValue），找到最优碳排放值
    print("Calculating RPD (Relative Performance Deviation)...")
    
    # 第一步：找到每个实例的最优（最小）碳排放
    instance_best_carbon = {}
    for _, row in df.iterrows():
        instance_key = (row['SeedId'], row['ExperimentType'], row['XValue'])
        carbon = row['CarbonEmission_gCO2']
        
        if instance_key not in instance_best_carbon:
            instance_best_carbon[instance_key] = float('inf')
        
        instance_best_carbon[instance_key] = min(instance_best_carbon[instance_key], carbon)
    
    # 第二步：为每个数据点计算RPD
    rpd_list = []
    for _, row in df.iterrows():
        instance_key = (row['SeedId'], row['ExperimentType'], row['XValue'])
        best_carbon = instance_best_carbon[instance_key]
        carbon = row['CarbonEmission_gCO2']
        
        # RPD = (C_algo - C_min) / C_min * 100
        if best_carbon > 0:
            rpd = ((carbon - best_carbon) / best_carbon) * 100
        else:
            rpd = 0.0
        
        rpd_list.append(rpd)
    
    df['RPD'] = rpd_list
    
    # 保存到CSV
    df.to_csv(args.output_csv, index=False)
    
    print(f"✓ Sample data generated successfully!")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Saved to: {args.output_csv}")
    print(f"\nData summary:")
    print(f"  - Experiment types: {df['ExperimentType'].unique().tolist()}")
    print(f"  - Algorithms: {df['Algorithm'].unique().tolist()}")
    print(f"  - Samples per setting: {args.num_samples}")
    print(f"\nYou can now test the visualization with:")
    print(f"  python plot_results.py --input-csv {args.output_csv} --output-image sample_results.png")


if __name__ == "__main__":
    generate_sample_data(tyro.cli(Args))

