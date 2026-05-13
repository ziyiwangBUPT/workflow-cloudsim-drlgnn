#!/usr/bin/env python3
"""
独立评估脚本
用于评估已训练好的模型，无需重新训练
"""

import time
from dataclasses import dataclass
from pathlib import Path
import tyro

# 导入评估相关
from scheduler.viz_results.evaluate import main as run_evaluation, Args as EvalArgs, EvaluationSetting
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.config.settings import ALGORITHMS, MIN_TESTING_DS_SEED


@dataclass
class EvalModelArgs:
    model_path: str = "logs/1762270027_test"
    """已训练模型的路径或run_name（例如: logs/1761733452_quick_test 或 logs/1761733452_quick_test/model.pt）"""
    
    simulator_jar_path: str = "simulator/target/cloudsim-simulator-1.0-SNAPSHOT.jar"
    """Path to the CloudSim simulator JAR file"""
    
    # 评估参数（与quick_train_and_eval.py保持一致）
    num_eval_samples: int = 5
    """Number of evaluation samples per setting"""
    
    eval_host_count: int = 4
    """Number of hosts for evaluation datasets"""
    
    eval_vm_count: int = 5
    """Number of VMs for evaluation datasets"""
    
    eval_workflow_count: int = 50
    """Number of workflows for evaluation datasets"""
    
    eval_task_min: int = 10
    """Minimum tasks per workflow in evaluation"""
    
    eval_task_max: int = 20
    """Maximum tasks per workflow in evaluation"""
    
    use_multiple_eval_settings: bool = False
    """If True, use multiple evaluation settings with different scales"""
    
    include_baseline_algorithms: bool = True
    """If True, compare with baseline algorithms (Random, RR, Min-Min, Max-Min, HEFT)"""
    
    output_csv: str | None = None
    """Optional: Custom output CSV path. If None, will save to model directory as evaluation_results.csv"""


def find_model_file(model_path: str) -> tuple[Path, str]:
    """查找模型文件并返回目录和run_name"""
    path = Path(model_path)
    
    # 如果是目录，查找其中的model.pt
    if path.is_dir():
        model_file = path / "model.pt"
        if model_file.exists():
            return path, path.name
        # 尝试查找检查点文件
        checkpoint_files = list(path.glob("model_*.pt"))
        if checkpoint_files:
            model_file = sorted(checkpoint_files, key=lambda x: int(x.stem.split("_")[1]))[-1]
            return path, path.name
        raise FileNotFoundError(f"在目录 {path} 中未找到模型文件")
    
    # 如果是文件路径
    if path.is_file():
        return path.parent, path.parent.name
    
    # 尝试在logs目录中查找
    logs_path = Path("logs") / path.name
    if logs_path.exists() and logs_path.is_dir():
        model_file = logs_path / "model.pt"
        if model_file.exists():
            return logs_path, logs_path.name
    
    raise FileNotFoundError(f"找不到模型路径: {model_path}")


def build_algorithm_list(model_identifier: str, include_baseline: bool) -> list[tuple[str, str]]:
    """构建算法对比列表，包含训练好的模型和基准算法"""
    algorithms = []
    
    # 添加训练好的模型
    algorithms.append(("My_New_Model", model_identifier))
    
    # 添加基准算法进行对比
    if include_baseline:
        algorithms.extend([
            ("Random", "random"),
            ("Round_Robin", "round_robin"),
            ("Min-Min", "min_min"),
            ("Max-Min", "max_min"),
            ("HEFT", "insertion_heft"),
        ])
    
    return algorithms


def create_evaluation_args(
    args: EvalModelArgs,
    model_identifier: str,
    csv_output_path: str
) -> EvalArgs:
    """创建评估配置"""
    
    # 创建评估设置
    eval_settings = []
    
    # 基础评估设置
    eval_settings.append(
        EvaluationSetting(
            id=1,
            dataset_args=DatasetArgs(
                host_count=args.eval_host_count,
                vm_count=args.eval_vm_count,
                workflow_count=args.eval_workflow_count,
                gnp_min_n=args.eval_task_min,
                gnp_max_n=args.eval_task_max,
                max_memory_gb=10,
                min_cpu_speed=100,
                max_cpu_speed=5000,
                min_task_length=50000,
                max_task_length=100_000,
                task_arrival="static",
                dag_method="gnp",
            ),
        )
    )
    
    # 如果启用多个评估设置
    if args.use_multiple_eval_settings:
        # 小规模设置
        eval_settings.append(
            EvaluationSetting(
                id=2,
                dataset_args=DatasetArgs(
                    host_count=4,
                    vm_count=5,
                    workflow_count=30,
                    gnp_min_n=5,
                    gnp_max_n=10,
                    max_memory_gb=10,
                    min_cpu_speed=100,
                    max_cpu_speed=5000,
                    min_task_length=50000,
                    max_task_length=100_000,
                    task_arrival="static",
                    dag_method="gnp",
                ),
            )
        )
        
        # 大规模设置
        eval_settings.append(
            EvaluationSetting(
                id=3,
                dataset_args=DatasetArgs(
                    host_count=4,
                    vm_count=8,
                    workflow_count=80,
                    gnp_min_n=15,
                    gnp_max_n=25,
                    max_memory_gb=10,
                    min_cpu_speed=10,
                    max_cpu_speed=500,
                    min_task_length=50000,
                    max_task_length=100_000,
                    task_arrival="static",
                    dag_method="gnp",
                ),
            )
        )
    
    return EvalArgs(
        simulator=args.simulator_jar_path,
        export_csv=csv_output_path,
        num_samples_per_setting=args.num_eval_samples,
        settings=eval_settings,
    )


def main(args: EvalModelArgs):
    """主函数：评估已训练好的模型"""
    
    print("=" * 80)
    print("模型评估脚本")
    print("=" * 80)
    
    # ========================================
    # 第一步：查找模型文件
    # ========================================
    print("\n[步骤 1/3] 查找模型文件...")
    
    try:
        output_dir, run_name = find_model_file(args.model_path)
        print(f"  ✓ 找到模型目录: {output_dir}")
        print(f"  ✓ Run名称: {run_name}")
        
        # 查找模型文件
        model_file = output_dir / "model.pt"
        if not model_file.exists():
            checkpoint_files = list(output_dir.glob("model_*.pt"))
            if checkpoint_files:
                model_file = sorted(checkpoint_files, key=lambda x: int(x.stem.split("_")[1]))[-1]
                print(f"  ✓ 使用检查点模型: {model_file.name}")
            else:
                raise FileNotFoundError(f"在目录 {output_dir} 中未找到模型文件")
        else:
            print(f"  ✓ 模型文件: {model_file.name}")
    
    except FileNotFoundError as e:
        print(f"  ✗ 错误: {e}")
        print("\n提示: 请提供以下之一:")
        print("  1. 模型目录路径，例如: logs/1761733452_quick_test")
        print("  2. 模型文件路径，例如: logs/1761733452_quick_test/model.pt")
        print("  3. Run名称，例如: 1761733452_quick_test")
        return
    
    # ========================================
    # 第二步：准备评估配置
    # ========================================
    print("\n[步骤 2/3] 准备评估配置...")
    
    # 构建模型标识符 (格式: gin:{run_name}:{model_filename})
    model_identifier = f"gin:{run_name}:{model_file.name}"
    print(f"  - 模型标识符: {model_identifier}")
    
    # 构建算法列表
    custom_algorithms = build_algorithm_list(model_identifier, args.include_baseline_algorithms)
    
    print(f"  - 对比算法数量: {len(custom_algorithms)}")
    for algo_name, algo_id in custom_algorithms:
        print(f"    · {algo_name}")
    
    # CSV 输出路径
    if args.output_csv:
        csv_output = Path(args.output_csv)
    else:
        csv_output = output_dir / "evaluation_results.csv"
    
    print(f"  - 结果输出路径: {csv_output}")
    
    # 创建评估参数
    eval_args = create_evaluation_args(args, model_identifier, str(csv_output))
    
    # ========================================
    # 第三步：运行评估
    # ========================================
    print("\n[步骤 3/3] 开始评估对比...")
    print(f"  - 评估设置数: {len(eval_args.settings)}")
    print(f"  - 每设置样本数: {args.num_eval_samples}")
    print(f"  - 总评估次数: {len(eval_args.settings) * args.num_eval_samples * len(custom_algorithms)}")
    
    evaluation_start_time = time.time()
    
    # 动态替换 ALGORITHMS
    import scheduler.config.settings as settings_module
    import scheduler.viz_results.evaluate as evaluate_module
    
    # 保存原始 ALGORITHMS
    original_algorithms = ALGORITHMS.copy()
    
    # 临时修改全局 ALGORITHMS
    settings_module.ALGORITHMS = custom_algorithms
    evaluate_module.ALGORITHMS = custom_algorithms
    
    try:
        # 运行评估
        run_evaluation(eval_args)
        
        evaluation_duration = time.time() - evaluation_start_time
        print(f"\n✓ 评估完成! 用时: {evaluation_duration / 60:.2f} 分钟")
        print(f"  - 结果已保存: {csv_output}")
        
    finally:
        # 恢复原始 ALGORITHMS
        settings_module.ALGORITHMS = original_algorithms
        evaluate_module.ALGORITHMS = original_algorithms
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - 模型文件: {model_file.name}")
    print(f"  - 评估结果: {csv_output.name}")
    
    print("\n📊 查看评估结果:")
    print(f"  python analyze_results.py --csv-file {csv_output}")
    print("\n或者手动查看:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{csv_output}')")
    print(f"  print(df.groupby('Algorithm')[['Makespan', 'CarbonEmission_gCO2', 'Time']].mean())")
    print("=" * 80)
    
    # 可选：快速展示结果预览
    try:
        import pandas as pd
        df = pd.read_csv(csv_output)
        print("\n📊 评估结果预览（平均值）:")
        print("-" * 80)
        # 检查列名（兼容新旧格式）
        preview_cols = ['Makespan', 'Time']
        if 'CarbonEmission_gCO2' in df.columns:
            preview_cols.insert(1, 'CarbonEmission_gCO2')
            carbon_label = "碳排放(gCO2)"
        elif 'EnergyJ' in df.columns:
            preview_cols.insert(1, 'EnergyJ')
            carbon_label = "能耗(焦耳)"
        else:
            carbon_label = None
        
        summary = df.groupby('Algorithm')[preview_cols].mean()
        summary = summary.sort_values('Makespan')
        print(summary.to_string())
        print("-" * 80)
        print("\n指标说明:")
        print("  - Makespan: 完工时间 (越小越好)")
        if carbon_label:
            print(f"  - {carbon_label}: {'总碳排放' if 'Carbon' in carbon_label else '总能耗'} (越小越好)")
        print("  - Time: 调度程序执行时间/秒 (越小越好)")
        print()
    except Exception as e:
        print(f"\n注意: 无法自动展示结果预览: {e}")
        print(f"请使用 analyze_results.py 脚本查看详细结果")


if __name__ == "__main__":
    # 使用 tyro 解析命令行参数
    args = tyro.cli(EvalModelArgs)
    
    # 运行评估
    main(args)

