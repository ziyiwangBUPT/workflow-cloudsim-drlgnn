from dataset_generator.gen_dataset import Args as DatasetArgs

TRAINING_DS_ARGS = DatasetArgs(
    host_count=10,
    vm_count=4,
    workflow_count=10,
    gnp_min_n=20,
    gnp_max_n=20,
    max_memory_gb=10,
    min_cpu_speed=500,
    max_cpu_speed=5000,
    min_task_length=500,
    max_task_length=100_000,
    task_arrival="static",
    dag_method="gnp",
)
