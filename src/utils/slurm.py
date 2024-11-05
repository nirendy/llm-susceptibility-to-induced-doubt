import submitit

def submit_job(
    func,
    *args,
    gpu_type,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
    slurm_gpus_per_node=1,
    slurm_nodelist=None,
):
    # Map GPU type and account type to partition and account options based on `sinfo` data
    partition_account_map = {
        'geforce_rtx_3090': {'partition': 'killable', 'account': 'gpu-students'},
        'v100': {'partition': 'killable', 'account': 'gpu-students'},
        'a5000': {'partition': 'killable', 'account': 'gpu-students'},
        'a6000': {'partition': 'killable', 'account': 'gpu-research'},
        'l40s': {'partition': 'killable', 'account': 'gpu-research'},
        'a100': {'partition': 'gpu-a100-killable', 'account': 'gpu-research'},
        'h100': {'partition': 'gpu-h100-killable', 'account': 'gpu-research'},
        'titan_xp-studentrun': {'partition': 'studentrun', 'account': 'gpu-students', 'nodelist': 's-005',},
        'titan_xp-studentbatch': {'partition': 'studentbatch', 'account': 'gpu-students', 'nodelist': 's-005',},
        'titan_xp-studentkillable': {'partition': 'studentkillable', 'account': 'gpu-students', 'nodelist': 's-005',}
    }

    # Determine the appropriate partition and account based on `gpu_type`
    partition_account = partition_account_map[gpu_type]
    slurm_partition = partition_account['partition']
    slurm_account = partition_account['account']
    slurm_nodelist = slurm_nodelist or partition_account.get('nodelist', slurm_nodelist)
    
    def ommit_none(d):
        return {k: v for k, v in d.items() if v is not None}
    # Setup the executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        slurm_nodes=slurm_nodes,
        tasks_per_node=tasks_per_node,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_gpus_per_node=slurm_gpus_per_node,
        slurm_mem=memory_required,
        slurm_constraint=gpu_type.split('-')[0] if gpu_type else None,
        **ommit_none(dict(
            slurm_nodelist=slurm_nodelist,
        ))
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job
