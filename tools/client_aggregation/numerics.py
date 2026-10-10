"""Registered CUDA execution policy; no changes to losses, batches or tolerances."""
import os

DETERMINISTIC = dict(policy='deterministic', cublas_workspace_config=':4096:8',
    deterministic_algorithms=True, warn_only=False, cudnn_benchmark=False,
    cudnn_deterministic=True, tf32='unchanged')


def policy_for(job):
    policy = job.get('settings', {}).get('cuda_policy', 'legacy')
    if policy not in ('legacy', 'deterministic'):
        raise ValueError('Unsupported CUDA numerical policy: '+str(policy))
    return policy


def prepare_environment(policy):
    """Call before importing/initializing CUDA in a fresh training worker."""
    if policy == 'deterministic':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = DETERMINISTIC['cublas_workspace_config']
    elif policy != 'legacy':
        raise ValueError('Unsupported CUDA numerical policy: '+str(policy))


def apply_policy(policy):
    if policy == 'legacy':
        return
    if policy != 'deterministic':
        raise ValueError('Unsupported CUDA numerical policy: '+str(policy))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != DETERMINISTIC['cublas_workspace_config']:
        raise ValueError('Deterministic worker must set CUBLAS_WORKSPACE_CONFIG before CUDA initialization')
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def snapshot():
    import torch
    import platform
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def verify_record(record):
    for key, wanted in DETERMINISTIC.items():
        if key not in ('policy', 'tf32') and record.get(key) != wanted:
            raise ValueError('CUDA numerical policy was not applied: '+key)


def verify_active(policy):
    record = snapshot()
    if policy == 'deterministic':
        verify_record(record)
    elif policy != 'legacy':
        raise ValueError('Unsupported CUDA numerical policy: '+str(policy))
    return record
