"""
Enhanced determinism module for production-grade reproducibility.

References:
- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- Optuna Reproducibility: https://optuna.readthedocs.io/en/stable/faq.html#how-to-make-optuna-training-reproducible
"""

import os
import random
import warnings
from typing import Dict, Optional, Any
import numpy as np
import hashlib
import json


def set_full_determinism(seed: int = 42, verify: bool = True) -> Dict[str, Any]:
    """
    Configure full deterministic environment for ML training.
    
    This function sets all necessary seeds and environment variables
    to ensure reproducible results across runs.
    
    Args:
        seed: Random seed for all generators
        verify: Whether to verify determinism after setting
        
    Returns:
        Dictionary with configuration status and verification results
    """
    results = {
        'seed': seed,
        'status': {},
        'verification': None
    }
    
    # 1. Python random
    random.seed(seed)
    results['status']['python_random'] = 'set'
    
    # 2. NumPy
    np.random.seed(seed)
    results['status']['numpy'] = 'set'
    
    # 3. Hash seed for Python (affects dict ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    results['status']['python_hash'] = 'set'
    
    # 4. PyTorch settings (if available)
    try:
        import torch
        
        # Basic seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic algorithms - THE KEY SETTING
        torch.use_deterministic_algorithms(True, warn_only=False)
        
        # cuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # CUDA settings for determinism
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Thread settings for reproducibility
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        results['status']['pytorch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'deterministic_algorithms': True,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark
        }
        
    except ImportError:
        results['status']['pytorch'] = 'not_available'
    except RuntimeError as e:
        # Some operations might not support deterministic mode
        warnings.warn(f"PyTorch determinism warning: {e}")
        results['status']['pytorch'] = f'partial: {str(e)}'
    
    # 5. TensorFlow settings (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        results['status']['tensorflow'] = tf.__version__
    except ImportError:
        results['status']['tensorflow'] = 'not_available'
    
    # 6. Additional environment variables for MKL/OpenBLAS
    os.environ['MKL_SEED'] = str(seed)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # 7. Verify determinism if requested
    if verify:
        results['verification'] = verify_full_determinism(seed)
    
    return results


def verify_full_determinism(seed: int = 42) -> Dict[str, bool]:
    """
    Comprehensive verification of deterministic behavior.
    
    Args:
        seed: Seed to use for verification
        
    Returns:
        Dictionary with verification results for each component
    """
    results = {}
    
    # 1. Python random verification
    random.seed(seed)
    rand1 = [random.random() for _ in range(100)]
    random.seed(seed)
    rand2 = [random.random() for _ in range(100)]
    results['python_random'] = rand1 == rand2
    
    # 2. NumPy verification
    np.random.seed(seed)
    arr1 = np.random.randn(100, 100)
    np.random.seed(seed)
    arr2 = np.random.randn(100, 100)
    results['numpy'] = np.allclose(arr1, arr2, rtol=1e-10, atol=1e-10)
    
    # 3. Hash seed verification
    results['python_hash'] = os.environ.get('PYTHONHASHSEED') == str(seed)
    
    # 4. PyTorch verification (if available)
    try:
        import torch
        
        # CPU operations
        torch.manual_seed(seed)
        t1_cpu = torch.randn(100, 100)
        torch.manual_seed(seed)
        t2_cpu = torch.randn(100, 100)
        results['pytorch_cpu'] = torch.allclose(t1_cpu, t2_cpu, rtol=1e-10, atol=1e-10)
        
        # CUDA operations (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            t1_cuda = torch.randn(100, 100, device='cuda')
            torch.cuda.manual_seed(seed)
            t2_cuda = torch.randn(100, 100, device='cuda')
            results['pytorch_cuda'] = torch.allclose(t1_cuda, t2_cuda, rtol=1e-10, atol=1e-10)
            
            # Test deterministic algorithm flag
            try:
                # This will raise an error if non-deterministic operations are used
                torch.use_deterministic_algorithms(True)
                # Test a potentially non-deterministic operation
                test_tensor = torch.randn(10, 10, device='cuda')
                _ = torch.nn.functional.interpolate(
                    test_tensor.unsqueeze(0).unsqueeze(0), 
                    size=(20, 20), 
                    mode='bilinear',
                    align_corners=False
                )
                results['pytorch_deterministic_ops'] = True
            except RuntimeError:
                results['pytorch_deterministic_ops'] = False
        else:
            results['pytorch_cuda'] = None
            results['pytorch_deterministic_ops'] = None
            
    except ImportError:
        results['pytorch_cpu'] = None
        results['pytorch_cuda'] = None
        results['pytorch_deterministic_ops'] = None
    
    # 5. Environment variables verification
    env_vars = {
        'PYTHONHASHSEED': str(seed),
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1'
    }
    
    results['environment'] = all(
        os.environ.get(key) == value 
        for key, value in env_vars.items()
    )
    
    return results


def get_reproducibility_hash(config: Dict[str, Any]) -> str:
    """
    Generate a hash for configuration to ensure reproducibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA256 hash of the configuration
    """
    # Sort keys for consistent ordering
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


def assert_determinism(results: Dict[str, Any], raise_on_fail: bool = True):
    """
    Assert that determinism is properly configured.
    
    Args:
        results: Results from verify_full_determinism
        raise_on_fail: Whether to raise exception on failure
        
    Raises:
        RuntimeError: If determinism check fails and raise_on_fail=True
    """
    failures = []
    
    if 'verification' in results and results['verification']:
        for component, is_deterministic in results['verification'].items():
            if is_deterministic is False:
                failures.append(component)
    
    if failures:
        msg = f"Determinism check failed for: {', '.join(failures)}"
        if raise_on_fail:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)
    
    return len(failures) == 0


class DeterministicContext:
    """Context manager for deterministic execution."""
    
    def __init__(self, seed: int = 42, verify: bool = True):
        self.seed = seed
        self.verify = verify
        self.original_state = {}
        
    def __enter__(self):
        # Save original state
        self.original_state['python_random'] = random.getstate()
        self.original_state['numpy'] = np.random.get_state()
        self.original_state['env'] = dict(os.environ)
        
        try:
            import torch
            self.original_state['torch_rng'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.original_state['torch_cuda_rng'] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        
        # Set deterministic state
        self.results = set_full_determinism(self.seed, self.verify)
        return self.results
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        random.setstate(self.original_state['python_random'])
        np.random.set_state(self.original_state['numpy'])
        
        # Restore environment variables
        for key in ['PYTHONHASHSEED', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
            if key in self.original_state['env']:
                os.environ[key] = self.original_state['env'][key]
            elif key in os.environ:
                del os.environ[key]
        
        try:
            import torch
            torch.set_rng_state(self.original_state['torch_rng'])
            if torch.cuda.is_available() and 'torch_cuda_rng' in self.original_state:
                torch.cuda.set_rng_state_all(self.original_state['torch_cuda_rng'])
        except ImportError:
            pass


# Convenience function for backward compatibility
def set_deterministic_environment(seed: int = 42) -> int:
    """Backward compatible function name."""
    results = set_full_determinism(seed, verify=False)
    return seed


if __name__ == "__main__":
    # Test determinism configuration
    print("Setting up full deterministic environment...")
    results = set_full_determinism(seed=42, verify=True)
    
    print("\nConfiguration Status:")
    print(json.dumps(results['status'], indent=2))
    
    print("\nVerification Results:")
    if results['verification']:
        for component, is_ok in results['verification'].items():
            status = "✅" if is_ok else "❌" if is_ok is False else "⚠️"
            print(f"  {component}: {status}")
    
    # Test assertion
    try:
        assert_determinism(results, raise_on_fail=True)
        print("\n✅ All determinism checks passed!")
    except RuntimeError as e:
        print(f"\n❌ Determinism check failed: {e}")