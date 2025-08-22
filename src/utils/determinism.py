"""
Configurações de determinismo e reprodutibilidade
"""

import os
import random
import numpy as np


def set_deterministic_environment(seed: int = 42):
    """
    Configura ambiente para determinismo completo
    
    Args:
        seed: Seed para todos os geradores aleatórios
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # PyTorch (se disponível)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✅ PyTorch configurado para determinismo (seed={seed})")
    except ImportError:
        pass
    
    # TensorFlow (se disponível)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"✅ TensorFlow configurado para determinismo (seed={seed})")
    except ImportError:
        pass
    
    print(f"✅ Ambiente determinístico configurado (SEED={seed})")
    
    return seed


def verify_determinism():
    """Verifica se o ambiente está determinístico"""
    
    # Teste NumPy
    np.random.seed(42)
    arr1 = np.random.randn(10)
    np.random.seed(42)
    arr2 = np.random.randn(10)
    
    numpy_ok = np.allclose(arr1, arr2)
    
    # Teste Python random
    random.seed(42)
    rand1 = [random.random() for _ in range(10)]
    random.seed(42)
    rand2 = [random.random() for _ in range(10)]
    
    python_ok = rand1 == rand2
    
    results = {
        'numpy': numpy_ok,
        'python': python_ok,
        'environment': os.environ.get('PYTHONHASHSEED') == '0'
    }
    
    # Teste PyTorch se disponível
    try:
        import torch
        torch.manual_seed(42)
        t1 = torch.randn(10)
        torch.manual_seed(42)
        t2 = torch.randn(10)
        results['pytorch'] = torch.allclose(t1, t2)
    except ImportError:
        results['pytorch'] = None
    
    return results