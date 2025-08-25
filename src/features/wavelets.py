"""
Wavelet features for multi-resolution analysis and denoising.
Aligned with PRD section 4 - Feature Engineering extensions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import pywt
from scipy import signal
import structlog

log = structlog.get_logger()


class WaveletFeatures:
    """
    Extract wavelet-based features for time series.
    PRD: Wavelets for decomposition/denoise as feature enhancement.
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',  # Daubechies 4 (good for financial data)
        levels: int = 4,  # Decomposition levels
        denoising: bool = True,  # Apply denoising
        feature_mode: str = 'both'  # 'coefficients', 'statistics', 'both'
    ):
        """Initialize wavelet transformer."""
        
        self.wavelet = wavelet
        self.levels = levels
        self.denoising = denoising
        self.feature_mode = feature_mode
        
        # Check if wavelet is valid
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Invalid wavelet: {wavelet}")
            
    def decompose(self, data: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform wavelet decomposition.
        
        Returns:
            Tuple of (approximation, [details])
        """
        
        # Perform multilevel decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=self.levels)
        
        # Split into approximation and details
        approximation = coeffs[0]
        details = coeffs[1:]
        
        return approximation, details
        
    def denoise(
        self,
        data: np.ndarray,
        method: str = 'soft',
        threshold_rule: str = 'sure'
    ) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            data: Input signal
            method: 'soft' or 'hard' thresholding
            threshold_rule: 'sure', 'universal', or numeric threshold
        """
        
        # Decompose
        coeffs = pywt.wavedec(data, self.wavelet, level=self.levels)
        
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # MAD estimator
        
        if threshold_rule == 'universal':
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
        elif threshold_rule == 'sure':
            # SURE threshold (Stein's Unbiased Risk Estimate)
            threshold = sigma * np.sqrt(2 * np.log(len(data) * np.log2(len(data))))
        else:
            threshold = float(threshold_rule)
            
        # Apply thresholding to detail coefficients
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            if method == 'soft':
                coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            else:
                coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='hard')
                
        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        # Handle length mismatch
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
        elif len(denoised) < len(data):
            denoised = np.pad(denoised, (0, len(data) - len(denoised)), mode='edge')
            
        return denoised
        
    def extract_coefficient_features(
        self,
        approximation: np.ndarray,
        details: List[np.ndarray]
    ) -> Dict[str, float]:
        """Extract statistical features from wavelet coefficients."""
        
        features = {}
        
        # Approximation features
        features['wavelet_approx_mean'] = np.mean(approximation)
        features['wavelet_approx_std'] = np.std(approximation)
        features['wavelet_approx_energy'] = np.sum(approximation ** 2)
        features['wavelet_approx_entropy'] = -np.sum(
            approximation ** 2 * np.log(approximation ** 2 + 1e-10)
        )
        
        # Detail features for each level
        for i, detail in enumerate(details):
            level = i + 1
            features[f'wavelet_d{level}_mean'] = np.mean(detail)
            features[f'wavelet_d{level}_std'] = np.std(detail)
            features[f'wavelet_d{level}_energy'] = np.sum(detail ** 2)
            features[f'wavelet_d{level}_max'] = np.max(np.abs(detail))
            
            # Relative energy (important for financial data)
            total_energy = np.sum(approximation ** 2) + sum(np.sum(d ** 2) for d in details)
            features[f'wavelet_d{level}_rel_energy'] = np.sum(detail ** 2) / (total_energy + 1e-10)
            
        return features
        
    def extract_packet_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract features using wavelet packet decomposition.
        More comprehensive than standard DWT.
        """
        
        # Wavelet packet decomposition
        wp = pywt.WaveletPacket(data, self.wavelet, maxlevel=self.levels)
        
        features = {}
        
        # Extract features from each node
        for node in wp.get_level(self.levels, 'natural'):
            node_data = node.data
            if len(node_data) > 0:
                features[f'wp_{node.path}_energy'] = np.sum(node_data ** 2)
                features[f'wp_{node.path}_std'] = np.std(node_data)
                
        return features
        
    def transform(self, data: pd.Series) -> pd.DataFrame:
        """
        Transform time series to wavelet features.
        
        Args:
            data: Input time series
            
        Returns:
            DataFrame with wavelet features
        """
        
        # Convert to numpy
        values = data.values
        
        # Initialize feature dict
        all_features = {}
        
        # Denoise if requested
        if self.denoising:
            denoised = self.denoise(values)
            all_features['wavelet_denoised'] = denoised[-1]  # Last value
            all_features['wavelet_noise_ratio'] = np.std(values - denoised) / (np.std(values) + 1e-10)
        else:
            denoised = values
            
        # Decompose
        approximation, details = self.decompose(denoised)
        
        # Extract features based on mode
        if self.feature_mode in ['coefficients', 'both']:
            # Use actual coefficients as features (subsampled)
            # This is useful for neural networks
            
            # Approximation (heavily subsampled)
            approx_subsample = signal.resample(approximation, min(10, len(approximation)))
            for i, val in enumerate(approx_subsample):
                all_features[f'wavelet_a_coef_{i}'] = val
                
            # Details (only most recent coefficients)
            for level, detail in enumerate(details):
                detail_subsample = detail[-min(5, len(detail)):]  # Last 5 coefficients
                for i, val in enumerate(detail_subsample):
                    all_features[f'wavelet_d{level+1}_coef_{i}'] = val
                    
        if self.feature_mode in ['statistics', 'both']:
            # Statistical features
            coef_features = self.extract_coefficient_features(approximation, details)
            all_features.update(coef_features)
            
        # Create DataFrame
        features_df = pd.DataFrame([all_features], index=[data.index[-1]])
        
        return features_df
        
    def transform_rolling(
        self,
        data: pd.Series,
        window: int = 64,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract wavelet features using rolling windows.
        
        Args:
            data: Input time series
            window: Rolling window size
            min_periods: Minimum periods for valid window
            
        Returns:
            DataFrame with rolling wavelet features
        """
        
        if min_periods is None:
            min_periods = window // 2
            
        features_list = []
        
        for i in range(min_periods, len(data) + 1):
            start_idx = max(0, i - window)
            window_data = data.iloc[start_idx:i]
            
            if len(window_data) >= min_periods:
                features = self.transform(window_data)
                features_list.append(features)
                
        if features_list:
            return pd.concat(features_list, axis=0)
        else:
            return pd.DataFrame()


class MultiResolutionAnalysis:
    """
    Multi-resolution analysis using wavelets for different time scales.
    Useful for capturing patterns at different frequencies.
    """
    
    def __init__(
        self,
        wavelets: List[str] = ['db4', 'sym4', 'coif2'],
        max_level: int = 4
    ):
        """Initialize multi-resolution analyzer."""
        
        self.wavelets = wavelets
        self.max_level = max_level
        
    def analyze(self, data: pd.Series) -> pd.DataFrame:
        """
        Perform multi-resolution analysis.
        
        Returns:
            DataFrame with features from multiple wavelets and scales
        """
        
        all_features = {}
        
        for wavelet_name in self.wavelets:
            # Create wavelet feature extractor
            wf = WaveletFeatures(
                wavelet=wavelet_name,
                levels=self.max_level,
                feature_mode='statistics'
            )
            
            # Extract features
            features = wf.transform(data)
            
            # Prefix with wavelet name
            for col in features.columns:
                all_features[f'{wavelet_name}_{col}'] = features[col].values[0]
                
        return pd.DataFrame([all_features], index=[data.index[-1]])


def add_wavelet_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    wavelet: str = 'db4',
    levels: int = 4,
    window: int = 64,
    denoising: bool = True
) -> pd.DataFrame:
    """
    Convenience function to add wavelet features to dataframe.
    
    Args:
        df: Input dataframe with OHLCV data
        price_col: Column to analyze
        wavelet: Wavelet to use
        levels: Decomposition levels
        window: Rolling window size
        denoising: Whether to include denoising features
        
    Returns:
        DataFrame with original data plus wavelet features
    """
    
    # Initialize wavelet transformer
    wf = WaveletFeatures(
        wavelet=wavelet,
        levels=levels,
        denoising=denoising,
        feature_mode='statistics'
    )
    
    # Extract rolling features
    wavelet_features = wf.transform_rolling(
        df[price_col],
        window=window
    )
    
    # Merge with original data
    df_with_wavelets = df.copy()
    
    # Align indices and merge
    for col in wavelet_features.columns:
        df_with_wavelets[col] = wavelet_features[col]
        
    # Fill NaN values for initial periods
    df_with_wavelets.fillna(method='ffill', inplace=True)
    
    log.info(
        f"Added {len(wavelet_features.columns)} wavelet features "
        f"using {wavelet} with {levels} levels"
    )
    
    return df_with_wavelets


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate sample financial data
    np.random.seed(42)
    n_points = 500
    t = np.linspace(0, 10, n_points)
    
    # Simulate price with trend, cycles, and noise
    trend = 100 + 5 * t
    cycle1 = 10 * np.sin(2 * np.pi * t)
    cycle2 = 5 * np.sin(8 * np.pi * t)
    noise = np.random.randn(n_points) * 2
    
    price = trend + cycle1 + cycle2 + noise
    
    # Create series
    dates = pd.date_range('2024-01-01', periods=n_points, freq='15T')
    price_series = pd.Series(price, index=dates, name='price')
    
    # Apply wavelet analysis
    wf = WaveletFeatures(wavelet='db4', levels=4, denoising=True)
    
    # Denoise
    denoised = wf.denoise(price)
    
    # Extract features
    features = wf.transform_rolling(price_series, window=64)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original vs denoised
    ax = axes[0]
    ax.plot(dates, price, 'b-', alpha=0.5, label='Original')
    ax.plot(dates, denoised, 'r-', linewidth=2, label='Denoised')
    ax.set_title('Wavelet Denoising')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Noise component
    ax = axes[1]
    ax.plot(dates, price - denoised, 'g-', alpha=0.7)
    ax.set_title('Removed Noise')
    ax.grid(True, alpha=0.3)
    
    # Sample features
    ax = axes[2]
    if 'wavelet_approx_energy' in features.columns:
        ax.plot(features.index, features['wavelet_approx_energy'], 'purple', label='Approx Energy')
    if 'wavelet_d1_rel_energy' in features.columns:
        ax2 = ax.twinx()
        ax2.plot(features.index, features['wavelet_d1_rel_energy'], 'orange', label='D1 Rel Energy')
        ax2.set_ylabel('Relative Energy', color='orange')
    ax.set_title('Wavelet Features')
    ax.set_ylabel('Energy', color='purple')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nExtracted {len(features.columns)} wavelet features")
    print("\nSample features:")
    print(features.iloc[-1].head(10))