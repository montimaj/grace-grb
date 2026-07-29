"""
Model wrappers for the BASIN-SCALE pipeline.

The recurrent models (LSTM, BiLSTM, BiLSTM+Attention) exist for the basin-scale
model comparison only. The gridded 0.1 degree pipeline uses tree ensembles
exclusively: its training set is ~19 independent mascons, far too few spatial
degrees of freedom to fit a recurrent network without memorising them, and its
target is monthly rather than a daily sequence.


ML Model Wrapper Classes for TWS Downscaling
Supports: BiLSTM+Attention, BiLSTM, LSTM, XGBoost, LightGBM, RandomForest
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List

# Optional imports for gradient boosting
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def set_seed(seed: int = 20) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# PyTorch Neural Network Architectures
# =============================================================================

class Attention(nn.Module):
    """Attention mechanism for sequence models."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


class BiLSTMAttentionNet(nn.Module):
    """BiLSTM with Attention mechanism."""
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bilstm_out, _ = self.bilstm(x)
        context, weights = self.attention(bilstm_out)
        self.attn_weights = weights
        return self.fc(context)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM network."""
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMNet(nn.Module):
    """Standard LSTM network."""
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# =============================================================================
# Abstract Base Model Wrapper
# =============================================================================

class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""
    
    def __init__(self, name: str, seed: int = 20):
        self.name = name
        self.seed = seed
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_fitted = False
        self.train_losses = []
        set_seed(seed)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModelWrapper':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'name': self.name, 'seed': self.seed}


# =============================================================================
# PyTorch-based Model Wrappers
# =============================================================================

class PyTorchModelWrapper(BaseModelWrapper):
    """Base wrapper for PyTorch models."""
    
    def __init__(self, name: str, network_class: type, seed: int = 20,
                 hidden_size: int = 64, learning_rate: float = 0.001,
                 epochs: int = 30, batch_size: int = 64, seq_length: int = 7):
        super().__init__(name, seed)
        self.network_class = network_class
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape a flat lagged design matrix into (n_samples, seq_length, n_features).

        `utils.prepare_features_target` emits columns FEATURE-MAJOR:

            SMS_lag1..SMS_lag7, ET_lag1..ET_lag7, rainfall_lag1..., ...

        so the flat row is [f0t0..f0t6, f1t0..f1t6, ...]. A direct
        `reshape(n, seq_length, n_features)` reads that as TIME-major and puts
        predictor f at lag t where the model expects lag t of predictor f --
        scrambling the time axis the recurrent layers exist to exploit. Reshape
        feature-major first, then transpose, so [i, t, f] really is predictor f
        at lag t+1.
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, self.n_features, self.seq_length).transpose(0, 2, 1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True, **kwargs) -> 'PyTorchModelWrapper':
        """Fit the PyTorch model."""
        set_seed(self.seed)
        
        # Infer number of features
        self.n_features = X.shape[1] // self.seq_length
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = self.network_class(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            output_size=1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss()
        
        # Training loop
        self.train_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            if verbose:
                print(f"[{self.name}] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        X_scaled = self.scaler_X.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        return self.scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    def get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Feature importance not directly available for neural networks."""
        return None
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights if model supports it."""
        if hasattr(self.model, 'attn_weights') and self.model.attn_weights is not None:
            return self.model.attn_weights.cpu().numpy()
        return None


class BiLSTMAttentionWrapper(PyTorchModelWrapper):
    """Wrapper for BiLSTM with Attention model."""
    
    def __init__(self, seed: int = 20, **kwargs):
        super().__init__(
            name="BiLSTM+Attention",
            network_class=BiLSTMAttentionNet,
            seed=seed,
            **kwargs
        )


class BiLSTMWrapper(PyTorchModelWrapper):
    """Wrapper for BiLSTM model."""
    
    def __init__(self, seed: int = 20, **kwargs):
        super().__init__(
            name="BiLSTM",
            network_class=BiLSTMNet,
            seed=seed,
            **kwargs
        )


class LSTMWrapper(PyTorchModelWrapper):
    """Wrapper for LSTM model."""
    
    def __init__(self, seed: int = 20, **kwargs):
        super().__init__(
            name="LSTM",
            network_class=LSTMNet,
            seed=seed,
            **kwargs
        )


# =============================================================================
# Tree-based Model Wrappers
# =============================================================================

class TreeModelWrapper(BaseModelWrapper):
    """Base wrapper for tree-based models (no sequence reshaping needed)."""
    
    def __init__(self, name: str, seed: int = 20):
        super().__init__(name, seed)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True, **kwargs) -> 'TreeModelWrapper':
        """Fit the tree-based model."""
        X_scaled = self.scaler_X.fit_transform(X)
        y_flat = y.flatten()
        
        if verbose:
            print(f"[{self.name}] Training on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.model.fit(X_scaled, y_flat)
        self.is_fitted = True
        
        if verbose:
            print(f"[{self.name}] Training complete.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from tree model."""
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))


class XGBoostWrapper(TreeModelWrapper):
    """Wrapper for XGBoost model."""
    
    def __init__(self, seed: int = 20, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        super().__init__(name="XGBoost", seed=seed)
        
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=-1,
            **kwargs
        )


class LightGBMWrapper(TreeModelWrapper):
    """Wrapper for LightGBM model."""
    
    def __init__(self, seed: int = 20, n_estimators: int = 100,
                 max_depth: int = -1, learning_rate: float = 0.1,
                 num_leaves: int = 31, **kwargs):
        super().__init__(name="LightGBM", seed=seed)
        
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **kwargs
        )


class RandomForestWrapper(TreeModelWrapper):
    """Wrapper for Random Forest model."""
    
    def __init__(self, seed: int = 20, n_estimators: int = 100,
                 max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, **kwargs):
        super().__init__(name="RandomForest", seed=seed)
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
            **kwargs
        )


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name: str, seed: int = 20, **kwargs) -> BaseModelWrapper:
    """
    Factory function to create model wrappers.
    
    Parameters
    ----------
    model_name : str
        Name of the model. Options: 'bilstm_attention', 'bilstm', 'lstm',
        'xgboost', 'lightgbm', 'randomforest'
    seed : int
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments passed to the model constructor
    
    Returns
    -------
    BaseModelWrapper
        Initialized model wrapper
    """
    model_map = {
        'bilstm_attention': BiLSTMAttentionWrapper,
        'bilstm+attention': BiLSTMAttentionWrapper,
        'bilstm': BiLSTMWrapper,
        'lstm': LSTMWrapper,
        'xgboost': XGBoostWrapper,
        'xgb': XGBoostWrapper,
        'lightgbm': LightGBMWrapper,
        'lgbm': LightGBMWrapper,
        'randomforest': RandomForestWrapper,
        'rf': RandomForestWrapper,
        'random_forest': RandomForestWrapper,
    }
    
    model_name_lower = model_name.lower().replace(' ', '_').replace('-', '_')
    
    if model_name_lower not in model_map:
        available = list(set(model_map.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    return model_map[model_name_lower](seed=seed, **kwargs)


def get_all_models(seed: int = 20, **kwargs) -> Dict[str, BaseModelWrapper]:
    """
    Get all available models.
    
    Returns
    -------
    Dict[str, BaseModelWrapper]
        Dictionary mapping model names to initialized wrappers
    """
    models = {
        'BiLSTM+Attention': BiLSTMAttentionWrapper(seed=seed, **kwargs.get('bilstm_attention', {})),
        'BiLSTM': BiLSTMWrapper(seed=seed, **kwargs.get('bilstm', {})),
        'LSTM': LSTMWrapper(seed=seed, **kwargs.get('lstm', {})),
        'RandomForest': RandomForestWrapper(seed=seed, **kwargs.get('randomforest', {})),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBoostWrapper(seed=seed, **kwargs.get('xgboost', {}))
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = LightGBMWrapper(seed=seed, **kwargs.get('lightgbm', {}))
    
    return models
