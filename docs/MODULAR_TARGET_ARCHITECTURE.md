# Modular Target Generation Architecture

## Overview

The represent package uses a **pluggable target generation system** that allows easy addition of new labeling logic for both classification and regression tasks. This architecture completely separates target generation from data processing, making it trivial to add new target types.

## Core Design Principles

1. **Single Responsibility**: Each target generator handles one specific labeling approach
2. **Pluggable Interface**: All generators implement the same interface
3. **Type Safety**: Clear separation between classification and regression targets
4. **Composability**: Multiple target generators can be combined in a single dataset
5. **Extensibility**: Adding new labeling logic requires only implementing one interface

## Architecture Components

### 1. Target Generator Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import polars as pl
import numpy as np

class TargetGenerator(ABC):
    """Base interface for all target generators."""
    
    @abstractmethod
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Generate target arrays from market data.
        
        Args:
            df: Market data DataFrame with required columns
            
        Returns:
            Dict mapping target names to numpy arrays
        """
        pass
    
    @abstractmethod
    def get_target_info(self) -> Dict[str, Any]:
        """Return metadata about the targets this generator creates."""
        pass
    
    @property
    @abstractmethod
    def target_type(self) -> str:
        """Return 'classification' or 'regression'."""
        pass
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Return list of required DataFrame columns."""
        pass
```

### 2. Classification Target Generators

#### Quantile-Based Classification
```python
class QuantileClassificationGenerator(TargetGenerator):
    """Generates uniform distribution classification targets using quantiles."""
    
    def __init__(self, nbins: int = 13, lookforward_window: int = 5000):
        self.nbins = nbins
        self.lookforward_window = lookforward_window
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        price_movements = self._calculate_price_movements(df)
        quantile_boundaries = np.quantile(price_movements, np.linspace(0, 1, self.nbins + 1))
        labels = np.digitize(price_movements, quantile_boundaries[1:-1])
        return {"classification_label": labels}
```

#### Global Threshold Classification
```python
class GlobalThresholdClassificationGenerator(TargetGenerator):
    """Uses pre-computed global thresholds for consistent classification."""
    
    def __init__(self, global_thresholds: GlobalThresholds):
        self.global_thresholds = global_thresholds
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        price_movements = self._calculate_price_movements(df)
        labels = np.digitize(price_movements, self.global_thresholds.quantile_boundaries[1:-1])
        return {"classification_label": labels}
```

### 3. Regression Target Generators

#### Directional MFE (Maximum Favorable Excursion)
```python
class DirectionalMFEGenerator(TargetGenerator):
    """Generates buy-side and sell-side MFE regression targets."""
    
    def __init__(self, lookforward_horizon: int = 3000, lookback_window: int = 200):
        self.lookforward_horizon = lookforward_horizon
        self.lookback_window = lookback_window
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        mfe_buy, mfe_sell = self._calculate_directional_mfe(df)
        return {
            "mfe_buy_bps": mfe_buy,
            "mfe_sell_bps": mfe_sell
        }
```

#### Price Movement Regression
```python
class PriceMovementGenerator(TargetGenerator):
    """Simple price movement regression targets."""
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        price_movements = self._calculate_price_movements(df)
        return {"price_movement_bps": price_movements}
```

### 4. Custom Target Generators

Users can easily create custom labeling logic:

```python
class CustomVolatilityGenerator(TargetGenerator):
    """Custom volatility-based regression target."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        # Custom volatility calculation
        volatility = self._calculate_rolling_volatility(df, self.window_size)
        return {"volatility_target": volatility}
    
    @property
    def target_type(self) -> str:
        return "regression"
    
    @property
    def required_columns(self) -> List[str]:
        return ["mid_price", "timestamp"]
```

### 5. Target Generator Factory

```python
class TargetGeneratorFactory:
    """Factory for creating target generators from configuration."""
    
    _GENERATORS = {
        "quantile_classification": QuantileClassificationGenerator,
        "global_threshold_classification": GlobalThresholdClassificationGenerator,
        "directional_mfe": DirectionalMFEGenerator,
        "price_movement": PriceMovementGenerator,
        "volatility": CustomVolatilityGenerator,
    }
    
    @classmethod
    def create(cls, generator_type: str, **kwargs) -> TargetGenerator:
        """Create a target generator by name."""
        if generator_type not in cls._GENERATORS:
            raise ValueError(f"Unknown generator type: {generator_type}")
        return cls._GENERATORS[generator_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, generator_class: type):
        """Register a new target generator type."""
        cls._GENERATORS[name] = generator_class
```

### 6. Modular Dataset Builder

```python
class ModularDatasetBuilder:
    """Dataset builder with pluggable target generation."""
    
    def __init__(self, target_generators: List[TargetGenerator]):
        self.target_generators = target_generators
        self._validate_generators()
    
    def build_dataset(self, symbol_df: pl.DataFrame) -> pl.DataFrame:
        """Build dataset with all configured targets."""
        result_df = symbol_df.clone()
        
        for generator in self.target_generators:
            # Validate required columns
            self._validate_required_columns(symbol_df, generator)
            
            # Generate targets
            targets = generator.generate_targets(symbol_df)
            
            # Add targets to DataFrame
            for target_name, target_array in targets.items():
                result_df = result_df.with_columns(
                    pl.Series(target_name, target_array)
                )
        
        return result_df
```

## Usage Examples

### Single Target Generation
```python
# Classification only
generator = QuantileClassificationGenerator(nbins=13)
builder = ModularDatasetBuilder([generator])
dataset = builder.build_dataset(market_data)

# Regression only  
generator = DirectionalMFEGenerator(lookforward_horizon=3000)
builder = ModularDatasetBuilder([generator])
dataset = builder.build_dataset(market_data)
```

### Multi-Target Generation
```python
# Combine classification and regression
generators = [
    QuantileClassificationGenerator(nbins=13),
    DirectionalMFEGenerator(lookforward_horizon=3000),
    CustomVolatilityGenerator(window_size=1000)
]

builder = ModularDatasetBuilder(generators)
dataset = builder.build_dataset(market_data)

# Result DataFrame contains:
# - classification_label (int)
# - mfe_buy_bps (float)  
# - mfe_sell_bps (float)
# - volatility_target (float)
```

### Custom Target Generator
```python
class MomentumGenerator(TargetGenerator):
    """Custom momentum-based target."""
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        momentum = self._calculate_momentum_score(df)
        return {"momentum_score": momentum}
    
    @property
    def target_type(self) -> str:
        return "regression"
    
    @property  
    def required_columns(self) -> List[str]:
        return ["mid_price", "volume"]

# Register and use
TargetGeneratorFactory.register("momentum", MomentumGenerator)
generator = TargetGeneratorFactory.create("momentum", window=500)
```

### ML-Based Labeling
```python
class MLLabelGenerator(TargetGenerator):
    """Use pre-trained ML model for labeling."""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        features = self._extract_features(df)
        predictions = self.model.predict(features)
        return {"ml_labels": predictions}
```

## Configuration System

```python
@dataclass
class TargetConfig:
    """Configuration for target generation."""
    generator_type: str
    target_name: str
    parameters: Dict[str, Any]

@dataclass  
class DatasetConfig:
    """Complete dataset configuration."""
    targets: List[TargetConfig]
    symbol_filters: Optional[Dict[str, Any]] = None
    output_format: str = "parquet"

# Example configuration
config = DatasetConfig(
    targets=[
        TargetConfig("quantile_classification", "price_direction", {"nbins": 13}),
        TargetConfig("directional_mfe", "mfe_targets", {"lookforward_horizon": 3000}),
        TargetConfig("custom_volatility", "vol_target", {"window_size": 1000})
    ]
)
```

## Adding New Target Types

To add a new target type, simply:

1. **Implement the interface**:
```python
class NewTargetGenerator(TargetGenerator):
    def generate_targets(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        # Your custom logic here
        return {"new_target": target_array}
    
    @property
    def target_type(self) -> str:
        return "regression"  # or "classification"
    
    @property
    def required_columns(self) -> List[str]:
        return ["required_column1", "required_column2"]
```

2. **Register with factory**:
```python
TargetGeneratorFactory.register("new_target", NewTargetGenerator)
```

3. **Use immediately**:
```python
generator = TargetGeneratorFactory.create("new_target", param1=value1)
builder = ModularDatasetBuilder([generator])
```

## Benefits

1. **Extreme Modularity**: Each target type is completely independent
2. **Zero Coupling**: Target generation is separated from data processing
3. **Easy Extension**: Adding new targets requires only implementing one interface
4. **Type Safety**: Clear contracts for classification vs regression
5. **Composability**: Mix and match any combination of targets
6. **Testing**: Each generator can be tested in isolation
7. **Performance**: Generators can be optimized independently

This architecture makes it trivial to experiment with new labeling approaches, integrate external ML models, or create domain-specific targets without touching any core data processing code.