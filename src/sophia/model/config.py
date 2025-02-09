from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LayerConfig(BaseModel):
    type: str  # The registered layer type (e.g. "TransformerBlock")
    config: Dict[str, Any]  # A dictionary of parameters for the layer
    repeat: Optional[int] = None  # Optional: number of repeats for this layer


class ModelConfig(BaseModel):
    name: str  # The name of the model
    dtype: str  # Global precision (e.g. "float32", "bfloat16")
    layers: List[LayerConfig]  # A list of layer configurations
