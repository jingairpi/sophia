from typing import Any, Dict

import jax.numpy as jnp
from flax import linen as nn
from pydantic import BaseModel

from sophia.model.graph import GraphConfig
from sophia.model.registry import get_class


def build_model(config: BaseModel) -> nn.Module:
    """
    Build a Flax module from a Pydantic config.

    If config is an instance of GraphConfig, build a graph-based module.
    Otherwise, instantiate a single module via instantiate_from_config.
    """
    if isinstance(config, GraphConfig):
        return _build_graph_module(config)
    else:
        return instantiate_from_config(config, top_level=True)


def instantiate_from_config(config: BaseModel, top_level: bool = False) -> Any:
    """
    Recursively instantiate a module from a config using its 'type' field.
    """
    cls = get_class(config.type)
    if top_level:
        return cls(config=config)
    kwargs = {}
    for field_name in config.model_fields:
        if field_name == "type":
            continue
        value = getattr(config, field_name)
        if isinstance(value, BaseModel) and hasattr(value, "type"):
            kwargs[field_name] = instantiate_from_config(value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def _build_graph_module(graph_config: GraphConfig) -> nn.Module:
    """
    Build a dynamic Flax module from a GraphConfig.

    For each node, the module looks up the building block using node.config.type,
    passes the node name as the submodule name (so that the parameter tree uses node.name),
    gathers its inputs from a results dict, and stores its output under node.name.

    The final model class name is set to graph_config.model_type.
    """

    class _DynamicGraphModule(nn.Module):
        config: GraphConfig

        @nn.compact
        def __call__(self, inputs: Any, deterministic: bool = True) -> Any:
            if not isinstance(inputs, dict):
                inputs = {"input_ids": inputs}
            results = {}
            for key, val in inputs.items():
                results[key] = val
            for node in self.config.nodes:
                sub_cls = get_class(node.config.type)
                node_kwargs = {}
                for field_name in node.config.model_fields:
                    if field_name == "type":
                        continue
                    value = getattr(node.config, field_name)
                    if isinstance(value, BaseModel) and hasattr(value, "type"):
                        node_kwargs[field_name] = instantiate_from_config(value)
                    else:
                        node_kwargs[field_name] = value
                submodule = sub_cls(name=node.name, **node_kwargs)

                input_tensors = [results[inp] for inp in node.inputs]
                if len(input_tensors) == 1:
                    out = submodule(input_tensors[0], deterministic=deterministic)
                else:
                    out = submodule(*input_tensors, deterministic=deterministic)
                results[node.name] = out

            if len(self.config.output_names) == 1:
                return results[self.config.output_names[0]]
            else:
                return {name: results[name] for name in self.config.output_names}

        def init(self, rng_key, inputs: Any) -> Any:
            return self.init_with_output(rng_key, inputs)[1]

    DynamicGraphModule = type(graph_config.model_type, (_DynamicGraphModule,), {})
    return DynamicGraphModule(config=graph_config)
