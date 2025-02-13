from typing import List, Optional

from pydantic import BaseModel


class NodeSpec(BaseModel):
    """
    A generic node specification for a model graph.

    - name: Unique identifier for the node’s output.
    - config: The configuration for the node. This config must contain a 'type' field,
              which is used to look up the corresponding building block in the registry.
    - inputs: A list of node names whose outputs feed into this node.
              The node's output will be stored under its own 'name'.
    """

    name: str
    config: BaseModel
    inputs: List[str] = []


class GraphConfig(BaseModel):
    """
    A generic graph configuration that describes a model architecture as a collection of nodes.

    - nodes: A list of NodeSpec objects, which will be executed in order (or in topological order).
    - output_names: A list of node names indicating which outputs are to be returned.
    - model_type: It isused as the name of the dynamically created model class.
    """

    nodes: List[NodeSpec]
    output_names: List[str]
    model_type: Optional[str] = None
