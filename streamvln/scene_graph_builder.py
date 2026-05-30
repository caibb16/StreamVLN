import sys
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# UniGoal is now copied to the local directory
STREAMVLN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIGOAL_PATH = os.path.join(STREAMVLN_PATH, 'UniGoal')
THIRD_PARTY_PATH = os.path.join(STREAMVLN_PATH, 'third_party', 'Grounded-Segment-Anything')

# Add paths to sys.path
if UNIGOAL_PATH not in sys.path:
    sys.path.insert(0, UNIGOAL_PATH)
if os.path.join(UNIGOAL_PATH, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(UNIGOAL_PATH, 'src'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.insert(0, THIRD_PARTY_PATH)
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'GroundingDINO'))
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'segment_anything'))


class SimpleSceneGraphBuilder:
    """Simple scene graph builder without heavy models."""

    def __init__(self, args):
        self.step_count = 0
        self.object_history = []

    def update(self, obs):
        self.step_count += 1

    def get_memory_text(self):
        if not self.object_history:
            return ""
        objects = ", ".join(self.object_history)
        return f"Objects observed: {objects}."

    def reset(self):
        self.step_count = 0
        self.object_history = []


class SceneGraphBuilder:
    USE_SIMPLE = False  # Set to True to use simple builder without heavy models

    def __init__(self, args):
        if self.USE_SIMPLE:
            self.simple_builder = SimpleSceneGraphBuilder(args)
        else:
            from src.graph.graph import Graph
            self.graph = Graph(args)
        self.step_count = 0
        self.object_history = []
        self.relation_history = []

    def update(self, obs):
        self.step_count += 1
        if self.USE_SIMPLE:
            self.simple_builder.update(obs)
        else:
            if self.step_count % 2 == 0:
                self.graph.set_observations(obs)
                self.graph.set_navigate_steps(self.step_count)
                self.graph.update_scenegraph()
                self._update_history()

    def _update_history(self):
        sg = self.graph.get_scenegraph()
        for node in sg['nodes']:
            obj_name = node['id'].rsplit('_', 1)[0]
            if obj_name not in self.object_history:
                self.object_history.append(obj_name)
        for edge in sg['edges']:
            # print(f"[DEBUG] edge in scenegraph: {edge['source']} -> {edge['target']}, type={repr(edge['type'])}")
            if edge['type']:
                rel = f"{edge['source'].rsplit('_', 1)[0]} {edge['type']} {edge['target'].rsplit('_', 1)[0]}"
                if rel not in self.relation_history:
                    self.relation_history.append(rel)
        # print(f"[DEBUG] _update_history: object_history={self.object_history}, relation_history={self.relation_history}")

    def get_memory_text(self):
        if self.USE_SIMPLE:
            return self.simple_builder.get_memory_text()
        if not self.object_history:
            return ""
        objects = ", ".join(self.object_history)
        # relations = ". ".join(self.relation_history) if self.relation_history else "None" 
        return f"Objects observed: {objects}."

    def reset(self):
        if self.USE_SIMPLE:
            self.simple_builder.reset()
        else:
            self.graph.reset()
        self.step_count = 0
        self.object_history = []
        self.relation_history = []