import sys
import os
import numpy as np
from PIL import Image

# Add third_party path to sys.path for Grounded-Segment-Anything
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Grounded-Segment-Anything')
if THIRD_PARTY_PATH not in sys.path:
    sys.path.insert(0, THIRD_PARTY_PATH)
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'GroundingDINO'))
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'segment_anything'))


class SceneGraphBuilder:
    """Wraps UniGoal Graph for incremental scene graph construction."""

    def __init__(self, args):
        # Defer actual Graph import to avoid heavy dependencies until needed
        from UniGoal.src.graph.graph import Graph
        self.graph = Graph(args)
        self.step_count = 0
        self.object_history = []  # accumulate observed objects
        self.relation_history = []  # accumulate observed relations

    def update(self, obs):
        """Update scene graph with current observation."""
        self.step_count += 1
        # Only update every 2 steps (like UniGoal)
        if self.step_count % 2 == 0:
            self.graph.set_observations(obs)
            self.graph.set_navigate_steps(self.step_count)
            self.graph.update_scenegraph()
            self._update_history()

    def _update_history(self):
        """Accumulate objects and relations from current scene graph."""
        sg = self.graph.get_scenegraph()
        for node in sg['nodes']:
            obj_name = node['id'].rsplit('_', 1)[0]  # e.g. "table_0" -> "table"
            if obj_name not in self.object_history:
                self.object_history.append(obj_name)
        for edge in sg['edges']:
            if edge['type']:
                rel = f"{edge['source'].rsplit('_', 1)[0]} {edge['type']} {edge['target'].rsplit('_', 1)[0]}"
                if rel not in self.relation_history:
                    self.relation_history.append(rel)

    def get_memory_text(self):
        """Build enhanced memory text for prompt injection."""
        if not self.object_history:
            return ""

        objects = ", ".join(self.object_history)
        relations = ". ".join(self.relation_history) if self.relation_history else "None"

        return (
            f"Objects observed: {objects}. "
            f"Spatial relations: {relations}."
        )

    def reset(self):
        """Reset scene graph for new episode."""
        self.graph.reset()
        self.step_count = 0
        self.object_history = []
        self.relation_history = []