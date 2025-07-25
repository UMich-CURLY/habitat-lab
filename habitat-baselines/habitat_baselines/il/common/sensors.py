from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor
from habitat.tasks.nav.nav import EmbodiedTask
from typing import Any, Dict
from habitat.core.simulator import Observations

@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uuid = "demonstration"
        self._observation_space = spaces.Discrete(1)
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self._uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        # Reset on new episode
        if task.is_resetting:
            self.timestep = 0

        if hasattr(episode, "reference_replay") and self.timestep < len(episode.reference_replay):
            action = episode.reference_replay[self.timestep].action
        else:
            action = 0

        self.timestep += 1
        return action
