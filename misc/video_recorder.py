import os
from typing import Callable

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class SingleEnvVecVideoRecorder(VecVideoRecorder):
    def __init__(
            self,
            venv: VecEnv,
            video_folder: str,
            record_video_trigger: Callable[[int], bool],
            video_length: int = 200,
            name_prefix: str = "rl-video",
    ):
        super().__init__(venv, video_folder, record_video_trigger, video_length, name_prefix)
        self.episode_id = 0

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id, self.episode_id)

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        self.episode_id += int(dones[0])
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length or dones[0]:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos
