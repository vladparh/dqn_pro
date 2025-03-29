from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import gymnasium as gym
import ale_py
import cv2
import numpy as np
import src.atari_wrappers as atari_wrappers
from src.framebuffer import FrameBuffer


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        super().__init__(env)

        self.img_size = (1, 84, 84)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb, channel_weights=[0.2125, 0.7154, 0.0721]):
        return np.dot(rgb[...,:3], channel_weights)[None, :, :]

    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        img = cv2.resize(img, (84, 84))  # resize
        img = self._to_gray_scale(img)  # grayscale
        img = img.astype(np.float32) / 256.  # float
        return img


def PrimaryAtariWrap(env, clip_rewards=True):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


def make_env(env_name, clip_rewards=True, sticky_actions=True):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="rgb_array", repeat_action_probability=0.25 if sticky_actions else 0.0)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env
