import pickle

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
ENVIRONMENT_IDS = (
    #  "HandManipulateEgg-v0",
    #  "HandManipulatePen-v0",
    #  "HandManipulateBlock-v0",
    "HandManipulateEgg-v1",
    "HandManipulatePen-v1",
    "HandManipulateBlock-v1",
)


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env1 = gym.make(environment_id, target_position="fixed")
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.unwrapped.target_position == env2.unwrapped.target_position, (
        env1.target_position,
        env2.target_position,
    )


def test_goal_distance_ignores_quaternion_sign():
    env = gym.make("HandManipulateBlockRotateZ-v1").unwrapped
    goal = np.array([0.0, 0, 0, 0, 0, 0, 1.0])  # 180 degree yaw
    achieved = goal.copy()
    achieved[3:] *= -1  # same orientation, opposite sign

    _, d_rot = env._goal_distance(achieved, goal)

    assert d_rot == pytest.approx(0.0, abs=1e-6)
    assert env._is_success(achieved, goal) == 1.0


def test_batched_rotation_rewards_are_invariant_to_quaternion_sign():
    with gym.make("HandManipulateBlockRotateZ-v1", reward_type="sparse") as wrapped:
        env = wrapped.unwrapped
        angles = env.rotation_threshold * np.array([0.0, 0.5, 1.5])
        desired = np.zeros((3, 7))
        desired[:, 3] = 1.0
        achieved = desired.copy()
        achieved[:, 3] = np.cos(angles / 2)
        achieved[:, 6] = np.sin(angles / 2)
        for sign in (1, -1):
            goals = achieved.copy()
            goals[:, 3:] *= sign
            rewards = env.compute_reward(goals, desired, {})
            np.testing.assert_array_equal(rewards, [0, 0, -1])
            np.testing.assert_array_equal(
                rewards,
                [env.compute_reward(a, d, {}) for a, d in zip(goals, desired)],
            )
