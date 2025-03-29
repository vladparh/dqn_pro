import numpy as np
import psutil
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian


def get_cum_discounted_rewards(rewards, gamma):
    """
    evaluates cumulative discounted rewards:
    r_t + gamma * r_{t+1} + gamma^2 * r_{t_2} + ...
    """
    cum_rewards = []
    cum_rewards.append(rewards[-1])
    for r in reversed(rewards[:-1]):
        cum_rewards.insert(0, r + gamma * cum_rewards[0])
    return cum_rewards


def play_and_log_episode(env, agent, gamma=0.99, t_max=10000):
    """
    always greedy
    """
    states = []
    v_mc = []
    v_agent = []
    q_spreads = []
    td_errors = []
    rewards = []

    s, _ = env.reset()
    for step in range(t_max):
        states.append(s)
        qvalues = agent.get_qvalues([s])
        max_q_value, min_q_value = np.max(qvalues), np.min(qvalues)
        v_agent.append(max_q_value)
        q_spreads.append(max_q_value - min_q_value)
        if step > 0:
            td_errors.append(
                np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

        action = qvalues.argmax(axis=-1)[0]

        s, r, terminated, truncated, _ = env.step(action)
        rewards.append(r)
        if terminated or truncated:
            break
    td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

    v_mc = get_cum_discounted_rewards(rewards, gamma)

    return_pack = {
        'states': np.array(states),
        'v_mc': np.array(v_mc),
        'v_agent': np.array(v_agent),
        'q_spreads': np.array(q_spreads),
        'td_errors': np.array(td_errors),
        'rewards': np.array(rewards),
        'episode_finished': np.array(terminated or truncated)
    }

    return return_pack


def img_by_obs(obs, state_dim):
    """
    Unwraps obs by channels.
    observation is of shape [c, h=w, w=h]
    """
    return obs.reshape([-1, state_dim[2]])


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')


def play_and_record(initial_state, agent, env, exp_replay, num_interactions, n_steps=1, max_steps_per_episode=27000, seed=None):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = agent.sample_actions([s])[0]
        next_s, r, terminated, truncated, _ = env.step(action)
        exp_replay.add(s, action, r, next_s, terminated)
        sum_rewards += r
        s = next_s
        num_interactions += 1
        if terminated or truncated or num_interactions >= max_steps_per_episode - 1:
            s, _ = env.reset(seed=seed)
            num_interactions = 0

    return sum_rewards, s, num_interactions


def evaluate(env, agent, n_games=1, epsilon=0.0, t_max=10000, seed=None):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    orig_eps = agent.epsilon
    agent.epsilon = epsilon
    for _ in range(n_games):
        s, _ = env.reset(seed=seed)
        reward = 0
        for _ in range(t_max):
            action = agent.sample_actions([s])[0]
            s, r, terminated, truncated, _ = env.step(action)
            reward += r
            if terminated or truncated:
                break

        rewards.append(reward)
    agent.epsilon = orig_eps
    return np.mean(rewards)


def update_weights(agent, target, lr, c_value):
    for agent_params, target_params in zip(agent.parameters(), target.parameters()):
        agent_params.data.copy_((1 - lr/c_value)*agent_params.data + lr/c_value*target_params.data)
