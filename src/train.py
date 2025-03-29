from bestconfig import Config
import random
import numpy as np
import torch
import torch.nn as nn
from src.create_env import make_env
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
import src.utils as utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from src.loss import compute_td_loss


def main():
    config = Config('configs/train.yaml')
    seed = config.int('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = make_env(config.str('env_name'))
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    state, _ = env.reset(seed=seed)
    agent = DQNAgent(n_actions, epsilon=config.float('init_epsilon')).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=config.float('lr'))
    target_network = DQNAgent(n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_replay = ReplayBuffer(config.int('replay_buffer_size'), state_shape)
    num_interactions = 0
    for i in range(1000):
        _, _, num_interactions = utils.play_and_record(initial_state=state,
                                                       agent=agent,
                                                       env=env,
                                                       exp_replay=exp_replay,
                                                       num_interactions=num_interactions,
                                                       n_steps=10 ** 2,
                                                       max_steps_per_episode=config.int('max_steps_per_episode'),
                                                       seed=seed)
        if len(exp_replay) == config.int('min_replay_buffer_size'):
            break

    writer = SummaryWriter()
    state, _ = env.reset(seed=seed)
    writer.add_hparams({'env': config.str('env_name'),
                        'model_type': 'dqn_pro' if config.bool('dqn_pro') else 'dqn'}, {})
    try:
        for step in trange(config.int('total_steps') + 1):
            agent.epsilon = utils.linear_decay(config.float('init_epsilon'),
                                               config.float('final_epsilon'),
                                               step,
                                               config.int('decay_steps'))

            # play
            _, state, num_interactions = utils.play_and_record(initial_state=state,
                                                               agent=agent,
                                                               env=env,
                                                               exp_replay=exp_replay,
                                                               num_interactions=num_interactions,
                                                               n_steps=config.int('timesteps_per_epoch'),
                                                               max_steps_per_episode=config.int('max_steps_per_episode'),
                                                               seed=seed)

            # train
            obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(config.int('batch_size'))

            loss = compute_td_loss(states=obs_batch,
                                   actions=act_batch,
                                   rewards=reward_batch,
                                   next_states=next_obs_batch,
                                   is_done=is_done_batch,
                                   agent=agent, target_agent=target_network,
                                   gamma=config.float('gamma'), device=device)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), config.float('max_grad_norm'))
            if config.bool('dqn_pro'):
                utils.update_weights(agent, target_network, config.float('lr'), config.float('c_value'))
            opt.step()
            opt.zero_grad()

            if step % config.int('loss_freq') == 0:
                writer.add_scalar("td_loss", loss, step)
                writer.add_scalar("grad_norm", grad_norm, step)

            if step % config.int('refresh_target_network_freq') == 0:
                # Load agent weights into target_network
                target_network.load_state_dict(agent.state_dict())

            if step % config.int('eval_freq') == 0:
                writer.add_scalar("mean_rw",
                                  utils.evaluate(
                                      make_env(config.str('env_name'), clip_rewards=False),
                                      agent, n_games=config.int('n_eval_episodes'),
                                      epsilon=config.float('eval_epsilon'),
                                      t_max=config.int('max_steps_per_episode'),
                                      seed=seed
                                  ),
                                  step
                                  )
                initial_state_q_values = agent.get_qvalues(
                    [make_env(config.str('env_name')).reset(seed=seed)[0]]
                )
                writer.add_scalar("initial_state_v",
                                  np.max(initial_state_q_values),
                                  step
                                  )

                writer.add_scalar("buffer_size",
                                  len(exp_replay),
                                  step
                                  )

                writer.add_scalar("epsilon",
                                  agent.epsilon,
                                  step
                                  )

                torch.save(agent.state_dict(), 'agent.pth')

        writer.close()
        torch.save(agent.state_dict(), 'last_agent.pth')
        print('done')

    except KeyboardInterrupt:
        writer.close()
        torch.save(agent.state_dict(), 'last_agent.pth')


if __name__ == '__main__':
    main()
