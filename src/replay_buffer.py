import numpy as np


class ReplayBuffer:
    def __init__(self, size, state_shape):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self.obs_buf = np.zeros([size, *state_shape], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *state_shape], dtype=np.float32)
        self.done_buf = np.zeros(size)
        self.max_size = size
        self.ptr, self.size, = 0, 0

    def __len__(self):
        return self.size

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        self.obs_buf[self.ptr] = obs_t
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = obs_tp1
        self.done_buf[self.ptr] = int(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """

        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return self.obs_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs], self.next_obs_buf[idxs], self.done_buf[idxs]
