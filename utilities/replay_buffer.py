import numpy as np



class TransReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        return self.get_truncated_episodes_batch(batch_size)

    def get_truncated_episodes_batch(self, batch_size):
        sample_range = len(self.buffer) - batch_size + 1
        start_indice = np.random.choice(sample_range, 1, replace=False)[0]
        batch_buffer = [self.buffer[i+start_indice] for i in range(batch_size)]
        return batch_buffer

    def add_experience(self, trans):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(trans)

    def clear(self):
        self.buffer = []


class EpisodeReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = []
        for i in indices:
            batch_buffer.extend(self.buffer[i])
        return batch_buffer

    def add_experience(self, episode):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(episode)
