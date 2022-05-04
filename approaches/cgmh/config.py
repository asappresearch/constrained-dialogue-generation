class CGMHConfig(object):
    def __init__(self):
        self.num_steps = 50

        self.GPU = '0'
        self.mode = 'use'
        self.sample_time = 500

        self.search_size = 100

        self.action_prob = [0.3, 0.3, 0.3, 0.1]  # the prior of 4 actions
        self.key_num = 20
        self.min_length = 7
