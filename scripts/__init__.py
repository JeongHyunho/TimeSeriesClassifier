from collections import defaultdict

from ray.tune import Stopper


class ValStopper(Stopper):
    def __init__(self, n_graces=10, max_itr=500):
        self.n_graces = n_graces
        self.max_itr = max_itr
        self.n_wait = defaultdict(lambda: 0)

    def __call__(self, trial_id, result):
        if result['best_val_loss'] < result['val_loss']:
            self.n_wait[trial_id] += 1
        else:
            self.n_wait[trial_id] = 0

        if self.n_wait[trial_id] >= self.n_graces or self.max_itr <= result['training_iteration']:
            return True
        else:
            return False

    def stop_all(self):
        return False
