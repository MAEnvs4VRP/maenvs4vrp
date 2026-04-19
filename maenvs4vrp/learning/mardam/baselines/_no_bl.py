from maenvs4vrp.learning.mardam.baselines._base import Baseline


class NoBaseline(Baseline):
    def __init__(self, learner):
        super().__init__(learner, True)

    def eval(self, vrp_dynamics):
        return None
