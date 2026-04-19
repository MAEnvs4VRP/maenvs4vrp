class RewardFn:
    """Agent rewards base class."""

    def __init__(self):
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """
        self.env = None

    def set_env(self, env):
        """
        Set Environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        self.env = env

    def get_reward(self):
        """
        Get Reward.

        Args:
            n/a.

        Returns:
            None.
        """

        raise NotImplementedError()
