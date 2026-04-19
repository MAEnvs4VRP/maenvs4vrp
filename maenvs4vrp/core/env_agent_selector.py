class BaseSelector:
    """Agent iterator base class."""

    def __init__(self):
        """
        Constructor

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        self.env = env

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(Tensor): Next agent.
        """
        raise NotImplementedError()
