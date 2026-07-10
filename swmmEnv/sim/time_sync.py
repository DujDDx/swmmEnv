"""
TimeSync - Synchronization between RL decision steps and SWMM simulation steps.

This module handles the mismatch between RL control intervals (typically minutes)
and SWMM routing steps (typically seconds), ensuring proper timing alignment.
"""


class TimeSync:
    """
    Manage synchronization between RL steps and SWMM simulation steps.

    RL agents typically make decisions at intervals of minutes (e.g., 5 minutes),
    while SWMM routing steps are typically seconds (e.g., 10 seconds).
    This class handles advancing the correct number of SWMM steps per RL step.

    Example:
        >>> time_sync = TimeSync(decision_interval=300, swmm_step=10)
        >>> time_sync.advance(engine)  # Advances 30 SWMM steps (300/10)
    """

    def __init__(self, decision_interval: int, swmm_step: int):
        """
        Initialize TimeSync.

        Args:
            decision_interval: RL decision interval in seconds
                              (e.g., 300 for 5-minute intervals)
            swmm_step: SWMM routing step in seconds
                      (e.g., 10 for 10-second steps)

        Raises:
            ValueError: If decision_interval is not divisible by swmm_step
        """
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")

        if swmm_step <= 0:
            raise ValueError("swmm_step must be positive")

        if decision_interval % swmm_step != 0:
            raise ValueError(
                f"decision_interval ({decision_interval}) must be "
                f"divisible by swmm_step ({swmm_step})"
            )

        self.decision_interval = decision_interval
        self.swmm_step = swmm_step
        self.skip_steps = decision_interval // swmm_step

        # Tracking
        self._swmm_steps_executed = 0
        self._rl_steps_executed = 0

    def advance(self, engine) -> None:
        """
        Advance SWMM simulation by one RL decision interval.

        Executes skip_steps number of SWMM steps to advance by decision_interval.
        Optimized to reduce Python↔C call overhead by inlining action application
        and batch step counting, avoiding PySWMM callback mechanism.

        Args:
            engine: SWMMEngine instance with sim and links attributes
        """
        sim = engine.sim
        if sim is None:
            return

        # Inline action application (replaces before_step callback)
        pending = engine._pending_actions
        if pending:
            links = engine.links
            for link_id, setting in pending.items():
                if link_id in links:
                    links[link_id].target_setting = setting
            pending.clear()

        # Batch step execution (replaces for-loop with individual steps)
        steps = self.skip_steps
        for _ in range(steps):
            next(sim)

        # Batch step count update (replaces after_step callback)
        engine._step_count += steps
        self._swmm_steps_executed += steps
        self._rl_steps_executed += 1

    def should_act(self, current_swmm_step: int) -> bool:
        """
        Check if agent should act at current SWMM step.

        Args:
            current_swmm_step: Current SWMM step number

        Returns:
            True if this step is a decision point for RL agent
        """
        return current_swmm_step % self.skip_steps == 0

    def reset(self) -> None:
        """
        Reset step counters.
        """
        self._swmm_steps_executed = 0
        self._rl_steps_executed = 0

    def get_swmm_steps(self) -> int:
        """
        Get total SWMM steps executed.

        Returns:
            Number of SWMM steps executed
        """
        return self._swmm_steps_executed

    def get_rl_steps(self) -> int:
        """
        Get total RL steps executed.

        Returns:
            Number of RL decision steps executed
        """
        return self._rl_steps_executed

    def get_elapsed_time(self) -> float:
        """
        Get elapsed simulation time in seconds.

        Returns:
            Elapsed time in seconds
        """
        return self._swmm_steps_executed * self.swmm_step

    def get_elapsed_time_minutes(self) -> float:
        """
        Get elapsed simulation time in minutes.

        Returns:
            Elapsed time in minutes
        """
        return self.get_elapsed_time() / 60.0

    def get_skip_steps(self) -> int:
        """
        Get number of SWMM steps per RL step.

        Returns:
            Number of SWMM steps to skip per RL step
        """
        return self.skip_steps

    def __repr__(self) -> str:
        return (
            f"TimeSync(decision_interval={self.decision_interval}, "
            f"swmm_step={self.swmm_step}, skip_steps={self.skip_steps})"
        )