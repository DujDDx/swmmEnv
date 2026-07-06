"""
Unit tests for TimeSync.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.sim.time_sync import TimeSync


class TestTimeSync:
    """Test TimeSync class."""

    def test_init_basic(self):
        """Test basic initialization."""
        ts = TimeSync(decision_interval=300, swmm_step=10)

        assert ts.decision_interval == 300
        assert ts.swmm_step == 10
        assert ts.skip_steps == 30

    def test_init_divisible_check(self):
        """Test that non-divisible intervals raise error."""
        with pytest.raises(ValueError, match="must be divisible"):
            TimeSync(decision_interval=300, swmm_step=7)

    def test_init_negative_interval(self):
        """Test that negative interval raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            TimeSync(decision_interval=-300, swmm_step=10)

    def test_init_negative_swmm_step(self):
        """Test that negative swmm_step raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            TimeSync(decision_interval=300, swmm_step=-10)

    def test_advance(self):
        """Test advance method."""
        ts = TimeSync(decision_interval=300, swmm_step=10)

        # Mock engine with step method
        mock_engine = Mock()
        mock_engine.step = Mock()

        ts.advance(mock_engine)

        # Should have called engine.step 30 times
        assert mock_engine.step.call_count == 30
        assert ts.get_swmm_steps() == 30
        assert ts.get_rl_steps() == 1

    def test_advance_multiple_times(self):
        """Test multiple advances."""
        ts = TimeSync(decision_interval=300, swmm_step=10)
        mock_engine = Mock()
        mock_engine.step = Mock()

        ts.advance(mock_engine)
        ts.advance(mock_engine)

        assert ts.get_swmm_steps() == 60
        assert ts.get_rl_steps() == 2

    def test_should_act(self):
        """Test should_act method."""
        ts = TimeSync(decision_interval=300, swmm_step=10)

        # Should act at multiples of skip_steps
        assert ts.should_act(0) == True
        assert ts.should_act(30) == True
        assert ts.should_act(60) == True

        # Should not act at other steps
        assert ts.should_act(1) == False
        assert ts.should_act(15) == False
        assert ts.should_act(29) == False

    def test_reset(self):
        """Test reset method."""
        ts = TimeSync(decision_interval=300, swmm_step=10)
        mock_engine = Mock()
        mock_engine.step = Mock()

        ts.advance(mock_engine)
        assert ts.get_rl_steps() == 1

        ts.reset()

        assert ts.get_swmm_steps() == 0
        assert ts.get_rl_steps() == 0

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        ts = TimeSync(decision_interval=300, swmm_step=10)
        mock_engine = Mock()
        mock_engine.step = Mock()

        ts.advance(mock_engine)

        # 30 steps * 10 seconds = 300 seconds
        assert ts.get_elapsed_time() == 300
        assert ts.get_elapsed_time_minutes() == 5.0

    def test_get_skip_steps(self):
        """Test get_skip_steps."""
        ts = TimeSync(decision_interval=300, swmm_step=10)

        assert ts.get_skip_steps() == 30

    def test_repr(self):
        """Test string representation."""
        ts = TimeSync(decision_interval=300, swmm_step=10)

        repr_str = repr(ts)

        assert 'TimeSync' in repr_str
        assert 'decision_interval=300' in repr_str
        assert 'swmm_step=10' in repr_str


from unittest.mock import Mock


if __name__ == '__main__':
    pytest.main([__file__, '-v'])