"""An OpenAI Gym interface to the NES game Arkanoid"""
from nes_py import NESEnv
import enum


class Arkanoid(NESEnv):
    """An OpenAI Gym interface to the NES game Arkanoid"""

    def __init__(self):
        """Initialize a new Arkanoid environment."""
        super(Arkanoid, self).__init__("./Arkanoid (USA).nes")
        # setup any variables to use in the below callbacks here

    def _will_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to perform setup before and episode resets.
        # the method returns None
        pass

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to access the RAM of the emulator
        # and perform setup for each episode.
        # the method returns None
        pass

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        pass

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return 0

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return False

    def _get_info(self):
        """Return the info after a step occurs."""
        score = 0
        remaining_lives = 0
        level = 1
        arkanoid_x = 0
        powerballs = 1
        sticked_powerball = False
        large_arkanoid = False
        laser_arkanoid = False
        open_portal = False
        # slowed = False

        return {
            "score": score,
            "remaining_lives": remaining_lives,
            "level": level,
            "arkanoid_x": arkanoid_x,
            "powerballs": powerballs,
            "sticked_powerball": sticked_powerball,
            "large_arkanoid": large_arkanoid,
            "laser_arkanoid": laser_arkanoid,
            "open_portal": open_portal,
        }


# explicitly define the outward facing API for the module
__all__ = [Arkanoid.__name__]
