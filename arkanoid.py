"""An OpenAI Gym interface to the NES game Arkanoid"""
import enum

from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace


class Arkanoid(NESEnv):
    """An OpenAI Gym interface to the NES game Arkanoid"""

    def __init__(self):
        """Initialize a new Arkanoid environment."""
        super(Arkanoid, self).__init__("./Arkanoid (USA).nes")
        self.reset()
        self._skip_start_screen()
        self._backup()

    def _skip_start_screen(self):
        while self.ram[0x00F] != 66:
            self._frame_advance(8)
            for _ in range(30):
                self._frame_advance(0)

        while self.ram[0x138] != 120:
            self._frame_advance(0)
        # breakpoint()

    # setup any variables to use in the below callbacks here

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.
        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read
        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places
        Returns:
            the integer value of this 10's place representation
        """
        return int("".join(map(str, self.ram[address : address + length])))

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

    @property
    def is_dead(self):
        return self.ram[0x012A] == 0

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        if self.is_dead:
            while self.is_dead:
                self._frame_advance(0)

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return 0

    @property
    def remaining_lives(self):
        return self.ram[0x001D]

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self.remaining_lives == 0 and self.is_dead

    def _get_info(self):
        """Return the info after a step occurs."""
        # source: http://www.romdetectives.com/Wiki/index.php?title=Arkanoid_(NES)_-_RAM
        return {
            "score": self._read_mem_range(0x0370, 6),
            "remaining_lives": self.remaining_lives,
            "level": self.ram[0x0023],
            "vaus_very_left_x": self.ram[0x011A],
            "vaus_left_x": self.ram[0x011B],
            "vaus_middle_left_x": self.ram[0x011C],
            "vaus_middle_right_x": self.ram[0x011D],
            "vaus_right_x": self.ram[0x011E],
            "vaus_very_right_x": self.ram[0x011F],
            "ball_grid_x": self.ram[0x010C],
            "ball_grid_y": self.ram[0x010D],
            "ball_speed": self.ram[0x0100],
            "hit_counter": self.ram[0x0102],
            "catch": self.ram[0x0128],
            "vaus_status": {0: "dead", 1: "normal", 2: "extended", 4: "laser"}[
                self.ram[0x012A]
            ],
            "ball_grid_impact": self.ram[0x012E],  # ?
            "ball_grid_impact": self.ram[0x012F],  # ?
        }


# explicitly define the outward facing API for the module
__all__ = [Arkanoid.__name__]
