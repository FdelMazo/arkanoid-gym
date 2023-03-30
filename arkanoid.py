"""An OpenAI Gym interface to the NES game Arkanoid.

Arkanoid is a block breaker video game where the player
controls "vaus", a space vessel, to bounce a ball and break
blocks.

The screen has the following structure, with vaus in between
positions 2 and 5:

| |        | |    | |         | |
0 1        2 3    4 5         6 7

0: position 0, the leftmost position on screen
1: at x=16, the right position for the left side wall
2: vaus_very_left_x
3: vaus_left_x = vaus_middle_left_x
4: vaus_right_x = vaus_middle_right_x
5: vaus_very_right_x
6: at x=184, the left position for the right side wall
7: position 200, the rightmost position on screen

Walls are 16 pixels wide.

When not extended, d(2,3) = 8, d(3,4) = 8, d(4,5) = 8
"""

from nes_py import NESEnv
from nes_py._image_viewer import ImageViewer
import sys
NES_BUTTONS = {
    'right':  0b10000000,
    'left':   0b01000000,
    'down':   0b00100000,
    'up':     0b00010000,
    'start':  0b00001000,
    'select': 0b00000100,
    'B':      0b00000010,
    'A':      0b00000001,
    'NOOP':   0b00000000,
}

class Arkanoid(NESEnv):
    """An OpenAI Gym interface to the NES game Arkanoid"""

    def __init__(self):
        """Initialize a new Arkanoid environment."""
        super(Arkanoid, self).__init__("./Arkanoid (USA).nes")
        self.reset()
        self._backup()

    def _skip_start_screen(self):
        while self.bricks_remaining != 66:
            self._frame_advance(NES_BUTTONS['start'])
            for _ in range(30):
                self._frame_advance(NES_BUTTONS['NOOP'])

        while self.delay_automatic_release != 120:
            self._frame_advance(NES_BUTTONS['NOOP'])

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
        self._skip_start_screen()

    @property
    def bricks_remaining(self):
        return self.ram[0x000F]

    @property
    def vaus_status(self):
        status = self.ram[0x012A]
        if status == 0:
            return "dead"
        elif status == 1:
            return "normal"
        elif status == 2:
            return "extended"
        elif status == 4:
            return "laser"
        else:
            raise ValueError(f"Bad status {status}")

    @property
    def is_dead(self):
        return self.vaus_status == "dead"

    @property
    def vaus_normal(self):
        return self.vaus_status == "normal"

    @property
    def vaus_extended(self):
        return self.vaus_status == "extended"

    @property
    def vaus_laser(self):
        return self.vaus_status == "laser"

    @property
    def remaining_lives(self):
        return self.ram[0x001D]

    @property
    def score(self):
        return self._read_mem_range(0x0370, 6)

    @property
    def level(self):
        return self.ram[0x0023]

    @property
    def vaus_pos(self):
        return {
            "vaus_very_left_x": self.ram[0x011A],
            "vaus_left_x": self.ram[0x011B],
            "vaus_middle_left_x": self.ram[0x011C],
            "vaus_middle_right_x": self.ram[0x011D],
            "vaus_right_x": self.ram[0x011E],
            "vaus_very_right_x": self.ram[0x011F],
        }

    @property
    def ball_speed(self):
        return self.ram[0x0100]

    @property
    def hit_counter(self):
        return self.ram[0x0102]

    @property
    def capsule_type(self):
        value = self.ram[0x008C]
        if value == 0:
            return None
        elif value == 1:
            return "slow"
        elif value == 2:
            return "catch"
        elif value == 3:
            return "extend"
        elif value == 4:
            return "disrupt"
        elif value == 5:
            return "laser"
        elif value == 6:
            return "break"
        elif value == 7:
            return "player_extend"
        else:
            raise ValueError(f"Bad capsule type {value}")

    @property
    def capsule(self):
        return {
            "type": self.capsule_type,
            "graphic_pos_y": self.ram[0x008F],
            "graphic_pos_x": self.ram[0x0090],
            "pos_y": self.ram[0x0091],
            "pos_x": self.ram[0x0094],
            "animation_offset": self.ram[0x0092],
            "palette": self.ram[0x0093],
            "animation_delay_x4": self.ram[0x008D],
            "animation_delay_x1": self.ram[0x008E],
        }

    @property
    def delay_automatic_release(self):
        return self.ram[0x0138]

    @property
    def bricks_rows(self):
        return {
            f"row_{i}": [self.ram[0x03A0 + 11 * (i - 1) + j] for j in range(0, 11)]
            for i in range(1, 25)
        }

    @property
    def ball_x(self):
        return self.ram[0x0038]

    @property
    def ball_y(self):
        return self.ram[0x0039]

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
                self._frame_advance(NES_BUTTONS["NOOP"])

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # TODO: change this
        return self.score + 10 * self.remaining_lives + 100 * self.level

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self.remaining_lives == 0 and self.is_dead

    def _get_info(self):
        """Return the info after a step occurs."""
        # source: http://www.romdetectives.com/Wiki/index.php?title=Arkanoid_(NES)_-_RAM
        return {
            "score": self.score,
            "remaining_lives": self.remaining_lives,
            "level": self.level,
            "vaus_pos": self.vaus_pos,
            "ball_grid_x": self.ram[0x010C],
            "ball_grid_y": self.ram[0x010D],
            "ball_x": self.ball_x,  # unconfirmed
            "ball_y": self.ball_y,  # unconfirmed
            "ball_speed": self.ball_speed,
            "hit_counter": self.hit_counter,
            "catch": self.ram[0x0128],
            "vaus_status": self.vaus_status,
            "ball_grid_impact": self.ram[0x012E],  # ?
            "ball_grid_impact": self.ram[0x012F],  # ?
            "capsule": self.capsule,
            "delay_automatic_release": self.delay_automatic_release,
            "bricks": {"remaining": self.bricks_remaining, "rows": self.bricks_rows},
        }


# explicitly define the outward facing API for the module
__all__ = [Arkanoid.__name__]
