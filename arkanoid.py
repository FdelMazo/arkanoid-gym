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

import numpy as np
import pandas as pd
from nes_py import NESEnv
from nes_py._image_viewer import ImageViewer

NES_BUTTONS = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
    "NOOP": 0b00000000,
}

INFO_COLS = [
    "ball.ball_grid_x",
    "ball.ball_grid_y",
    "ball.ball_speed",
    "vaus.vaus_status",
    "capsule.type",
    "capsule.pos_x",
    "capsule.pos_y",
    "game.is_touching",
    "vaus.vaus_very_left_x",
    "vaus.vaus_left_x",
    "vaus.vaus_middle",
    "vaus.vaus_middle_grid",
    "vaus.vaus_middle_right_x",
    "vaus.vaus_right_x",
    "vaus.vaus_very_right_x",
    "vaus.vaus_status",
]


def info_to_array(info):
    return pd.json_normalize(info)[INFO_COLS].iloc[0].values.astype(np.float32)


class Arkanoid(NESEnv):
    """An OpenAI Gym interface to the NES game Arkanoid"""

    def __init__(self, render: bool = False):
        """Initialize a new Arkanoid environment."""
        rom = "Arkanoid (USA)"
        super().__init__(f"./{rom}.nes")
        self.episode = 0
        self.reset()
        self._backup()
        if render:
            self.viewer = ImageViewer(
                caption=rom, height=256, width=256, monitor_keyboard=True
            )
        self._prev_score = None
        self.cols, self.flatten_cols = self._process_columns()
        self.arrayinfo_shape = self.info_to_array(self.info).shape[0]

    def _skip_start_screen(self):
        while self.bricks["bricks_remaining"] != 66:
            self._frame_advance(NES_BUTTONS["start"])
            for _ in range(30):
                self._frame_advance(NES_BUTTONS["NOOP"])

        while self.game["delay_automatic_release"] != 1:
            self._frame_advance(NES_BUTTONS["NOOP"])

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
        self.episode += 1
        self._prev_score = None

    @property
    def vaus(self):
        middle_left_x = self.ram[0x011C]
        middle = middle_left_x + 4
        return {
            "vaus_very_left_x": self.ram[0x011A],
            "vaus_left_x": self.ram[0x011B],
            "vaus_middle_left_x": middle_left_x,
            "vaus_middle": middle,
            "vaus_middle_grid": int(middle / 16),
            "vaus_middle_right_x": self.ram[0x011D],
            "vaus_right_x": self.ram[0x011E],
            "vaus_very_right_x": self.ram[0x011F],
            "vaus_status": self.ram[0x012A],
            "vaus_status_string": {
                0: "dead",
                1: "normal",
                2: "extended",
                3: "laser",
            }.get(self.ram[0x012A]),
        }

    @property
    def game(self):
        distance = abs(int(self.vaus["vaus_middle"]) - int(self.ball["ball_x"]))
        return {
            "remaining_lives": self.ram[0x001D],
            "score": self._read_mem_range(0x0370, 6),
            "level": self.ram[0x0023],
            "hit_counter": self.ram[0x0102],
            "delay_automatic_release": self.ram[0x0138],
            "distance_to_ball": distance,
            "ball_near": distance <= 60,
            "assume_dead": self.ball["ball_y"] == 26
            and not self.ball["ball_contained"],
            "is_dead": self.vaus["vaus_status_string"] == "dead",
            "is_catch": self.ram[0x0128],
            "is_touching": self.ball["ball_y"] == 23 and self.ball["ball_contained"],
            "will_die": not self.ball["ball_high"] and not self.ball["ball_contained"],
            "will_touch": not self.ball["ball_high"] and self.ball["ball_contained"],
        }

    @property
    def ball(self):
        ball_x = self.ram[0x0038]
        ball_contained = (
            self.vaus["vaus_very_left_x"] <= ball_x <= self.vaus["vaus_very_right_x"]
        )
        ball_side = 0
        if not ball_contained:
            ball_side = -1 if ball_x <= self.vaus["vaus_very_left_x"] else 1

        return {
            "ball_speed": self.ram[0x0100],
            "ball_side": ball_side,
            "ball_x": self.ram[0x0038],
            "ball_y": self.ram[0x0039],
            "ball_high": False if self.ram[0x0039] == 0 else self.ram[0x0039] <= 20,
            "ball_contained": ball_contained,
            "ball_grid_x": self.ram[0x010D],
            "ball_grid_y": self.ram[0x010C],
            "ball_grid_impact": self.ram[0x012E],  # ?
            "ball_grid_impact_2": self.ram[0x012F],  # ?
        }

    @property
    def bricks(self):
        rows = np.array(
            [
                [self.ram[0x03A0 + 11 * (i - 1) + j] for j in range(0, 11)]
                for i in range(1, 25)
            ]
        )
        remaining_rows_count = np.count_nonzero(rows, axis=1)
        remaining_rows_index = np.nonzero(remaining_rows_count)[0]
        remaining_columns_count = np.count_nonzero(rows, axis=0)
        remaining_columns_index = np.nonzero(remaining_columns_count)[0]
        return {
            "bricks_row": rows,
            "bricks_row_bool": rows.astype(bool).astype(int),
            "bricks_rows_remaining_count": remaining_rows_count,
            "bricks_rows_remaining_index": remaining_rows_index,
            "bricks_columns_remaining_count": remaining_columns_count,
            "bricks_columns_remaining_index": remaining_columns_index,
            "bricks_remaining": self.ram[0x000F],
        }

    @property
    def capsule(self):
        return {
            "type": self.ram[0x008C],
            "type_string": {
                0: None,
                1: "slow",
                2: "catch",
                3: "extend",
                4: "disrupt",
                5: "laser",
                6: "break",
                7: "player_extend",
            }.get(self.ram[0x008C]),
            "graphic_pos_y": self.ram[0x008F],
            "graphic_pos_x": self.ram[0x0090],
            "pos_y": self.ram[0x0091],
            "pos_x": self.ram[0x0094],
            "animation_offset": self.ram[0x0092],
            "palette": self.ram[0x0093],
            "animation_delay_x4": self.ram[0x008D],
            "animation_delay_x1": self.ram[0x008E],
        }

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        while self.game["is_dead"]:
            self._frame_advance(NES_BUTTONS["NOOP"])

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # Distance based reward -> Use with qlearning
        # return 150 - self.game['distance_to_ball']

        # Smart reward -> Use with DQN
        if self.game["is_dead"]:
            #   print("Died: -100")
            return -100

        if self.game["is_touching"]:
            #    print("Touching: +1")
            return 5

        if self._prev_score is None:
            self._prev_score = self.game["score"]
            return self.game["score"]

        delta = self.game["score"] - self._prev_score
        self._prev_score = self.game["score"]

        # if delta > 0:
        #    print(f"Delta: {delta}")
        return delta  # / 100 + self.remaining_lives + self.level / 10

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self.game["remaining_lives"] == 1

    @property
    def info(self):
        # TODO: fix this!
        return self._get_info()

    def _get_info(self):
        """Return the info after a step occurs."""
        # source: http://www.romdetectives.com/Wiki/index.php?title=Arkanoid_(NES)_-_RAM
        return {
            "game": self.game,
            "vaus": self.vaus,
            "ball": self.ball,
            "capsule": self.capsule,
            "bricks": self.bricks,
        }

    def reset(self, seed=None, options=None, return_info=True):
        return super().reset(seed, options, return_info), self._get_info()

    def _process_columns(self):
        cs = pd.json_normalize(self.info)
        cols = []
        flatten_cols = []
        for c in cs.columns:
            splitc = tuple(c.split("."))
            if isinstance(cs.loc[0, c], np.ndarray):
                flatten_cols.append(splitc)
            cols.append(splitc)

        cols = tuple(cols)
        flatten_cols = set(flatten_cols)
        return cols, flatten_cols

    def crop_screen(self, screen):
        return screen[16:, 16:192, :]

    def info_to_array(self, info):
        return info_to_array(info)
        values = []
        for x in self.cols:
            y = info
            for k in x:
                y = y[k]
            if x in self.flatten_cols:
                y = y.flatten()
            else:
                y = np.array(y)
            values.append(y)
        return np.hstack(values)


# explicitly define the outward facing API for the module
__all__ = [Arkanoid.__name__]
