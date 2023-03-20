import curses
import os
import pprint

class Terminal():
    def __init__(self):
        # We don't allow a really small terminal
        self.toosmall = False

        # Init curses terminal
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.screen.keypad(True)

        # Load all our asciiarts
        self.asciiarts = {}
        for art in os.listdir("./asciiart"):
            with open(f"./asciiart/{art}", 'r') as f:
                width = art.split(".")[0].split("-")[1]
                self.asciiarts[width] = f.readlines()

    def close(self):
        # Clean our mess
        curses.nocbreak()
        self.screen.keypad(False)
        curses.echo()
        curses.endwin()

    def startframe(self):
        # Clear the screen
        self.screen.erase()

        # Check if the terminal is too small
        if (self.screen.getmaxyx()[0] < 16 or self.screen.getmaxyx()[1] < 70):
            self.screen.addstr("Terminal too small!!\n")
            self.toosmall = True
        else:
            self.toosmall = False

        # Draw the logo
        asciiart = None
        for width in sorted(self.asciiarts.keys(), key=int):
            if self.screen.getmaxyx()[1] >= int(width):
                asciiart = self.asciiarts[width]

        if (not self.toosmall and asciiart):
            for i, line in enumerate(asciiart):
                self.write(line, i, 0)
            self.write("\n\n")

    def endframe(self):
        self.screen.refresh()

    def write(self, string, y=None, x=None):
        if (self.toosmall):
            return
        if (x != None and y != None):
            self.screen.addstr(y, x, string)
        else:
            self.screen.addstr(string)

    def writedict(self, dict):
        if (self.toosmall):
            return
        self.write(f"{pprint.pformat(dict, sort_dicts=False, width=70)}\n")
