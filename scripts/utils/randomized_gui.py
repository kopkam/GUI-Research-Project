# imports
import tkinter as tk
from PIL import ImageGrab
from random import randint, random
from numpy.random import choice

# ----------------------------------------
# constants
MIN_WIDTH = 50
MAX_WIDTH = 700

MIN_HEIGHT = 50
MAX_HEIGHT = 300

MERGE_RATE = 0.3

# ----------------------------------------
# the widget grid
# ----------------------------------------

import tkinter as tk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# generating rows and columns

# columns
row_coords = [0]
column_coords = [0]

while 1:
    new_x = randint(MIN_WIDTH, MAX_WIDTH)

    if (row_coords[-1] + new_x) > screen_width:
        break
    else:
        row_coords.append( row_coords[-1] + new_x)

while 1:
    new_y = randint(MIN_HEIGHT, MAX_HEIGHT)

    if (column_coords[-1] + new_y) > screen_height:
        break
    else:
        column_coords.append( column_coords[-1] + new_y)

print(row_coords)
print(column_coords)