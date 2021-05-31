import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Define and parse input arguments
def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', help='Path of map',
                        default='room338.bmp')
    parser.add_argument('--heatmap', help='Path of heatmap',
                        default='heatmap.csv')
    parser.add_argument('--start', help='Starting point',
                        default=(150, 450))
    return parser.parse_args()

args = get_argument()

room = cv2.imread(args.map, cv2.IMREAD_GRAYSCALE)
grid = pd.read_csv('heatmap.csv').to_numpy()
color_map = cv2.imread(args.map)
color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

# Define color parameters
yellow = (255, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)

def move_next(grid, x, y):
    max_value = (float('-inf'), 0, 0)
    
    if y-1 >= 0:  # Left
        max_value = (grid[x][y-1], x, y-1) if grid[x][y-1] > max_value[0] else max_value
    if y+1 < room.shape[1]:  # Right
        max_value = (grid[x][y+1], x, y+1) if grid[x][y+1] > max_value[0] else max_value
    if x-1 >= 0:  # Top
        max_value = (grid[x-1][y], x-1, y) if grid[x-1][y] > max_value[0] else max_value
    if x+1 < room.shape[0]:  # Bottom
        max_value = (grid[x+1][y], x+1, y) if grid[x+1][y] > max_value[0] else max_value
    
    if max_value[0] > grid[x][y]:
        cv2.circle(color_map, (max_value[2], max_value[1]), 3, yellow, -1)
        return True, max_value[1], max_value[2]
    else:
        cv2.circle(color_map, (y, x), 10, red, -1)
        return False, x, y

steps = (True, args.start[0], args.start[1])  # Move_next?, x, y
cv2.circle(color_map, (args.start[1], args.start[0]), 10, blue, -1)
while steps[0]:
    steps = move_next(grid, steps[1], steps[2])
    print(steps)

# Save routemap image
plt.imshow(color_map)
plt.title('Routemap')
plt.savefig('routemap.jpg', dpi=300)