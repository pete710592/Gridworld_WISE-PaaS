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
    parser.add_argument('--epochs', help='Num of training epochs',
                        default=500, type=int)
    parser.add_argument('--discount_value', help='I don\'t know',
                        default=0.9, type=float)
    parser.add_argument('--resize', help='Image resize ratio',
                        default=1, type=int)
    parser.add_argument('--reward_points', help='Set rewards',
                        default=(120, 115, 125, 175, 63, 100, 50, 200), type=int, nargs='+')
    return parser.parse_args()

args = get_argument()

# Set reward
reward_points = args.reward_points
position = []
for point in range(0, len(reward_points), 2):
    position.append((reward_points[point], reward_points[point+1]))

room = cv2.imread(args.map, cv2.IMREAD_GRAYSCALE)
room = cv2.resize(room, (room.shape[1]//args.resize, room.shape[0]//args.resize))

grid = np.zeros((room.shape[0], room.shape[1]))
grid[room >= 210] = 255  # Filter out black & gray color
grid = (grid - 128) / 128.  # Normalized (-1 ~ 1)

# plt.figure(figsize=(15, 10))
# plt.subplot(221)
# plt.title('Original Map')
# plt.imshow(room, cmap = 'gray')

# plt.subplot(222)
# plt.title('Pre-processing Map')
# plt.imshow(grid, cmap = 'gray')

print('Start training ...')
for num in tqdm(range(args.epochs)): #number of times we will go through the whole grid
    for x in range(room.shape[0]):  # all the rows
        for y in range(room.shape[1]):  # all the columns
            up_grid = grid[x-1][y] if x > 0 else 0   # if going up takes us out of the grid then its value be 0
            down_grid = grid[x+1][y] if x < grid.shape[0]-1 else 0  # if going down takes us out of the grid then its value be 0
            left_grid = grid[x][y-1] if y > 0 else 0  # if going left takes us out of the grid then its value be 0
            right_grid = grid[x][y+1] if y < grid.shape[1]-1 else 0  # if going right takes us out of the grid then its value be 0

            all_dirs = [up_grid, down_grid, left_grid, right_grid]     

            value = 0
            if (x, y) in position: # the position of doors
                value = 1
            elif room[x][y] <= 210:
                value = -1
            else:
                for direc in all_dirs:
                    if direc != 0:
                        if direc > value:
                            value = direc
            grid[x][y] = value - 0.005 * args.resize

# Save heatmap image
plt.imshow(grid)
plt.title('Heatmap')
plt.colorbar()
plt.show()
plt.savefig('heatmap.jpg', dpi=300)

# Save heatmap parameters
df = pd.DataFrame(grid)
df.to_csv('heatmap.csv', index=False)