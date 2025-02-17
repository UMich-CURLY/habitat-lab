from PIL import Image
import numpy as np
from IPython import embed

def get_image():
    img = Image.

def make_start_goal_image(img, full_state):
    for i in range(len(full_state)):


def main():
    #### Our agent first ####
    folder = 'data/vids/Test_Data_nov_5/single_test/exp7.46'
    full_states = np.load(folder + '/episode_36.npy')

if __name__ == "__main__":
    main()