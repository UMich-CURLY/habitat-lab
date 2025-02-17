from PIL import Image
import numpy as np
from IPython import embed
def extract_frames_from_gif(gif_path):
    with Image.open(gif_path) as img:
        frames = []
        num_frames = img.n_frames
        img_path = "robo_frames/exp_7.62/demo_3/"
        for i in range(num_frames):

            # img_frame = img.copy()
            # try:
            img.seek(i)
            frames.append(img.copy().convert("RGBA"))  # Convert to RGBA for alpha
            # img.save(img_path + str(i) + ".png")
            save_stacked_image(frames[-1], img_path + str(i) + ".png")
                # frames.append(img.seek(i).convert("RGBA"))  # Convert to RGBA for alpha
            # img.seek(img.tell() + 1)
            # except:
                # print("Error on frame: ", i)
                
    return frames

def create_stacked_image(frames):
    width, height = frames[0].size
    stacked_image = np.zeros((height, width, 4), dtype=np.uint8)  # Create a blank RGBA image

    for i, frame in enumerate(frames):
        alpha_value = int((i + 1) * (255 // len(frames)))  # Calculate alpha value based on frame index
        frame_array = np.array(frame)
        frame_array[..., 3] = alpha_value  # Set alpha channel

        # Add the frame to the stacked image (this will blend the images based on alpha)
        stacked_image = np.mean(stacked_image, frame_array)
        # stacked_image = (stacked_image+frame_array)/2
    return Image.fromarray(stacked_image, 'RGBA')

def save_stacked_image(stacked_image, output_path):
    stacked_image.save(output_path)

def main():
    gif_path = 'data/vids/robot_traj_demo_3_7.62_mid.gif'
    output_path = 'orca_backoff_24.png'

    frames = extract_frames_from_gif(gif_path)
    # stacked_image = create_stacked_image(frames)
    # save_stacked_image(stacked_image, output_path)

if __name__ == "__main__":
    main()
