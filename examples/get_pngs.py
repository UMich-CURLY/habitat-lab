import os
from PIL import Image
from IPython import embed
import numpy as np

def extract_gif_frames(gif_path, output_dir):
    """
    Extract frames from a GIF, resize them to 60x60, and save them into separate folders.

    :param gif_path: Path to the GIF file.
    :param output_dir: Path to the output directory where frames will be saved.
    """
    # Open the GIF file
    with Image.open(gif_path) as gif:
        num_frames = gif.n_frames
        print("Number of frames: ", num_frames)
    # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for frame_index in range(num_frames):
            # Seek to the current frame
            gif.seek(frame_index)

            # Create a folder for the current frame
            frame_folder = os.path.join(output_dir, str(frame_index))
            os.makedirs(frame_folder, exist_ok=True)

            # Resize the current frame to 60x60
            # resized_frame = gif.resize((60, 60))
            resized_frame = gif.copy()
            # Save the current frame as an image
            resized_frame.save(frame_folder+"/raw_img.png")
            resized_frame = resized_frame.resize((60, 60))
            resized_frame.save(frame_folder+"/grid_img.png")

def get_trajs(npy_path, output_dir):
    trajectory_data = np.load(npy_path, allow_pickle=True)
    print("Number of frames: ", len(trajectory_data))
    robot_traj = np.zeros((len(trajectory_data), 2))
    human_traj = np.zeros((len(trajectory_data), 2))
    human_pos_prev = trajectory_data[0][1]
    for frame_index in range(trajectory_data.shape[0]):
        robot_pos, human_pos, robot_goal, time_based = trajectory_data[frame_index]
        
        # Convert positions from 512x512 to 60x60 basis
        robot_pos_scaled = (robot_pos[0] * 60 // 512, robot_pos[1] * 60 // 512)
        human_pos_scaled = (human_pos[0] * 60 // 512, human_pos[1] * 60 // 512)
        robot_goal_scaled = (robot_goal[0] * 60 // 512, robot_goal[1] * 60 // 512)
        heading = [0.0, time_based[0]]
        vels = [0.0, np.linalg.norm((human_pos-human_pos_prev)*6/512)/time_based[1]]
        human_pos_prev = human_pos
        robot_traj[frame_index] = robot_pos_scaled
        human_traj[frame_index] = human_pos_scaled
        np.save(output_dir+"/"+str(frame_index)+"/robot_past_traj.npy", robot_traj[:frame_index+1])
        np.save(output_dir+"/"+str(frame_index)+"/human_past_traj.npy", human_traj[:frame_index+1])
        robot_goal = robot_goal_scaled
        np.save(output_dir+"/"+str(frame_index)+"/goal.npy", robot_goal)
        np.save(output_dir+"/"+str(frame_index)+"/heading.npy", heading)
        np.save(output_dir+"/"+str(frame_index)+"/vels.npy", vels)
    np.save(output_dir+"/traj.npy", robot_traj)
    return robot_traj, human_traj, robot_goal



# Example usage
dir = "data/vids/Test_data_finalest/Only8/exp_8.12"
number_of_runs = (len(os.listdir(dir))-1)//2
for i in range(number_of_runs):
    gif_path = dir+"/episode_"+str(i)+".gif"
    output_dir = "test_irl/demo_"+str(i)
    extract_gif_frames(gif_path, output_dir)
    get_trajs(dir+"/episode_"+str(i)+".npy", output_dir)
