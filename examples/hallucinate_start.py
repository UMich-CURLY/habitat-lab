import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List

import magnum as mn
import numpy as np
from habitat.core.spaces import ActionSpace
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
import cv2
import habitat
import habitat.tasks.rearrange.rearrange_task
from habitat.utils.geometry_utils import quaternion_from_two_vectors
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import euler_to_quat, write_gfx_replay
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import orthonormalize_rotation_shear

from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations.utils import images_to_video
import habitat_sim
# sys.path.append("/home/catkin_ws/src/")
from get_trajectory_rvo import *
# sys.path.remove("/home/catkin_ws/src/")
# sys.path.append("/root/miniconda3/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/opt/conda/envs/robostackenv/lib/python3.9/site-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages/")
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64, Int32MultiArray
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import tf
import tf2_ros
import threading
from gym import spaces
from collections import OrderedDict
import imageio
import struct
from nav_msgs.msg import Path

from habitat.core.simulator import Observations

from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
import torch
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.agents.simple_agents import GoalFollower
import csv
from IPython import embed
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
THIRD_RGB_SIZE = 128
def traj_interp(c):
    d = c.astype(int)
    iter = len(d) - 1
    added = 0
    i = 0
    while i < iter:
        while np.sqrt((d[i+added,0]-d[i+1+added,0])**2 + (d[i+added,1]-d[i+1+added,1])**2) > np.sqrt(1):
            d = np.insert(d, i+added+1, [0, 0], axis=0)
            if d[i+added+2, 0] - d[i+added, 0] > 0:
                d[i+added+1, 0] = d[i+added, 0] + 1
                d[i+added+1, 1] = d[i+added, 1]
            elif d[i+added+2, 0] - d[i+added, 0] < 0:
                d[i+added+1, 0] = d[i+added, 0] - 1
                d[i+added+1, 1] = d[i+added, 1]
            else:
                d[i+added+1, 0] = d[i+added, 0]
                if d[i+added+2, 1] - d[i+added, 1] > 0:
                    d[i+added+1, 1] = d[i+added, 1] + 1
                elif d[i+added+2, 1] - d[i+added, 1] < 0:
                    d[i+added+1, 1] = d[i+added, 1] - 1
                else:
                    d[i+added+1, 1] = d[i+added, 1]
            added += 1
        i += 1
    return np.array(d)
def to_grid(pathfinder, points, grid_dimensions):
    map_points = maps.to_grid(
                        points[2],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    return ([map_points[1]*0.025, map_points[0]*0.025])

def from_grid(pathfinder, points, grid_dimensions):
    floor_y = 0.0
    map_points = maps.from_grid(
                        points[1],
                        points[0],
                        grid_dimensions,
                        pathfinder=pathfinder,
                    )
    map_points_3d = np.array([map_points[1], floor_y, map_points[0]])
    # # agent_state.position = np.array(map_points_3d)  # in world space
    # # agent.set_state(agent_state)
    map_points_3d = pathfinder.snap_point(map_points_3d)
    return map_points_3d

def img_to_world(proj, cam, W,H, u, v, debug = False):
    K = proj
    T_world_camera = cam
    rotation_0 = T_world_camera[0:3,0:3]
    translation_0 = T_world_camera[0:3,3]
    uv_1=np.array([[u,v,1]], dtype=np.float32)
    uv_1=np.array([[2*u/W -1,-2*v/H +1,1]], dtype=np.float32)
    uv_1=np.array([[2*v/H -1,-2*u/W +1,1]], dtype=np.float32)
    uv_1=uv_1.T
    assert(W == H)
    if (debug):
        embed()
    inv_rot = np.linalg.inv(rotation_0)
    A = np.matmul(np.linalg.inv(K[0:3,0:3]), uv_1)
    A[2] = 1
    t = np.array([translation_0])
    c = (A-t.T)
    d = inv_rot.dot(c)
    return d

def world_to_img(proj, cam, agent_state, W, H, debug = False):
    K = proj
    T_cam_world = cam
    pos = np.array([agent_state[0], agent_state[1], agent_state[2], 1.0])
    projection = np.matmul(T_cam_world, pos)
    # projection = np.array([projection[0], projection[2], projection[1], 1.0])
    image_coordinate = np.matmul(K, projection)
    if (debug):
        embed()
    image_coordinate = image_coordinate/image_coordinate[2]
    v = H-(image_coordinate[0]+1)*(H/2)
    u = W-(1-image_coordinate[1])*(W/2)
    
    return [int(u),int(v)]

def grid_coord_to_raw(coord, img_res, grid_img_res):
    factor = grid_img_res/img_res
    # factor = 1/factor
    return [int((60-coord[0])*factor), int((60-coord[1])*factor)]

def raw_to_grid(coord, img_res, grid_img_res):
    factor = grid_img_res/img_res
    # factor = 1/factor
    return [60-int(coord[0]/factor), 60-int(coord[1]/factor)]

class Hallucinate():
    def __init__(self, config):
        self.env = habitat.Env(config = config)
        remove_ep_list = [0,1,2,8]
        self.observations = self.env.reset()
        # while self.env.current_episode.episode_id in remove_ep_list:
        #     self.observations = self.env.reset()
        meters_per_pixel =0.025
        map_name = "sample_map"
        hablab_topdown_map = maps.get_topdown_map(
                self.env._sim.pathfinder, 0.0, meters_per_pixel=meters_per_pixel
            )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        floor_y = 0.0
        self.top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.third_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
        self.third_camera.render_camera.projection_matrix = mn.Matrix4([
            [0.3000000059604645, 0, 0, 0],
            [0, 0.3000000059604645, 0, 0],
            [0, 0, -0.002000020118430257, 0],
            [0, 0, -1.0000200271606445, 1]
        ])
        
        # self.new_img = np.asarray(self.top_down_map)
        # self.new_img = cv2.cvtColor(self.new_img,cv2.COLOR_GRAY2RGB)  
        self.new_img = hablab_topdown_map      
        self.proj = np.array(self.third_camera.render_camera.projection_matrix)
        self.cam = np.array(self.third_camera.render_camera.camera_matrix)
        world_coord = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = 0, v = 0)
        # head_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_head_rgb']
        world_coord_1 = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = THIRD_RGB_SIZE, v = THIRD_RGB_SIZE)
        self.img_res_1 = abs(world_coord_1[0] - world_coord[0]+1)/THIRD_RGB_SIZE
        self.img_res_2 = abs(world_coord_1[2] - world_coord[2]+1)/THIRD_RGB_SIZE
        self.img_res = (self.img_res_1+self.img_res_2)/2
        grid_img_res_1 = abs(world_coord_1[0] - world_coord[0]+1)/60
        grid_img_res_2 = abs(world_coord_1[2] - world_coord[2]+1)/60
        self.grid_img_res = (grid_img_res_1+grid_img_res_2)/2
        self.initial_state = []
        self.number_of_agents = len(self.env.sim.agents_mgr)
        self.objs = []
        self.grid_dimensions = (self.top_down_map.shape[0], self.top_down_map.shape[1])
        agents_goal_pos_3d = [self.env.current_episode.info['robot_goal'], self.env.current_episode.info['human_goal']]
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            print(start_pos)
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            # agents_goal_pos_3d = [self.env.current_episode.info['human_start']]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
        self.linear_velocity = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.objs[0].base_rot = self.env.current_episode.start_rotation[2]
        try:
            self.objs[1].base_rot = self.env.current_episode.info['human_rot'][2]
        except:
            self.objs[1].base_rot = 0.0
        self.failed_list = []
        print("started epsiode")
        
    def read_data(self, folder, noise_counter):
        img = self.observations["agent_1_third_rgb"][:,:,:3]
        old_image = Image.open(folder+"/raw_img.png")
        old_image = old_image.resize((THIRD_RGB_SIZE,THIRD_RGB_SIZE))
        both_img = np.concatenate((np.asarray(old_image), img), axis = 1)
        im = Image.fromarray(np.uint8(both_img))
        im.save(folder+"/old_obs.png")
        self.image_fol = folder
        with open(self.image_fol+"/human_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        human_past_traj = full_traj
        

        with open(self.image_fol+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        
        robot_past_traj = full_traj

        with open(self.image_fol+"/heading.npy", 'rb') as f:
            full_traj = np.load(f)
        heading = full_traj
        robot_pos = np.array([robot_past_traj[-1,0], robot_past_traj[-1,1]])
        human_pos = np.array([human_past_traj[-1,0], human_past_traj[-1,1]])
        raw_coord_robot = grid_coord_to_raw(robot_pos, self.img_res, self.grid_img_res)
        robot_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = raw_coord_robot[1], v = raw_coord_robot[0])
        robot_pos_world[1] = 0.0
        self.objs[0].base_pos = mn.Vector3([robot_pos_world[0], robot_pos_world[1], robot_pos_world[2]])
        self.objs[0].base_rot = heading[0]
        raw_coord_human = grid_coord_to_raw(human_pos, self.img_res, self.grid_img_res)
        human_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = raw_coord_human[1], v = raw_coord_human[0])
        human_pos_world[1] = 0.0
        self.objs[1].base_pos = mn.Vector3([human_pos_world[0], human_pos_world[1], human_pos_world[2]])
        self.objs[1].base_rot = heading[1]
        # base_vel = [0.0, 0.0]
        # self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
        self.observations.update(self.env._sim.get_sensor_observations())
        img = self.observations["agent_1_third_rgb"][:,:,:3]
        both_img = np.concatenate((np.asarray(old_image), img), axis = 1)
        im = Image.fromarray(np.uint8(both_img))
        im.save(folder+"/new_obs"+str(noise_counter)+".png")
        return im
        

    def get_trajectory(self, folder, data_dir, noise_counter):
        img = self.observations["agent_1_third_rgb"][:,:,:3]
        self.image_fol = folder
        with open(self.image_fol+"/human_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        human_past_traj = full_traj
        

        with open(self.image_fol+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        
        robot_past_traj = full_traj

        with open(self.image_fol+"/heading.npy", 'rb') as f:
            full_traj = np.load(f)
        heading = full_traj
        robot_pos = np.array([robot_past_traj[-1,0], robot_past_traj[-1,1]])
        human_pos = np.array([human_past_traj[-1,0], human_past_traj[-1,1]])
        raw_coord_robot = grid_coord_to_raw(robot_pos, self.img_res, self.grid_img_res)
        robot_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = raw_coord_robot[1], v = raw_coord_robot[0])
        robot_pos_world[1] = 0.0
        self.objs[0].base_pos = mn.Vector3([robot_pos_world[0], robot_pos_world[1], robot_pos_world[2]])
        self.objs[0].base_rot = heading[0]
        raw_coord_human = grid_coord_to_raw(human_pos, self.img_res, self.grid_img_res)
        human_pos_world = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = raw_coord_human[1], v = raw_coord_human[0])
        human_pos_world[1] = 0.0
        self.objs[1].base_pos = mn.Vector3([human_pos_world[0], human_pos_world[1], human_pos_world[2]])
        self.objs[1].base_rot = heading[1]
        file = open(self.image_fol+ '/new_crossing_count.txt', 'r')
        counter_crossing_data = file.read().split('\n')
        number_of_stops = len(counter_crossing_data)
        current_fol_number = int(self.image_fol.split('/')[-1])
        # print("Number of stops ", number_of_stops, counter_crossing_data)
        for counter_crossing in counter_crossing_data:
            counter_crossing = int(counter_crossing)
            if counter_crossing >= current_fol_number:
                break
        counter_fol = data_dir+"/"+self.image_fol.split('/')[-2] + '/' + str(counter_crossing)
        print("counter_fol", counter_fol)
        robot_past_at_crossing = np.load(counter_fol+"/robot_past_traj.npy")
        human_past_at_crossing = np.load(counter_fol+"/human_past_traj.npy")
        heading_at_crossing = np.load(counter_fol+"/heading.npy")
        robot_pos_at_crossing = np.array([robot_past_at_crossing[-1,0], robot_past_at_crossing[-1,1]])
        human_pos_at_crossing = np.array([human_past_at_crossing[-1,0], human_past_at_crossing[-1,1]])
        raw_coord_robot_at_crossing = grid_coord_to_raw(robot_pos_at_crossing, self.img_res, self.grid_img_res)
        robot_pos_world_at_crossing = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = raw_coord_robot_at_crossing[1], v = raw_coord_robot_at_crossing[0])    
        robot_pos_world_at_crossing[1] = 0.0
        sampled_new_pos = self.env._sim.pathfinder.get_random_navigable_point_near(robot_pos_world, radius = 0.5)
        radius = 0.5
        while np.isnan(sampled_new_pos[0]):
            sampled_new_pos = self.env._sim.pathfinder.get_random_navigable_point_near(robot_pos_world, radius = radius)
            radius +=0.1
        self.objs[0].base_pos = mn.Vector3([sampled_new_pos[0], sampled_new_pos[1], sampled_new_pos[2]])
        self.observations.update(self.env._sim.get_sensor_observations())
        img = self.observations["agent_1_third_rgb"][:,:,:3]
        img[raw_coord_robot_at_crossing[1]-2:raw_coord_robot_at_crossing[1]+2, raw_coord_robot_at_crossing[0]-2:raw_coord_robot_at_crossing[0]+2] = [255,0,0]
        Image.fromarray(np.uint8(img)).save(folder+"/new_goal"+str(noise_counter)+".png")
        new_robot_traj = []
        new_robot_traj_raw = []
        im_array = []
        path = habitat_sim.ShortestPath()
        path.requested_start = self.objs[0].base_pos
        path.requested_end = robot_pos_world_at_crossing
        found_path = self.env._sim.pathfinder.find_path(path)
        points_outside = True
        max_tries = 50
        trial_counter = 0
        while not found_path or points_outside:
            sampled_new_pos = self.env._sim.pathfinder.get_random_navigable_point_near(robot_pos_world, radius = radius)
            path.requested_start = sampled_new_pos
            found_path = self.env._sim.pathfinder.find_path(path)
            for point in path.points:
                path_point_raw_image = world_to_img(proj = self.proj, cam = self.cam, agent_state = [point[0], point[1], point[2]], W = img.shape[0], H = img.shape[1])
                new_robot_traj_raw.append(path_point_raw_image)
                new_robot_traj.append(raw_to_grid(path_point_raw_image, self.img_res, self.grid_img_res))
                try:
                    img[path_point_raw_image[0], path_point_raw_image[1]] = [0,255,0]
                except:
                    points_outside = True
                    break
                points_outside = False
            trial_counter += 1
            if trial_counter > max_tries:
                self.failed_list.append(folder)
                return Image.fromarray(np.uint8(img)), Image.fromarray(np.uint8(img))
        # new_robot_traj_raw = np.array(new_robot_traj_raw)
        # new_robot_traj_raw = traj_interp(new_robot_traj_raw)
        # new_robot_traj = np.array(new_robot_traj)
        # new_robot_traj = traj_interp(new_robot_traj)
        # for point in new_robot_traj_raw:
        #     img[point[0], point[1]] = [0,0,255]
        # print(new_robot_traj)
        # Image.fromarray(np.uint8(img)).save(folder+"/path.png")
        # return Image.fromarray(np.uint8(img))
        self.objs[0].base_pos = mn.Vector3([sampled_new_pos[0], sampled_new_pos[1], sampled_new_pos[2]])
        robot_pos_world_at_crossing = path.points[-1]
        steps = 0
        k = 'agent_0_oracle_nav_randcoord_action'
        self.observations.update(self.env._sim.get_sensor_observations())
        img = self.observations["agent_1_third_rgb"][:,:,:3]
        robot_traj_grid = []
        while not self.env.task.actions[k].skill_done and not self.env.episode_over:
            robot_pos_now_world = self.objs[0].base_pos
            self.env.task.actions[k].coord_nav = robot_pos_world_at_crossing
            self.observations.update(self.env.step({"action": k, "action_args": {"agent_0_oracle_nav_randcoord": robot_pos_world_at_crossing}}))
            robot_pos_now = world_to_img(proj = self.proj, cam = self.cam, agent_state = [robot_pos_now_world[0], robot_pos_now_world[1], robot_pos_now_world[2]], W = img.shape[0], H = img.shape[1])
            point = robot_pos_now
            if point[0] <0 and point[0]>-5:
                point[0] = 0
            if point[1] <0 and point[1]>-5:
                point[1] = 0
            if point[0] >= THIRD_RGB_SIZE and point[0] <THIRD_RGB_SIZE+5:
                point[0] = THIRD_RGB_SIZE-1
            if point[1] >= THIRD_RGB_SIZE and point[1] <THIRD_RGB_SIZE+5:
                point[1] = THIRD_RGB_SIZE-1
            robot_pos_now = point
            img[robot_pos_now[0], robot_pos_now[1]] = [255,0,0]

            im_array.append(self.observations["agent_1_third_rgb"][:,:,:3])
            robot_pos_grid = raw_to_grid(robot_pos_now, self.img_res, self.grid_img_res)
            robot_traj_grid.append(robot_pos_grid)
            steps += 1
        Image.fromarray(np.uint8(img)).save(folder+"/raw_img_noise_overlay"+str(noise_counter)+".png")
        print("steps", steps)
        imageio.mimsave(folder+"/noise"+str(noise_counter)+".gif", im_array)
        self.env.task.actions[k].skill_done = False
        if self.env.episode_over:
            print("Episode over")
            self.observations = self.env.reset()
        robot_traj_grid = np.array(robot_traj_grid)
        robot_traj_grid = traj_interp(robot_traj_grid)
        FIXED_LEN = 20
        if len(robot_traj_grid) <FIXED_LEN:
            for i in range(FIXED_LEN - len(robot_traj_grid)):
                robot_traj_grid = np.insert(robot_traj_grid, len(robot_traj_grid), robot_traj_grid[-1], axis=0)
        robot_traj_grid = robot_traj_grid[:FIXED_LEN]
        grid_img = np.array(Image.open(self.image_fol+"/grid_map.png"))[:,:,0:3]
        for point in robot_traj_grid:
            if point[0] <0:
                point[0] = 0
            if point[1] <0:
                point[1] = 0
            if point[0] >= 60:
                point[0] = 59
            if point[1] >= 60:
                point[1] = 59
            grid_img[point[0], point[1]] = [0,0,255]
        grid_img[point[0], point[1]] = [255,0,0]
        Image.fromarray(np.uint8(grid_img)).save(folder+"/grid_map_noise_overlay"+str(noise_counter)+".png")
        with open(folder+"/robot_noise_traj"+str(noise_counter)+".npy", 'wb') as f:
            np.save(f, robot_traj_grid)
        
        metrics = self.env.get_metrics()
        if metrics['did_collide']:
            print("Collided")
            imageio.mimsave(folder+"/noise_collision"+str(noise_collision)+".gif", im_array)
            self.failed_list.append(folder)
        # self.observations = self.env.reset()
        return img, grid_img
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=True)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--save-actions", action="store_true", default=False)
    parser.add_argument(
        "--save-actions-fname", type=str, default="play_actions.txt"
    )
    parser.add_argument(
        "--save-actions-count",
        type=int,
        default=200,
        help="""
            The number of steps the saved action trajectory is clipped to. NOTE
            the episode must be at least this long or it will terminate with
            error.
            """,
    )
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=512, width=512, 
                        orientation =[-1.519, 0.0, 0.0], position = [0, 2.39, 0]
                    )
                }
            )
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"
        if task_config.type == "RearrangePddlTask-v0":
            task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
    
    my_env = Hallucinate(config)
    data_dir = "/home/catkin_ws/src/habitat_ros_interface/data/training/irl_sept_24_3_new_cross/train"
    out_dir = "/home/catkin_ws/src/habitat_ros_interface/data/training/irl_sept_24_3_noise_1/train"
    NUMBER_OF_EXTRA_DEMOS = 5
    for counter in range(NUMBER_OF_EXTRA_DEMOS):
        demos = os.listdir(data_dir)
        demos = [ x for x in demos if x[5:].isdigit()]
        demos.sort(key=lambda x:int(x[5:]))
        for demo in demos:
            items = os.listdir(data_dir+"/"+demo)
            items = [ x for x in items if x.isdigit() ]
            items.sort(key=lambda x:int(x))
            im_array = []
            steps = 0
            if not (os.path.exists(data_dir+"/"+demo+"/0"+"/new_crossing_count.txt")):
                continue
            for item in items:
                folder = data_dir+"/"+demo+"/"+item
                
                print("reading folder", folder)
                im = my_env.read_data(folder, counter)
                
                im0, im = my_env.get_trajectory(folder, data_dir, counter)
                im_array.append(im)
                # steps += 1
                # print("Steps are !!!!", steps )
                # if steps >20:
                #     my_env.observations = my_env.env.reset()
            print("Failed list", my_env.failed_list)

            # while len(my_env.failed_list) > 0:
            #     for folder in my_env.failed_list:
            #         my_env.failed_list.remove(folder)
            #         im0, im = my_env.get_trajectory(folder, data_dir)
            #         im_array.append(im)
                    
            imageio.mimsave(data_dir+"/"+demo+"/new_paths"+str(counter)+".gif", im_array)
