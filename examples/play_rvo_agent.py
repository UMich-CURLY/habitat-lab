#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
from torch.utils.tensorboard import SummaryWriter
import torch
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
from tf.transformations import euler_from_quaternion
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
import json
from IPython import embed
# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "/habitat-lab/data/vids/Test_data_finalest/Only8/exp_8.12"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"
MAP_DIR = "/home/catkin_ws/src/habitat_ros_interface/maps/"
THIRD_RGB_SIZE = 128
GRID_SIZE = 6
HUMAN_HEAD_START = 0       #60
CSV_PATH = "/habitat-lab/data/vids/Test_data_finalest/Only8/exp_8.12/results.csv"

USE_TOPO_MAP = True
USE_RL_CONTROL = False
USE_CLICKED_POINT = True
HUMAN_RVO = True
HUMAN_SFM = False
USE_INSTANT_VEL = True
USE_IRL_AGENT = False
SAVE_DATA = True
USE_CLICKED_POINT_IRL = False
RANDOM_AGENT = False
ALLOW_BACKOFF = False
NO_ROBOT = False
lock = threading.Lock()

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


class sim_env(threading.Thread):
    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 10
    
    _current_episode = 0
    _total_number_of_episodes = 0
    
    replan_freq = 1
    replan_counter = 0
    def __init__(self, config):
        threading.Thread.__init__(self)
        # Initialize TensorBoard writer（每个 episode 一个子目录，在 TensorBoard 里选不同 run 查看）
        # 若 Images 滑块在高 step 时“一跳很大”，请用：tensorboard --logdir runs --samples_per_plugin images=0
        self.writer = SummaryWriter('runs/habitat_visualization/episode_0')
        # self.writer = SummaryWriter('runs/habitat_visualization')
        self.step = 0
        self.env = habitat.Env(config = config)
        # remove_ep_list = [0,1,2,8]
        remove_ep_list = [1,2,8]
        self.observations = self.env.reset()
        while self.env.current_episode.episode_id in remove_ep_list:
            self.observations = self.env.reset()
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
        self.new_img = np.asarray(hablab_topdown_map)
        self.top_down_map = maps.get_topdown_map(
            self.env._sim.pathfinder, height=floor_y, meters_per_pixel=0.025
        )
        self.grid_dimensions = (self.top_down_map.shape[0], self.top_down_map.shape[1])

        imageio.imsave(os.path.join(MAP_DIR, map_name + ".pgm"), hablab_topdown_map)
        print("writing Yaml file! ")
        complete_name = os.path.join(MAP_DIR, map_name + ".yaml")
        f = open(complete_name, "w+")
        f.write("image: " + map_name + ".pgm\n")
        f.write("resolution: " + str(meters_per_pixel) + "\n")
        f.write("origin: [" + str(-1) + "," + str(-self.grid_dimensions[0]*meters_per_pixel+1) + ", 0.000000]\n")
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196")
        f.close()
        if NO_ROBOT:
            self.env.sim.agents_mgr[0].articulated_agent.base_pos = self.env.sim.pathfinder.get_random_navigable_point()
        rospy.init_node("sim", anonymous=False)
        sim_cfg = config['habitat']['simulator']
        self.control_frequency = int(np.floor(sim_cfg['ctrl_freq']/sim_cfg['ac_freq_ratio']))
        time_step = 1.0 / (self.control_frequency)
        self.control_frequency = 5.0
        self._r = rospy.Rate(self._sensor_rate)
        self._r_control = rospy.Rate(self.control_frequency)
        
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_rgb_2 = rospy.Publisher("~rgb2", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._robot_pose = rospy.Publisher("~robot_pose", PoseStamped, queue_size = 1)
        self._pub_all_agents = rospy.Publisher("~agent_poses", PoseArray, queue_size = 1)
        self._pub_door = rospy.Publisher("~door", MarkerArray, queue_size = 1)
        self._pub_goal_marker = rospy.Publisher("~goal", Marker, queue_size = 1)
        self._pub_rvo_goal_marker = rospy.Publisher("~rvo_goal", Marker, queue_size = 1)
        self._pub_goal_marker_human = rospy.Publisher("~human_goal", Marker, queue_size = 1)
        self.cloud_pub = rospy.Publisher("top_down_img", PointCloud2, queue_size=2)
        self._pub_img_res = rospy.Publisher("img_res", Float64, queue_size= 1)
        self._reload_map_server = rospy.Publisher("reload_map_server", Bool, queue_size= 1)
        # self._sub_wait = rospy.Subscriber("wait_for_traj", Bool, self.start_wait, queue_size = 1)
        self._pub_start_ep = rospy.Publisher("start_ep", Bool, queue_size= 1)
        self._pub_get_traj = rospy.Publisher("query_irl", Bool, queue_size= 1)
        if USE_IRL_AGENT:
            self._sub_path = rospy.Subscriber("irl_path", Path, self.get_path,  queue_size=1)
            self._sub_cloud = rospy.Subscriber("got_cloud", Bool, self.got_cloud, queue_size=1)
        # self.sub_traj = rospy.Subscriber("irl_traj", Int32MultiArray, self.get_irl_traj, queue_size = 1)
        self.br = tf.TransformBroadcaster()
        self.br_tf_2 = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,self.pose_callback, queue_size=1)
        self.third_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
        self.third_camera.render_camera.projection_matrix = mn.Matrix4([
            [0.3000000059604645, 0, 0, 0],
            [0, 0.3000000059604645, 0, 0],
            [0, 0, -0.002000020118430257, 0],
            [0, 0, -1.0000200271606445, 1]
        ])
        self.img_res = None
        # self.new_img = np.asarray(self.top_down_map)
        # self.new_img = cv2.cvtColor(self.new_img,cv2.COLOR_GRAY2RGB)  
        self.new_img = hablab_topdown_map      
        self.proj = np.array(self.third_camera.render_camera.projection_matrix)
        self.cam = np.array(self.third_camera.render_camera.camera_matrix)
        self.initial_state = []
        self.number_of_agents = len(self.env.sim.agents_mgr)
        self.objs = []
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
        self.grid_size_in_m = GRID_SIZE
        self.grid_resolution = 0.15
        self.grid_dimension = self.grid_size_in_m/self.grid_resolution
        sim_config = habitat.get_config("/habitat-lab/habitat-baselines/habitat_baselines/config/social_nav/social_nav_fetch_test.yaml")
        agent_config = sim_config.habitat_baselines
        ppo = baseline_registry.get_trainer(agent_config.trainer_name)
        checkpoint_path = agent_config.eval_ckpt_path_dir
        
        self.ppo = ppo(sim_config)
        ckpt_dict = self.ppo.load_checkpoint(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        if torch.cuda.is_available():
            self.device = torch.device("cuda", agent_config.torch_gpu_id)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.device = torch.device("cpu") 
        # actor_critic = PointNavResNetPolicy(
        #          observation_space=self.env.observation_space,
        #          action_space=self.env.action_space,
        #          hidden_size=512,
        #      )
        action_space = ActionSpace({"agent_0_base_velocity": self.env.action_space["agent_0_base_velocity"]})
        self.use_names = [x for x in config.habitat.gym.obs_keys if x.startswith("agent_0")]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in self.env.observation_space.items()
                    if k in self.use_names 
                )
            )
        )
        self.actor_critic = PointNavResNetPolicy.from_config(config = ckpt_dict["config"], observation_space = filtered_obs, action_space = action_space, agent_name  = "agent_0")
        
        self.actor_critic.load_state_dict(
                     {  # type: ignore
                         k : v
                         for k, v in ckpt_dict[0]["state_dict"].items()
                     }
                 )
        select_observations = {}
        for names in self.use_names:
            select_observations[names] = self.observations[names]
        batch = batch_obs([select_observations], device=self.device)
        self.obs_transform = get_active_obs_transforms(ckpt_dict["config"], agent_name = "agent_0")
        batch = apply_obs_transforms_batch(batch, self.obs_transform)
        self.current_episode_reward = torch.zeros(
            1, 1, device="cpu"
        )
        ppo_cfg = ckpt_dict["config"].habitat_baselines.rl.ppo
        action_shape = (1,)
        discrete_actions = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            1,
            2,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        self.not_done_masks = torch.zeros(
            1,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        self._reload_map_server.publish(True)
        self.goal_agent = GoalFollower(
                0.2,
                "dist_to_goal",
            )
        
        self.start_ep = False
        self.waiting_for_traj = True
        self.current_point = None
        self.wait_counter = 0
        self.sfm = ped_rvo(self, map_path = MAP_DIR+"sample_map.pgm", resolution = 0.025)
        self.current_point_2d = self.initial_state[0][4:6]
        self.line = []
        self.obs = []
        self.human_goal_2d = self.initial_state[1][4:6]
        self.cheating_point = None
        self.drift_counter = 0
        self.actual_num_steps = 0
        self.prev_query_time = rospy.Time.now()
        self.got_sem_cloud = False
        self.cloud_wait_counter = 0
        print("Starting the work now")
        self.im_array = []
        self._current_episode = 0
        self.clicked_point = None
        self.prev_positions = None
        self.counter_deadlock = 0
        self.backoff_mode = False
        self.robot_moved = []
        self.human_moved = []
        self.human_num_steps = 0
        self.full_init_state = []
        self.hack_to_save = None
        self.time_between_saves = rospy.Time.now().to_sec()
        
        # 用于存储自定义俯视图的pixel点（可以在tensorboard上点击获取）
        # 格式: [[pixel_x1, pixel_y1], [pixel_x2, pixel_y2]]
        # 如果为None，则使用默认的门的位置
        self.custom_topdown_pixel_points = [[300,231],[300,243]]  # 可以设置为例如: [[370, 167], [400, 200]]
        
        # 缓存新的俯视图图像（在主线程中更新，在run线程中使用）
        self.cached_topdown_new_img = None
        self.topdown_update_counter = 0  # 用于控制更新频率
        self.topdown_update_freq = 5  # 每5帧更新一次

        # self.sfm.get_velocity(self.initial_state, filename = MAP_DIR+"run_rvo2", save_anim = True)

    def reset(self):
        #### Save the results of the previous episode ####
        if SAVE_DATA:
            metrics = self.env.get_metrics()
            results_dict = {}
            results_dict["num_steps"] = metrics["num_steps"]
            results_dict["did_collide"] = metrics["did_collide"]
            results_dict["robot_scene_collision"] = metrics["robot_collisions"]["robot_scene_colls"]
            results_dict["social_nav_to_pos_success"] = metrics["social_nav_to_pos_success"]
            results_dict["social_dist_to_goal"] = metrics["social_dist_to_goal"]
            results_dict["avg_robot_to_human_dis_over_epi"] = metrics["social_nav_stats"]["avg_robot_to_human_dis_over_epi"]
            results_dict["social_nav_reward"] = metrics["social_nav_reward"]
            results_dict["actual_num_steps"] = self.actual_num_steps   
            results_dict["human_num_steps"] = self.human_num_steps
            results_dict["ep_no"] =  self.env.current_episode.episode_id
            results_dict["drift_counter"] = self.drift_counter
            results_dict["USE_HUMAN_RVO"] = HUMAN_RVO
            results_dict["USE_TOPO_MAP"] = USE_TOPO_MAP
            results_dict["USE_RL_CONTROL"] = USE_RL_CONTROL
            results_dict["USE_CLICKED_POINT"] = USE_CLICKED_POINT
            results_dict["USE_INSTANT_VEL"] = USE_INSTANT_VEL
            results_dict["USE_IRL_AGENT"] = USE_IRL_AGENT
            # Create directory if it doesn't exist
            csv_dir = os.path.dirname(CSV_PATH)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            with open(CSV_PATH, "a", newline="") as fp:
            # Create a writer object
                writer = csv.DictWriter(fp, fieldnames=results_dict.keys())
                if self._current_episode == 0:
                    writer.writeheader()
                writer.writerow(results_dict)
        #### Finish writing csv file ####
        if NO_ROBOT:
            self.env.sim.agents_mgr[0].articulated_agent.base_pos = self.env.sim.pathfinder.get_random_navigable_point()

        self.observations = self.env.reset()
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
        self.grid_dimensions = (self.top_down_map.shape[0], self.top_down_map.shape[1])
        imageio.imsave(os.path.join(MAP_DIR, map_name + ".pgm"), hablab_topdown_map)
        print("writing Yaml file! ")
        complete_name = os.path.join(MAP_DIR, map_name + ".yaml")
        f = open(complete_name, "w+")
        f.write("image: " + map_name + ".pgm\n")
        f.write("resolution: " + str(meters_per_pixel) + "\n")
        f.write("origin: [" + str(-1) + "," + str(-self.grid_dimensions[0]*meters_per_pixel+1) + ", 0.000000]\n")
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196")
        f.close()
        self.map_to_base_link({'x': 0, 'y': 0, 'theta': self.get_object_heading(self.env.sim.agents_mgr[0].articulated_agent.base_transformation)})
        
        self.start_ep = False
        if SAVE_DATA:
            self.im_array[0].save(SAVE_VIDEO_DIR + "/episode_" + str(self._current_episode) + ".gif", save_all=True, append_images=self.im_array[1:], duration=100, loop=0)
            self.im_array = []
            init_states_array = np.array(self.full_init_state)
            np.save(SAVE_VIDEO_DIR + "/episode_" + str(self._current_episode) + ".npy", init_states_array)
        self._current_episode += 1

        # 每个 episode 单独一个 TensorBoard run，便于在 Images 里选 episode_0 / episode_1 / ... 查看
        lock.acquire()
        try:
            self.writer.close()
        except Exception:
            pass
        self.writer = SummaryWriter(f'runs/habitat_visualization/episode_{self._current_episode}')
        self.step = 0
        lock.release()

        self.initial_state = []
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            agents_goal_pos_3d = [self.env.current_episode.info['robot_goal'], self.env.current_episode.info['human_goal']]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
        self.objs[0].base_rot = self.env.current_episode.start_rotation[2]
        try:
            self.objs[1].base_rot = self.env.current_episode.info['human_rot'][2]
        except:
            self.objs[1].base_rot = 0.0
        self.sfm.reset_peds(self.initial_state)
        self.got_sem_cloud = False
        self.cloud_wait_counter = 0
        self.actual_num_steps = 0
        self.backoff_mode = False
        self.cheating_point = None
        self.waiting_for_traj = True
        self.counter_deadlock = 0
        # self.step = 0  # 每 episode 重置，避免 TensorBoard Images 滑块在高 step 时“一跳很大”
        self.robot_moved = []
        self.human_moved = []
        self.human_num_steps = 0
        self.full_init_state = []
        self._reload_map_server.publish(True)
        self.time_between_saves = rospy.Time.now().to_sec()
        rospy.sleep(10)

    def img_to_grid(self):
        img = self.observations["agent_1_third_rgb"]

        points = []
        if self.img_res is None:
            return
        for i in range(0,img.shape[0], 1):
            for j in range(0, img.shape[1], 1):
                world_coord = img_to_world(proj = self.proj, cam = self.cam, W = img.shape[0], H = img.shape[1], u = i, v = j)
                world_coord[1] = 0.0
                map_coord = np.array(to_grid(self.env._sim.pathfinder, world_coord, self.grid_dimensions))
                # print([i,j], map_coord)
                map_coord = map_coord + np.array([-1,-1])
                # map_coord = map_coord/0.025
                if (i ==j == 32):
                    print("world coord at origin is :", world_coord)
                    print("map coord at origin is :", map_coord)
                [r,g,b] = img[j,i,:]
                a = 255
                z = 0.1
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [i*self.img_res , j*self.img_res , z, rgb]
                points.append(pt)
        # cv2.imwrite(MAP_DIR+"/overlayed_img.png", self.new_img)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]
        header = Header()
        header.frame_id = "camera_frame"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.cloud_pub.publish(pc2)

    def got_cloud(self, msg):
        self.got_sem_cloud = msg.data

    def act(self) -> Dict[str, int]:
        select_observations = {}
        for names in self.use_names:
            select_observations[names] = self.observations[names]

        batch = batch_obs([select_observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transform)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            self.test_recurrent_hidden_states = action_data.rnn_hidden_states
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)  # type: ignore
        return [action_data.env_actions[0][0].item(), action_data.env_actions[0][1].item()]
    
    def run(self):
        """Publish sensor readings through ROS on a different thread.
            This method defines what the thread does when the start() method
            of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()
            
            # Add TensorBoard visualization
            rgb_img = self.observations["agent_1_head_rgb"][:,:,:3]
            # Convert numpy array to PyTorch tensor and add batch dimension
            rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)
            self.writer.add_images('agent_1_head_rgb', rgb_tensor, self.step)
            # self.new_img = cv2.cvtColor(self.new_img,cv2.COLOR_GRAY2RGB)  
            self.writer.add_images('top_down_map', torch.from_numpy(self.new_img).permute(2, 0, 1).unsqueeze(0), self.step)

            if "full_top_down" in self.observations:
                topdown_img = self.observations["full_top_down"][:,:,:3]
                # Convert numpy array to PyTorch tensor and add batch dimension
                topdown_tensor = torch.from_numpy(topdown_img).permute(2, 0, 1).unsqueeze(0)
                self.writer.add_images('full_top_down_view', topdown_tensor, self.step)
            
            # 添加新的自定义俯视图 full_top_down_view_new
            # 注意：不在这个线程中调用 get_sensor_observations()，因为 OpenGL 上下文不能跨线程使用
            # 使用主线程已经缓存好的图像
            try:
                if self.cached_topdown_new_img is not None:
                    # 转换为tensor并添加到tensorboard
                    topdown_new_tensor = torch.from_numpy(self.cached_topdown_new_img).permute(2, 0, 1).unsqueeze(0)
                    self.writer.add_images('full_top_down_view_new', topdown_new_tensor, self.step)
            except Exception as e:
                # 如果出错，跳过这次显示
                print(f"[Warning] 无法显示 full_top_down_view_new: {e}")
            
            self.step += 1
            
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_1_head_rgb"][:,:,:3].ravel()),
                    np.array(
                        [512,512]
                    ),
                )
            )
            rgb2_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_1_third_rgb"][:,:,:3].ravel()),
                    np.array(
                        [THIRD_RGB_SIZE,THIRD_RGB_SIZE]
                    ),
                )
            )
            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_0_head_depth"].ravel() * 10),
                    np.array(
                        [
                            128,128
                        ]
                    ),
                )
            )       
            cv2.imwrite(MAP_DIR+"/sample_img.png", self.observations["agent_1_third_rgb"])
            # for i in range(self.number_of_agents):
            #     agent_pos = self.objs[i].base_pos
            #     start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            #     initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            #     # agents_goal_pos_3d = [self.env._task.my_nav_to_info.robot_info.nav_goal_pos, self.env._task.my_nav_to_info.human_info.nav_goal_pos]
            #     agents_initial_velocity = [0.5,0.0]
            #     # goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            #     self.initial_state[i][0:4] = initial_pos+agents_initial_velocity
            #     # if self.current_point is None:
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_rgb_2.publish(np.float32(rgb2_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            self.map_to_base_link({'x': self.initial_state[0][0], 'y': self.initial_state[0][1], 'theta': self.get_object_heading(self.env.sim.agents_mgr[0].articulated_agent.base_transformation)})
            lock.release()
            self._r.sleep()

    def start_wait(self, msg):
        self.waiting_for_traj = msg.data
        print("Waiting for traj? ", self.waiting_for_traj)

        
        
    def update_agent_pos_vel(self):
        if (self.env._episode_over):
            print("Done with episode and starting a new one")
            self.reset()
            return

        dist_human_moved = np.linalg.norm(self.objs[1].base_pos -  self.env.current_episode.info['human_start'])
        print("Human has moved ", dist_human_moved)
        if USE_IRL_AGENT:
            if not self.got_sem_cloud:
                self.cloud_wait_counter +=1
                print("Didnt get cloud yet")
                if self.cloud_wait_counter > 30:
                    self._reload_map_server.publish(True)
                    self.cloud_wait_counter = 0
                    print("Didnt get cloud yet")
                    return
                return
            self.start_ep = True
        else:
            self.start_ep = True
        if not self.start_ep:
            base_vel = [0.0, 0.0]
            # print("Caught here ", not self.start_ep,  self.waiting_for_traj , self.current_point is None)
            self.observations.update(self.env._sim.get_sensor_observations())
            # 在主线程中更新缓存的俯视图
            self.update_cached_topdown_new()
            # self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
            return
        self.wait_counter +=1
        while (dist_human_moved<0.2):
            self._pub_start_ep.publish(False)
            print("Giving the human a head start")
            k = 'agent_1_oracle_nav_randcoord_action'
            for i in range(1):
                self.observations.update(self.env.step({"action":k, "action_args":{}}))
                
            dist_human_moved = np.linalg.norm(self.objs[1].base_pos -  self.env.current_episode.info['human_start'])
            self.waiting_for_traj = True
            return
        # computed_velocity = self.sfm.get_velocity(np.array(self.initial_state))
        
        path = habitat_sim.ShortestPath()
        path.requested_start = self.objs[1].base_pos
        path.requested_end = self.env.current_episode.info['human_goal']
        pathfinder = self.env._sim.pathfinder
        found_path = pathfinder.find_path(path)
        if found_path:
            for points in path.points:
                dist_between_human_and_point = np.linalg.norm(points - self.objs[1].base_pos)
                if dist_between_human_and_point > 0.5:
                    break
        self.initial_state[1][4:6] = to_grid(self.env._sim.pathfinder, points, self.grid_dimensions)
        # print("Human goal in 2d update in play agent is ", self.initial_state[1][4:6])
    
        self._pub_start_ep.publish(True)
        self._pub_get_traj.publish(True)
        
        self.prev_query_time = rospy.Time.now()
        
        # if self.cheating_point is not None:
        #     robot_final_goal = self.env.current_episode.info['robot_goal']
        #     dist_to_goal = np.linalg.norm((np.array(robot_final_goal)-np.array(self.current_point))[[0, 2]])
        #     print("Distance to final goal is ", dist_to_goal)
        #     if dist_to_goal <0.8:
        #         self.current_point = robot_final_goal
        #         self.waiting_for_traj = False
        #         print("Setting to robot final goal")
        #     print("Current point is ", self.current_point)
        if USE_IRL_AGENT:
            robot_final_goal = self.env.current_episode.info['robot_goal']
            dist_to_goal = np.linalg.norm((np.array(robot_final_goal)-np.array(self.objs[0].base_pos))[[0, 2]])
            metrics = self.env.get_metrics()
            # dist_to_goal_human = np.linalg.norm((np.array(self.env.current_episode.info['human_goal'])-np.array(self.objs[1].base_pos))[[0,2]])
            # print("Distance to human goal is ", dist_to_goal_human)
            if dist_to_goal <=1.3:
                self.current_point = robot_final_goal
                self.waiting_for_traj = False
            while self.cheating_point is None or self.waiting_for_traj:
                # base_vel = [0.0, 0.0]
                wait_time = (rospy.Time.now()-self.prev_query_time).to_sec()
                if wait_time > 0.5 and self.waiting_for_traj:
                    print("Time between queries is ", (rospy.Time.now()-self.prev_query_time).to_sec())
                    self._pub_get_traj.publish(True)
                    self.prev_query_time = rospy.Time.now()
                    self.drift_counter +=1
                    # dist_to_goal_human = np.linalg.norm((np.array(self.env.current_episode.info['human_goal'])-np.array(self.objs[1].base_pos))[[0,2]])
                    # if dist_to_goal_human <= 0.3:
                    #     self.current_point = self.env.current_episode.info['robot_goal']
                    #     self.waiting_for_traj = False
                    #     print("Setting to robot final goal")
                    #     break
                    return
                if self.drift_counter >20:
                    print("Not sure what to do now")

                    # self._reload_map_server.publish(True)
                    self.drift_counter = 0
                if self.drift_counter >100000:
                    self.reset()
                    return
        if USE_TOPO_MAP:
            path = habitat_sim.ShortestPath()
            path.requested_start = self.objs[0].base_pos
            path.requested_end = self.env.current_episode.info['robot_goal']
            # if self.cheating_point is not None:
            #     path.requested_end = from_grid(self.env._sim.pathfinder, [self.cheating_point[0]/0.025, self.cheating_point[1]/0.025], self.grid_dimensions)     
            pathfinder = self.env._sim.pathfinder
            found_path = pathfinder.find_path(path)
            if found_path:
                for points in path.points:
                    dist_between_human_and_point = np.linalg.norm(points - self.objs[0].base_pos)
                    if dist_between_human_and_point > 0.5:
                        break
                self.initial_state[0][4:6] = to_grid(self.env._sim.pathfinder, points, self.grid_dimensions)
        if USE_CLICKED_POINT:
            if self.cheating_point is not None:
                self.initial_state[0][4:6] = self.cheating_point
        if USE_CLICKED_POINT_IRL:
            if self.clicked_point is not None:
                self.initial_state[0][4:6] = self.clicked_point
        positions = self.sfm.get_future_position(np.array(self.initial_state), num_steps=1)
        if ALLOW_BACKOFF:
            if self.prev_positions is not None and not self.backoff_mode:
                human_final_goal = self.env.current_episode.info['human_goal']
                dist_to_goal = np.linalg.norm((np.array(human_final_goal)-np.array(self.objs[1].base_pos))[[0, 2]])
                if len(self.robot_moved )>50:
                    print("Robot total moved is ", np.sum(self.robot_moved[-50:], axis = 0))
                    print("Human total moved is ", np.sum(self.human_moved[-50:], axis = 0))
                    if (np.sum(self.robot_moved[-50:], axis = 0) < [0.1, 0.1]).all() and (np.sum(self.human_moved[-50:], axis = 0) < [0.1, 0.1]).all():
                        self.counter_deadlock +=1
                        if self.counter_deadlock > 20 and dist_to_goal >0.2:
                            self.backoff_mode = True
                            self.backoff_goal_3d = self.env.current_episode.start_position
                            self.backoff_goal = to_grid(self.env._sim.pathfinder, self.backoff_goal_3d, self.grid_dimensions)
                            self.backing_off_counter = 0
                            print("Backoff mode activated")
                    else:
                        self.counter_deadlock = 0
            self.prev_positions = positions
            if self.backoff_mode:
                print("Backing off ")
                self.initial_state[0][4:6] = self.backoff_goal
                positions = self.sfm.get_future_position(np.array(self.initial_state), num_steps=1)
                
                # if positions[0][0] == self.backoff_goal[0] and positions[0][1] == self.backoff_goal[1]:
                human_final_goal = self.env.current_episode.info['human_goal']
                dist_to_goal = np.linalg.norm((np.array(human_final_goal)-np.array(self.objs[1].base_pos))[[0, 2]])
                if dist_to_goal < 0.2:  
                    self.backoff_mode = False
                    print("Stopping backoff mode")
                    self.counter_deadlock = 0
                self.backing_off_counter+=1

        self.current_point_2d = positions[0]
        if USE_INSTANT_VEL:
            distx = (self.current_point_2d[0] - self.initial_state[0][0])
            disty = (self.current_point_2d[1] - self.initial_state[0][1])
            dist = np.linalg.norm([distx, disty])
            positionx = self.initial_state[0][0] + (distx/dist)*0.5
            positiony = self.initial_state[0][1] + (disty/dist)*0.5
            self.current_point_2d = [positionx, positiony]
        if USE_RL_CONTROL:
            if self.cheating_point is not None:
                self.current_point_2d = self.cheating_point
            if USE_CLICKED_POINT_IRL:
                if self.clicked_point is not None:
                    self.current_point_2d = self.clicked_point

        point_3d = from_grid(self.env._sim.pathfinder, [self.current_point_2d[0]/0.025, self.current_point_2d[1]/0.025], self.grid_dimensions)
        self.current_point = np.array([point_3d[0], point_3d[1], point_3d[2]])
        self.human_goal_2d = positions[1]
        if USE_INSTANT_VEL:
            distx = (self.human_goal_2d[0] - self.initial_state[1][0])
            disty = (self.human_goal_2d[1] - self.initial_state[1][1])
            dist = np.linalg.norm([distx, disty])
            positionx = self.initial_state[1][0] + (distx/dist)*0.5
            positiony = self.initial_state[1][1] + (disty/dist)*0.5
            self.human_goal_2d = [positionx, positiony]
        point_3d = from_grid(self.env._sim.pathfinder, [self.human_goal_2d[0]/0.025,self.human_goal_2d[1]/0.025], self.grid_dimensions)
        self.human_goal_rvo = np.array([point_3d[0], point_3d[1], point_3d[2]])
        # print("Human goal in 2d is ", self.human_goal_2d)
        robot_final_goal = self.env.current_episode.info['robot_goal']
        dist_to_goal = np.linalg.norm((np.array(robot_final_goal)-np.array(self.objs[0].base_pos))[[0, 2]])
        metrics = self.env.get_metrics()
        print("Distance to final goal is ", dist_to_goal == metrics["social_dist_to_goal"])
        # dist_to_goal_human = np.linalg.norm((np.array(self.env.current_episode.info['human_goal'])-np.array(self.objs[1].base_pos))[[0,2]])
        # print("Distance to human goal is ", dist_to_goal_human)
        if dist_to_goal <=1.0:
            self.current_point = robot_final_goal
            self.waiting_for_traj = False
        # print("Current point is ", self.current_point)
        # print("Computed Poistion for robot and human is ", positions)
        # if (self.wait_counter < HUMAN_HEAD_START):
        # while (dist_human_moved<0.2):
        #     print("Giving the human a head start")
        #     positions_here = []
        #     for agent in self.objs:
        #         points = np.array([np.array(agent.base_pos)]).T
        #         positions_here.append(np.array(to_grid(self.env._sim.pathfinder, points, self.grid_dimensions)))
        #     k = 'agent_1_oracle_nav_randcoord_action'
        #     for i in range(2):
        #         self.observations.update(self.env.step({"action":k, "action_args":{}}))
        #     dist_human_moved = np.linalg.norm(self.objs[1].base_pos -  self.env.current_episode.info['human_start'])
        #     positions_now = []
        #     for agent in self.objs: 
        #         points = np.array([np.array(agent.base_pos)]).T
        #         positions_now.append(np.array(to_grid(self.env._sim.pathfinder, points, self.grid_dimensions))) 
        #     dist_moved_robot = np.array(positions_now[0]) - np.array(positions_here[0])
        #     dist_moved_human = np.array(positions_now[1]) - np.array(positions_here[1])
        #     print("Robot moved ", dist_moved_robot, " and human moved ", dist_moved_human)
        #     self.initial_state[0][2:4] = dist_moved_robot/self.sfm.dt
        #     self.initial_state[1][2:4] = dist_moved_human/self.sfm.dt
        #     self.initial_state[0][0:2] = positions_now[0]
        #     self.initial_state[1][0:2] = positions_now[1]
        #     self.sfm.reset_peds(self.initial_state)
        #     return
        # if self.current_point is None:
        #     base_vel = [0.0, 0.0]
        #     print("Caught here ", not self.start_ep,  self.waiting_for_traj , self.current_point is None)
        #     self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
        #     return
        ### When driving agent ####
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        
        ### When RL drives agent ####
        # base_vel = self.act()
        positions_here = []
        for agent in self.objs:
            points = np.array([np.array(agent.base_pos)]).T
            positions_here.append(np.array(to_grid(self.env._sim.pathfinder, points, self.grid_dimensions)))

        for i in range(1):
            if (self.env._episode_over):
                print("Done with episode and starting a new one")
                self.reset()
                return
            self.actual_num_steps +=1
            human_final_goal = self.env.current_episode.info['human_goal']
            dist_to_goal_human = np.linalg.norm((np.array(human_final_goal)-np.array(self.objs[1].base_pos))[[0, 2]])
            # print("Distance to human goal is!!!! ", dist_to_goal_human)
            if dist_to_goal_human > 0.2:
                self.human_num_steps +=1
            else:
                self.human_goal_rvo = human_final_goal
            k = 'agent_1_oracle_nav_randcoord_action'
            if HUMAN_RVO:
                self.env.task.actions[k].coord_nav = self.human_goal_rvo
            # my_env.env.task.actions[k].coord_nav = self.observations['agent_0_localization_sensor'][:3]
                self.observations.update(self.env.step({"action":k, "action_args":{"agent_1_oracle_nav_randcoord_action":self.human_goal_rvo}}))
            else:
                self.observations.update(self.env.step({"action":k, "action_args":{}}))
            k = 'agent_0_oracle_nav_randcoord_action'
            if metrics["social_dist_to_goal"] <1.0:
                self.current_point = robot_final_goal
                self.waiting_for_traj = False
                print("Setting to robot final goal")
            self.env.task.actions[k].coord_nav = self.current_point### Read the point from the traj
            
            coord_nav = self.current_point
            # coord_nav = np.array([point_3d[0], point_3d[1], point_3d[2]])
            # print("Coord nav here is ", coord_nav)
            # print("Agent currently at ", self.env.sim.agents_mgr[0].articulated_agent.base_pos)
            if (self.env._episode_over):
                print("Done with episode and starting a new one")
                self.reset()
                return
            if not NO_ROBOT:
                self.observations.update(self.env.step({"action":k, "action_args":{"agent_0_oracle_nav_randcoord_action":coord_nav}}))
            # 在主线程中更新缓存的俯视图（每几帧更新一次）
            self.update_cached_topdown_new()
            robot_pos_in_img = world_to_img(proj = self.proj, cam = self.cam, W = self.observations["agent_1_head_rgb"].shape[0], H = self.observations["agent_1_head_rgb"].shape[1], agent_state = self.objs[0].base_pos)
            human_pos_in_img = world_to_img(proj = self.proj, cam = self.cam, W = self.observations["agent_1_head_rgb"].shape[0], H = self.observations["agent_1_head_rgb"].shape[1], agent_state = self.objs[1].base_pos)
            robot_goal_in_img = world_to_img(proj = self.proj, cam = self.cam, W = self.observations["agent_1_head_rgb"].shape[0], H = self.observations["agent_1_head_rgb"].shape[1], agent_state = self.env.current_episode.info['robot_goal'])
            self.full_init_state.append([robot_pos_in_img, human_pos_in_img, robot_goal_in_img, [self.hack_to_save, float(rospy.Time.now().to_sec()-self.time_between_saves)]])
            self.time_between_saves = rospy.Time.now().to_sec()
            self.im_array.append(Image.fromarray(self.observations["agent_1_third_rgb"].astype(np.uint8)))
        # self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
        if (self.actual_num_steps%10 == 0):
            self.waiting_for_traj = True
        positions_now = []
        for agent in self.objs: 
            points = np.array([np.array(agent.base_pos)]).T
            positions_now.append(np.array(to_grid(self.env._sim.pathfinder, points, self.grid_dimensions))) 
        dist_moved_robot = np.array(positions_now[0]) - np.array(positions_here[0])
        dist_moved_human = np.array(positions_now[1]) - np.array(positions_here[1])
        print("Robot moved ", dist_moved_robot, " and human moved ", dist_moved_human)
        self.robot_moved.append(dist_moved_robot)
        self.human_moved.append(dist_moved_human)
        self.initial_state[0][2:4] = dist_moved_robot*self.control_frequency
        self.initial_state[1][2:4] = dist_moved_human*self.control_frequency
        self.initial_state[0][0:2] = positions_now[0]
        self.initial_state[1][0:2] = positions_now[1]
        self.sfm.reset_peds(self.initial_state)
        
    def get_object_heading(self,obj_transform):
        a = obj_transform
        b = a.transform_point([0.5,0.0,0.0])
        d = a.transform_point([0.0,0.0,0.0])
        c = np.array(to_grid(self.env._sim.pathfinder, [b[0],b[1],b[2]], self.grid_dimensions))
        e = np.array(to_grid(self.env._sim.pathfinder, [d[0],d[1],d[2]], self.grid_dimensions))
        vel = (c-e)*(0.5/np.linalg.norm(c-e)*np.ones([1,2]))[0]
        return mn.Rad(np.arctan2(vel[1], vel[0]))

    def pub_door(self, debug = False):
        door_start_3d = self.env.current_episode.info['door_start']
        door_end_3d = self.env.current_episode.info['door_end']
        door_start_2d = np.array(to_grid(self.env._sim.pathfinder, door_start_3d, self.grid_dimensions))
        door_end_2d = np.array(to_grid(self.env._sim.pathfinder, door_end_3d, self.grid_dimensions))
        self.door = []
        self.door.append(door_start_2d)
        self.door.append(door_end_2d)
        poseArrayMsg = MarkerArray()
        for i in range (2):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "my_map_frame"
            marker.header.stamp = rospy.Time.now()
            marker.type = 2
            marker.pose.position.x = self.door[i][0]-1
            marker.pose.position.y = self.door[i][1]-1
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0         
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            poseArrayMsg.markers.append(marker)
        self._pub_door.publish(poseArrayMsg)
        self.door_middle = (door_start_2d+door_end_2d)/2
        door_middle_3d = (np.array(door_start_3d)+np.array(door_end_3d))/2
        pos = mn.Vector3(door_middle_3d[0], 2.0, door_middle_3d[2])
        ori = mn.Vector3(-1.57,0.,0.)
        Mt = mn.Matrix4.translation(pos)
        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
        cam_transform = Mt @ Mz @ My @ Mx
        agent_node = self.env._sim._default_agent.scene_node
        inv_T = agent_node.transformation.inverted()
        cam_transform = inv_T @ cam_transform
        camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
        camera.node.transformation = (
            orthonormalize_rotation_shear(cam_transform)
        )
        self.third_camera.render_camera.node.transformation = camera.node.transformation
        # self.proj = np.linalg.inv(np.array(self.third_camera.render_camera.projection_matrix))
        # self.cam = np.linalg.inv(np.array(cam_transform))
        self.proj = (np.array(self.third_camera.render_camera.projection_matrix))
        self.cam = (np.array(self.third_camera.render_camera.camera_matrix))
        # temp = self.cam[2].copy()
        # self.cam[2] = self.cam[1]
        # self.cam[1] = temp
        world_coord = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = 0, v = 0, debug = debug)
        # head_camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_head_rgb']
        world_coord_1 = img_to_world(proj = self.proj, cam = self.cam, W = THIRD_RGB_SIZE, H = THIRD_RGB_SIZE, u = THIRD_RGB_SIZE, v = THIRD_RGB_SIZE, debug = debug)
        self.img_res = abs(world_coord_1[0] - world_coord[0])/THIRD_RGB_SIZE
        self._pub_img_res.publish(self.img_res)
        map_coord = np.array(to_grid(self.env._sim.pathfinder, world_coord, self.grid_dimensions))
        map_coord = map_coord + np.array([-1,-1])
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "my_map_frame"
        t.child_frame_id = "camera_frame"
        t.transform.translation.x = map_coord[0]
        t.transform.translation.y = map_coord[1]
        t.transform.translation.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, 0.0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.br_tf_2.sendTransform(t)

    def map_to_base_link(self, msg):
        theta = msg['theta']
        use_tf_2 = True
        self.pub_door()
        
        if (not use_tf_2):
            self.br.sendTransform((-self.initial_state[0][0]+1, -self.initial_state[0][1]+1,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, 0.0),
                            rospy.Time(0),
                            "my_map_frame",
                            "interim_link"
            )
            self.br.sendTransform((0.0,0.0,0.0),
                            tf.transformations.quaternion_from_euler(0, 0, -theta),
                            rospy.Time(0),
                            "interim_link",
                            "base_link"
            )
        else:
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "interim_link"
            t.child_frame_id = "my_map_frame"
            t.transform.translation.x = -self.initial_state[0][0]+1
            t.transform.translation.y = -self.initial_state[0][1]+1
            t.transform.translation.z = 0.0
            q = tf.transformations.quaternion_from_euler(0, 0, 0.0)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = "interim_link"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            
            q = tf.transformations.quaternion_from_euler(0, 0, -theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "my_map_frame"
            t.child_frame_id = "door_frame"
            t.transform.translation.x = self.door_middle[0]-1
            t.transform.translation.y = self.door_middle[1]-1
            t.transform.translation.z = 0.0
            a = np.append(self.door_middle- self.door[0], [0])
            quat = quaternion_from_two_vectors(np.array([1,0,0]), a)
            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w
            self.br_tf_2.sendTransform(t)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "door_frame"
            t.child_frame_id = "small_grid_frame"
            t.transform.translation.x = self.grid_size_in_m/2
            t.transform.translation.y = -self.grid_size_in_m/2
            t.transform.translation.z = 0.0
            q = tf.transformations.quaternion_from_euler(0, 0, 1.57)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.br_tf_2.sendTransform(t)

            


        poseMsg = PoseStamped()
        poseMsg.header.stamp = rospy.Time.now()
        poseMsg.header.frame_id = "my_map_frame"
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        poseMsg.pose.orientation.x = quat[0]
        poseMsg.pose.orientation.y = quat[1]
        poseMsg.pose.orientation.z = quat[2]
        poseMsg.pose.orientation.w = quat[3]
        poseMsg.pose.position.x = self.initial_state[0][0]-1
        poseMsg.pose.position.y = self.initial_state[0][1]-1
        poseMsg.pose.position.z = 0.0
        self._robot_pose.publish(poseMsg)

        ##### Publish other agents 
        poseArrayMsg = PoseArray()
        poseArrayMsg.header.frame_id = "my_map_frame"
        poseArrayMsg.header.stamp = rospy.Time.now()
        # follower_pos = my_env.follower.rigid_state.translation
        # theta = my_env.get_object_heading(my_env.follower.transformation)
        # quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        # follower_pose_2d = to_grid(my_env.env._sim.pathfinder, follower_pos, my_env.grid_dimensions)
        # follower_pose_2d = follower_pose_2d*(0.025*np.ones([1,2]))[0]
        
        for i in range(len(self.initial_state)-1):
            poseMsg = Pose()
            obj_theta = self.get_object_heading(self.objs[i+1].base_transformation) #- mn.Rad(np.pi/2-0.97 +np.pi)
            quat = tf.transformations.quaternion_from_euler(0, 0, obj_theta)
            poseMsg.orientation.x = quat[0]
            poseMsg.orientation.y = quat[1]
            poseMsg.orientation.z = quat[2]
            poseMsg.orientation.w = quat[3]
            poseMsg.position.x = self.initial_state[i+1][0]-1
            poseMsg.position.y = self.initial_state[i+1][1]-1
            poseMsg.position.z = 0.0
            poseArrayMsg.poses.append(poseMsg)
        self.hack_to_save = -euler_from_quaternion(quat)[2]
        self._pub_all_agents.publish(poseArrayMsg)


        robot_final_goal_2d = list(to_grid(self.env._sim.pathfinder, self.env.current_episode.info['robot_goal'], self.grid_dimensions))
        goal_marker = Marker()
        goal_marker.header.frame_id = "my_map_frame"
        goal_marker.type = 2
        goal_marker.pose.position.x = robot_final_goal_2d[0]-1
        goal_marker.pose.position.y = robot_final_goal_2d[1]-1
        goal_marker.pose.position.z = 0.0
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.2
        goal_marker.scale.z = 0.2
        goal_marker.color.a = 1.0         
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        self._pub_goal_marker.publish(goal_marker)

        goal_marker = Marker()
        goal_marker.header.frame_id = "my_map_frame"
        goal_marker.type = 2
        goal_marker.pose.position.x = self.initial_state[0][4]-1
        goal_marker.pose.position.y = self.initial_state[0][5]-1
        goal_marker.pose.position.z = 0.0
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.2
        goal_marker.scale.z = 0.2
        goal_marker.color.a = 1.0         
        goal_marker.color.r = 0.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        self._pub_rvo_goal_marker.publish(goal_marker)

    def convert_pixel_to_world(self, pixel_x, pixel_y):
        """
        将tensorboard上top_down_map的pixel坐标转换为world frame坐标
        
        使用方法示例:
            # 假设你在tensorboard上点击了 (150, 200) 这个pixel
            world_pos = self.convert_pixel_to_world(150, 200)
            print(f"World坐标 [x, y, z]: {world_pos}")
        
        Args:
            pixel_x: map上的列坐标（x方向，对应width）
            pixel_y: map上的行坐标（y方向，对应height）
        
        Returns:
            world_pos_3d: numpy array [x, y, z] 在world frame中的3D坐标
        """
        # 注意：maps.from_grid中，grid_x是行索引，grid_y是列索引
        # 所以pixel_y对应grid_x（行），pixel_x对应grid_y（列）
        world_pos_3d = self.env._sim.pixel_to_world_coordinate(
            pixel_x=pixel_y,  # pixel_y是行索引，对应grid_x
            pixel_y=pixel_x,  # pixel_x是列索引，对应grid_y
            grid_resolution=self.grid_dimensions
        )
        return world_pos_3d
    
    def generate_episodes_from_door_pixels(self, door_pixel_pairs=None, distance=2.7, episodes_per_door=3):
        """
        使用多组pixel点生成episodes
        
        使用方法:
            # 在代码中直接设置多组pixel点
            door_pixel_pairs = [
                [[300, 231], [300, 243]],  # 第一组：门起点和终点
                [[400, 200], [400, 250]],  # 第二组：门起点和终点
                # 可以添加更多组...
            ]
            my_env.generate_episodes_from_door_pixels(door_pixel_pairs)
            
            # 或者使用custom_topdown_pixel_points（只支持一组）
            my_env.generate_episodes_from_door_pixels()
        
        Args:
            door_pixel_pairs: 多组pixel点的列表，格式: [[[pixel_x1, pixel_y1], [pixel_x2, pixel_y2]], ...]
                            如果为None，则使用custom_topdown_pixel_points（只支持一组）
            distance: 距离门中点的距离（米），默认2.7
            episodes_per_door: 每个门生成多少个episode，默认3
        """
        import sys
        import os
        
        # 如果没有提供pixel点对，使用custom_topdown_pixel_points
        if door_pixel_pairs is None:
            if self.custom_topdown_pixel_points is None or len(self.custom_topdown_pixel_points) < 2:
                print("错误: 需要提供door_pixel_pairs，或者设置custom_topdown_pixel_points")
                return
            door_pixel_pairs = [self.custom_topdown_pixel_points]
        
        # 确保door_pixel_pairs是列表的列表
        if not isinstance(door_pixel_pairs[0][0], list):
            # 如果只有一组，转换为列表的列表
            door_pixel_pairs = [door_pixel_pairs]
        
        print(f"准备生成episodes，共 {len(door_pixel_pairs)} 组门，每组生成 {episodes_per_door} 个episode")
        
        # 导入生成函数
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generate_script_path = os.path.join(script_dir, '..', 'generate_episodes_from_doors.py')
        generate_script_path = os.path.abspath(generate_script_path)
        
        # 动态导入生成模块
        sys.path.insert(0, os.path.dirname(generate_script_path))
        try:
            import generate_episodes_from_doors as gen_module
        except ImportError:
            print(f"错误: 无法导入generate_episodes_from_doors模块")
            return
        
        # 加载原始JSON和配置
        input_json = os.path.join(script_dir, '..', 'scene_wise', 'test_dataset15_finalest_only_8.json.gz')
        input_json = os.path.abspath(input_json)
        base_data = gen_module.load_json_dataset(input_json)
        if len(base_data['episodes']) == 0:
            print("错误: 原始JSON中没有episode")
            return
        
        base_episode = base_data['episodes'][0]
        
        # 直接使用已有的环境，避免配置问题
        # 使用self.env而不是创建新环境
        pathfinder = self.env._sim.pathfinder
        
        # 设置输出路径 - 保存到 habitat-lab/data/data_xinyuan 目录
        # 获取habitat-lab的根目录
        habitat_lab_root = os.path.join(script_dir, '..')
        habitat_lab_root = os.path.abspath(habitat_lab_root)
        output_dir = os.path.join(habitat_lab_root, 'data', 'data_xinyuan')
        os.makedirs(output_dir, exist_ok=True)
        output_json = os.path.join(output_dir, 'test_dataset15_generated.json.gz')
        print(f"输出目录: {output_dir}")
        print(f"输出文件: {output_json}")
        
        # 生成所有episodes
        all_episodes = []
        next_episode_id = 0
        
        for group_idx, (pixel_point1, pixel_point2) in enumerate(door_pixel_pairs):
            print(f"\n处理第 {group_idx + 1}/{len(door_pixel_pairs)} 组门:")
            print(f"  门起点像素: {pixel_point1}")
            print(f"  门终点像素: {pixel_point2}")
            
            # 转换pixel到world坐标
            door_start_world = self.convert_pixel_to_world(pixel_point1[0], pixel_point1[1])
            door_end_world = self.convert_pixel_to_world(pixel_point2[0], pixel_point2[1])
            door_start = door_start_world.tolist()
            door_end = door_end_world.tolist()
            print(f"  门起点world: {door_start}")
            print(f"  门终点world: {door_end}")
            
            # 为这组门生成episodes
            for ep_idx in range(episodes_per_door):
                print(f"  生成episode {ep_idx + 1}/{episodes_per_door}...")
                new_episode = gen_module.generate_episode_from_door(
                    base_episode,
                    door_start,
                    door_end,
                    pathfinder,
                    next_episode_id,
                    distance
                )
                all_episodes.append(new_episode)
                next_episode_id += 1
                
                print(f"    Episode {new_episode['episode_id']}:")
                print(f"      机器人起始: {new_episode['start_position']}")
                print(f"      机器人目标: {new_episode['info']['robot_goal']}")
                print(f"      人起始: {new_episode['info']['human_start']}")
                print(f"      人目标: {new_episode['info']['human_goal']}")
        
        # 创建新的数据集
        new_data = {
            'config': base_data.get('config'),
            'episodes': all_episodes
        }
        
        # 保存新JSON
        print(f"\n保存新JSON到: {output_json}")
        gen_module.save_json_dataset(new_data, output_json)
        
        print(f"\n完成! 共生成了 {len(all_episodes)} 个episodes ({len(door_pixel_pairs)} 组门 × {episodes_per_door} 个episode/组)")
        
        # 不需要关闭环境，因为使用的是已有的self.env

    def set_topdown_camera_from_pixel_points(self, pixel_point1, pixel_point2, height=20.0):
        """
        根据两个pixel点设置俯视相机位置（取中点作为拍摄中心）
        
        Args:
            pixel_point1: [pixel_x, pixel_y] 第一个点的pixel坐标
            pixel_point2: [pixel_x, pixel_y] 第二个点的pixel坐标
            height: 相机高度，默认20.0米
        
        Returns:
            door_middle_3d: 计算得到的中点world坐标
        """
        # 1. 转换pixel到world坐标
        world_point1 = self.convert_pixel_to_world(pixel_point1[0], pixel_point1[1])
        world_point2 = self.convert_pixel_to_world(pixel_point2[0], pixel_point2[1])
        
        # 2. 计算中点
        door_middle_3d = (world_point1 + world_point2) / 2
        
        # 3. 设置相机位置和朝向
        pos = mn.Vector3(door_middle_3d[0], height, door_middle_3d[2])
        ori = mn.Vector3(-1.57, 0., 0.)  # pitch=-1.57 表示向下看
        
        # 4. 构建相机变换矩阵
        Mt = mn.Matrix4.translation(pos)
        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
        cam_transform = Mt @ Mz @ My @ Mx
        
        # 5. 转换到agent坐标系
        agent_node = self.env._sim._default_agent.scene_node
        inv_T = agent_node.transformation.inverted()
        cam_transform = inv_T @ cam_transform
        
        # 6. 应用变换到相机
        camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
        camera.node.transformation = orthonormalize_rotation_shear(cam_transform)
        
        # 7. 设置投影矩阵（与原有代码保持一致）
        camera.render_camera.projection_matrix = mn.Matrix4([
            [0.3000000059604645, 0, 0, 0],
            [0, 0.3000000059604645, 0, 0],
            [0, 0, -0.002000020118430257, 0],
            [0, 0, -1.0000200271606445, 1]
        ])
        
        return door_middle_3d

    def capture_topdown_from_pixel_points(self, pixel_point1, pixel_point2, height=20.0):
        """
        从两个pixel点拍摄俯视图（必须在主线程中调用）
        
        Args:
            pixel_point1: [pixel_x, pixel_y] 第一个点的pixel坐标
            pixel_point2: [pixel_x, pixel_y] 第二个点的pixel坐标
            height: 相机高度，默认20.0米
        
        Returns:
            topdown_img: 拍摄得到的俯视图（numpy array）
        """
        # 设置相机位置
        self.set_topdown_camera_from_pixel_points(pixel_point1, pixel_point2, height)
        
        # 获取观测（必须在主线程中调用）
        observations = self.env._sim.get_sensor_observations()
        
        # 返回full_top_down图像
        if "full_top_down" in observations:
            return observations["full_top_down"][:,:,:3]
        else:
            # 如果没有full_top_down，使用agent_1_third_rgb
            return observations["agent_1_third_rgb"][:,:,:3]
    
    def update_cached_topdown_new(self, force_update=False):
        """
        在主线程中更新缓存的俯视图（应该在 update_agent_pos_vel 中调用）
        
        Args:
            force_update: 如果为True，强制更新；否则根据更新频率决定是否更新
        """
        # 控制更新频率，避免每帧都更新
        if not force_update:
            self.topdown_update_counter += 1
            if self.topdown_update_counter % self.topdown_update_freq != 0:
                return
        
        try:
            if self.custom_topdown_pixel_points is not None:
                # 使用自定义的pixel点拍摄
                pixel_point1, pixel_point2 = self.custom_topdown_pixel_points
                self.cached_topdown_new_img = self.capture_topdown_from_pixel_points(
                    pixel_point1, pixel_point2, height=20.0
                )
            else:
                # 如果没有设置自定义点，使用默认的门的位置
                door_start_3d = self.env.current_episode.info['door_start']
                door_end_3d = self.env.current_episode.info['door_end']
                door_middle_3d = (np.array(door_start_3d) + np.array(door_end_3d)) / 2
                pos = mn.Vector3(door_middle_3d[0], 20.0, door_middle_3d[2])
                ori = mn.Vector3(-1.57, 0., 0.)
                Mt = mn.Matrix4.translation(pos)
                Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
                My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
                Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
                cam_transform = Mt @ Mz @ My @ Mx
                agent_node = self.env._sim._default_agent.scene_node
                inv_T = agent_node.transformation.inverted()
                cam_transform = inv_T @ cam_transform
                camera = self.env.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['agent_1_third_rgb']
                camera.node.transformation = orthonormalize_rotation_shear(cam_transform)
                camera.render_camera.projection_matrix = mn.Matrix4([
                    [0.3000000059604645, 0, 0, 0],
                    [0, 0.3000000059604645, 0, 0],
                    [0, 0, -0.002000020118430257, 0],
                    [0, 0, -1.0000200271606445, 1]
                ])
                observations_new = self.env._sim.get_sensor_observations()
                if "full_top_down" in observations_new:
                    self.cached_topdown_new_img = observations_new["full_top_down"][:,:,:3]
                else:
                    self.cached_topdown_new_img = observations_new["agent_1_third_rgb"][:,:,:3]
        except Exception as e:
            print(f"[Warning] 无法更新缓存的俯视图: {e}")
            self.cached_topdown_new_img = None

    def convert_pixel_to_world_test(self, pixel_x, pixel_y):
        """
        简单转换函数：输入pixel坐标，在终端日志中输出world frame坐标
        
        使用方法:
            my_env.convert_pixel_to_world_test(150, 200)
        
        Args:
            pixel_x: pixel x坐标（列坐标，对应width）
            pixel_y: pixel y坐标（行坐标，对应height）
        """
        # 验证坐标范围
        if not (0 <= pixel_x < self.grid_dimensions[1]):
            print(f"[Pixel转换] 错误: pixel_x ({pixel_x}) 超出范围 [0, {self.grid_dimensions[1]-1}]")
            return None
        if not (0 <= pixel_y < self.grid_dimensions[0]):
            print(f"[Pixel转换] 错误: pixel_y ({pixel_y}) 超出范围 [0, {self.grid_dimensions[0]-1}]")
            return None
        
        # 转换坐标
        world_pos = self.convert_pixel_to_world(pixel_x, pixel_y)
        
        # 在终端日志中输出
        print(f"[Pixel转换] Pixel({pixel_x}, {pixel_y}) -> World[{world_pos[0]:.4f}, 30, {world_pos[2]:.4f}]")
        
        return world_pos

    # def batch_convert_pixels_to_world(self, pixel_coords, output_json_path="pixel_to_world.json"):
    #     """
    #     批量转换pixel坐标到world frame坐标，并导出为JSON文件
        
    #     使用方法:
    #         # 方法1: 直接在代码中传入坐标列表
    #         pixel_coords = [
    #             [370, 167],
    #             [400, 200],
    #             [500, 300]
    #         ]
    #         my_env.batch_convert_pixels_to_world(pixel_coords)
            
    #         # 方法2: 从CSV文件读取（CSV格式：pixel_x,pixel_y）
    #         import csv
    #         pixel_coords = []
    #         with open('pixels.csv', 'r') as f:
    #             reader = csv.reader(f)
    #             next(reader)  # 跳过标题行（如果有）
    #             for row in reader:
    #                 pixel_coords.append([int(row[0]), int(row[1])])
    #         my_env.batch_convert_pixels_to_world(pixel_coords, "output.json")
        
    #     Args:
    #         pixel_coords: 列表，每个元素是[pixel_x, pixel_y]
    #         output_json_path: 输出JSON文件路径
    #     """
    #     results = []
    #     valid_count = 0
    #     invalid_count = 0
        
    #     print(f"\n[批量转换] 开始处理 {len(pixel_coords)} 个pixel坐标...")
        
    #     for i, (pixel_x, pixel_y) in enumerate(pixel_coords):
    #         # 验证坐标范围
    #         if not (0 <= pixel_x < self.grid_dimensions[1]):
    #             print(f"[批量转换] 第{i+1}个坐标无效: pixel_x ({pixel_x}) 超出范围 [0, {self.grid_dimensions[1]-1}]")
    #             invalid_count += 1
    #             results.append({
    #                 "pixel_x": pixel_x,
    #                 "pixel_y": pixel_y,
    #                 "world_x": None,
    #                 "world_y": None,
    #                 "world_z": None,
    #                 "valid": False,
    #                 "error": f"pixel_x超出范围"
    #             })
    #             continue
    #         if not (0 <= pixel_y < self.grid_dimensions[0]):
    #             print(f"[批量转换] 第{i+1}个坐标无效: pixel_y ({pixel_y}) 超出范围 [0, {self.grid_dimensions[0]-1}]")
    #             invalid_count += 1
    #             results.append({
    #                 "pixel_x": pixel_x,
    #                 "pixel_y": pixel_y,
    #                 "world_x": None,
    #                 "world_y": None,
    #                 "world_z": None,
    #                 "valid": False,
    #                 "error": f"pixel_y超出范围"
    #             })
    #             continue
            
    #         # 转换坐标
    #         world_pos = self.convert_pixel_to_world(pixel_x, pixel_y)
    #         valid_count += 1
            
    #         result = {
    #             "pixel_x": pixel_x,
    #             "pixel_y": pixel_y,
    #             "world_x": float(world_pos[0]),
    #             "world_y": float(world_pos[1]),
    #             "world_z": float(world_pos[2]),
    #             "valid": True
    #         }
    #         results.append(result)
            
    #         # 在终端输出
    #         print(f"[批量转换] {i+1}/{len(pixel_coords)}: Pixel({pixel_x}, {pixel_y}) -> World[{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]")
        
    #     # 保存为JSON文件
    #     output_data = {
    #         "map_info": {
    #             "grid_dimensions": {
    #                 "height": int(self.grid_dimensions[0]),
    #                 "width": int(self.grid_dimensions[1])
    #             },
    #             "pixel_range": {
    #                 "x": [0, int(self.grid_dimensions[1]-1)],
    #                 "y": [0, int(self.grid_dimensions[0]-1)]
    #             }
    #         },
    #         "conversion_results": results,
    #         "summary": {
    #             "total": len(pixel_coords),
    #             "valid": valid_count,
    #             "invalid": invalid_count
    #         }
    #     }
        
    #     # 获取绝对路径
    #     abs_output_path = os.path.abspath(output_json_path)
        
    #     with open(abs_output_path, 'w', encoding='utf-8') as f:
    #         json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    #     print(f"\n[批量转换] 完成！")
    #     print(f"  有效转换: {valid_count}/{len(pixel_coords)}")
    #     print(f"  无效坐标: {invalid_count}/{len(pixel_coords)}")
    #     print(f"  结果已保存到: {abs_output_path}")
        
    #     return output_data

    def point_callback(self, msg):
        point_map = [msg.point.x, msg.point.y, msg.point.z]
        p = [point_map[0]+1, point_map[1]+1]
        if self.start_ep and not USE_IRL_AGENT:
            self.cheating_point = p
            return
        if self.start_ep and USE_IRL_AGENT:
            self.clicked_point = p
            return
        self.line.append(p)
        
        prompt = '> '
        print("Done with points? ")
        # done = input(prompt)
        done = 'b'
        if done == 'y':
            self.start_ep = True
            self.sfm.orca_sim.processObstacles()
            self.sfm.plot_obstacles()
            print("Done with points", self.obs)
            return
        elif done == 'a' and len(self.line) >= 2:
            # self.sfm.obs.append([self.line[0][0], self.line[-1][0], self.line[0][1], self.line[-1][1]])
            self.obs.append([tuple([self.line[0][1], self.line[0][0]]), tuple([self.line[-1][1],self.line[-1][0]])])
            self.sfm.orca_sim.addObstacle([tuple([self.line[0][1], self.line[0][0]]), tuple([self.line[-1][1],self.line[-1][0]])])
            print("Adding object ", self.line)
            self.line = [] 
        elif done =='b':
            list_table_1 = [[11.588330268859863, 15.655499458312988], [11.802519798278809, 15.68454360961914], [11.816425323486328, 15.763737678527832], [11.961987495422363, 15.808496475219727], [12.042765617370605, 15.904221534729004], [14.214700698852539, 15.886024475097656], [14.234253883361816, 15.708464622497559], [14.550985336303711, 15.485882759094238], [14.538640975952148, 15.317964553833008], [14.42050552368164, 15.308235168457031], [14.435802459716797, 15.120619773864746], [14.264750480651855, 15.12462043762207], [14.23427677154541, 15.033063888549805], [13.839558601379395, 15.027502059936523], [13.844256401062012, 14.517410278320312], [13.253166198730469, 14.511800765991211], [13.280839920043945, 15.014991760253906], [13.059856414794922, 15.03444766998291], [13.032105445861816, 14.581336975097656], [12.469135284423828, 14.541062355041504], [12.462349891662598, 15.02047348022461], [11.95616340637207, 15.026178359985352], [11.93496322631836, 15.216469764709473], [11.674623489379883, 15.231595039367676], [11.646646499633789, 15.32201099395752], [11.567437171936035, 15.326179504394531]]
            list_table_1 = list_table_1[:-2]
            list_table_2 = [[12.555318832397461, 13.287395477294922], [12.742836952209473, 13.305675506591797], [12.741669654846191, 13.491787910461426], [13.129401206970215, 13.505683898925781], [13.116826057434082, 13.305461883544922], [13.34615421295166, 13.297121047973633], [13.36707878112793, 13.496026039123535], [13.736709594726562, 13.511171340942383], [13.743658065795898, 13.322402000427246], [14.034088134765625, 13.309804916381836], [14.064624786376953, 13.48914623260498], [14.32740306854248, 13.495879173278809], [14.34408950805664, 13.308235168457031], [14.751039505004883, 13.30294132232666], [14.749950408935547, 13.127532958984375], [14.831957817077637, 13.105292320251465], [14.833391189575195, 12.528411865234375], [12.57761001586914, 12.559179306030273], [12.56087875366211, 13.29295539855957]]
            list_two = [[10.297499656677246, 16.22457790374756], [15.36426067352295, 16.23254680633545], [15.39486312866211, 11.830757141113281], [11.898244857788086, 11.7006254196167], [11.90356731414795, 11.212478637695312], [10.246871948242188, 11.215006828308105], [10.3082914352417, 16.21781635284424]]
            # for i in range(len(list_tuples)-1):
            #   sim.addObstacle([tuple([list_tuples[i][1], list_tuples[i][0]]), tuple([list_tuples[i+1][1], list_tuples[i+1][0]])])
            # sim.addObstacle([tuple([list_tuples[i][1], list_tuples[i][0]]), tuple([list_tuples[0][1], list_tuples[0][0]])])
            
            
            # new_list_tuples = []
            # for i in list_table_1:
            #     new_list_tuples.append(tuple(i))
            # self.sfm.orca_sim.addObstacle(new_list_tuples)
            # new_list_tuples = []
            # for i in list_table_2: 
            #     new_list_tuples.append(tuple(i))
            # self.sfm.orca_sim.addObstacle(new_list_tuples)

            # new_list_tuples = []
            # for i in list_two:
            #     new_list_tuples.append(tuple(i))
            # self.sfm.orca_sim.addObstacle(new_list_tuples)
            # # for i in list_tuples:
            # #     sim.addObstacle(i)
            # self.sfm.orca_sim.processObstacles()
            self.sfm.plot_obstacles()
            self.sfm.get_future_position(np.array(self.initial_state), num_steps=1200)
            self.sfm.reset_peds(self.initial_state)
            self.start_ep = True
            return
        else:
            if len(self.line) >-2:    
                self.sfm.orca_sim.addObstacle([tuple([self.line[-1][1], self.line[-1][0]]), tuple([self.line[-2][1],self.line[-2][0]])])
                self.obs.append([tuple([self.line[-1][1], self.line[-1][0]]), tuple([self.line[-2][1],self.line[-2][0]])])
                print("Adding line in same object", self.line)
           
        
        # point_map = [msg.point.x, msg.point.y, msg.point.z]
        # p = [point_map[0]+1, point_map[1]+1]
        # point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
        # print("Placing human at ",point_3d)
        # k = 'agent_1_oracle_nav_randcoord_action'
        # chance = np.random.randn(1)
        # if chance>0.5:
        #     self.env.task.actions[k].coord_nav = np.array([point_3d[0], point_3d[1], point_3d[2]])
        #     self.env._task.my_nav_to_info.human_info.nav_goal_pos = np.array([point_3d[0], point_3d[1], point_3d[2]])
        #     print("setting human goal to ",point_3d)
        
    def get_path(self,msg):
        
        traj = []
        traj_2d = []
        norm_list = []
        traj_interpolated = []
        traj_interpolated_2d = []
        for pose in msg.poses:
            point_map = [pose.pose.position.x, pose.pose.position.y]
            p = [point_map[0]+1, point_map[1]+1]
            traj_2d.append(p)
            p0 = np.array(self.initial_state[0][0:2]) 
            diff_interp = np.array(p) - np.array(p0) 
            norm_interp = np.linalg.norm(diff_interp)
            p2 = diff_interp/norm_interp*0.4 + np.array(p0)
            traj_interpolated_2d.append(p2)
            p2_3d = from_grid(self.env._sim.pathfinder, [p2[0]/0.025, p2[1]/0.025], self.grid_dimensions)
            point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
            traj.append(point_3d)
            traj_interpolated.append(p2_3d)
            norm_list.append([np.linalg.norm(np.array(p) - np.array(p0))])   
        start_index = np.argmin(norm_list)
        traj = traj[start_index:]
        traj_2d = traj_2d[start_index:]
        traj_interpolated = traj_interpolated[start_index:]
        print("Robot drift is ", norm_list[0], start_index)
        if start_index>1 and norm_list[0]>0.1:
            print("The robot state has drifted from the initial state ", norm_list[0:start_index])
            self.waiting_for_traj = True
            return
        end_index = len(traj_2d)-1
        # print("Length of norm list is ", len(norm_list))
        # for i in range(len(norm_list)):
        #     print("I is ", i)
        #     print("Condition is ", norm_list[i][0]>=0.2 and i < len(norm_list)/2)
        #     if norm_list[i][0]>=0.2 and i < len(norm_list)/2:
        #         end_index = int(i)
        #         break
        print("chosen end index is ", end_index)
        self.cheating_point = traj_2d[end_index]
        print("Should execute trajectory now ")
        self.waiting_for_traj = False
        
        # self.cheating_point = np.array([traj[end_index][0], traj[end_index][1], traj[end_index][2]])
        
    def pose_callback(self, msg):
        point_map = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        p = [point_map[0]+1, point_map[1]+1]
        point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
        print("Setting robot goal at  ",point_3d)
        self.current_point = np.array([point_3d[0], point_3d[1], point_3d[2]])
        self.start_ep = True

def callback(vel, my_env):
    #### Robot Control ####
    my_env.linear_velocity = np.array([(1.0 * vel.linear.y), 0.0, (1.0 * vel.linear.x)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])

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

    my_env = sim_env(config)
    
    # ============================================================
    # 使用pixel点生成episodes
    # ============================================================
    # 直接在代码中设置多组pixel点，每组生成3个episode
    # 格式: [[[门起点pixel], [门终点pixel]], ...]
    
    DOOR_PIXEL_PAIRS = [
        [[668, 231], [668, 243]],  # 第一组门
        # [[400, 200], [400, 250]],  # 第二组门（取消注释来添加）
        # [[500, 300], [500, 350]],  # 第三组门（取消注释来添加）
        # 可以添加更多组...
    ]
    
    # # 生成episodes
    my_env.generate_episodes_from_door_pixels(DOOR_PIXEL_PAIRS)
    
    my_env.start()
    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    while not rospy.is_shutdown():
   
        my_env.update_agent_pos_vel()
        # rospy.spin()
        my_env._r_control.sleep()
