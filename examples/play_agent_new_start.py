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
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0

from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations.utils import images_to_video
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
import gzip
import csv
from tf.transformations import euler_from_quaternion
import copy
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
import json
from IPython import embed
# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"
MAP_DIR = "/home/catkin_ws/src/habitat_ros_interface/maps/"
THIRD_RGB_SIZE = 128
GRID_SIZE = 6
HUMAN_HEAD_START = 0       #60
DATASET_PATH = "/habitat-lab/data/scene_wise/test_dataset15.json"
# PREV_OUTPUT_DATSET_PATH = "/habitat-lab/data/scene_wise/test_dataset15.json"
OUTPUT_DATSET_PATH = "/habitat-lab/data/scene_wise/test_dataset15_other.json.gz"

CSV_PATH = "/habitat-lab/data/run0.csv"
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
    _sensor_rate = 50  # hz
    
    _current_episode = 0
    _total_number_of_episodes = 0
    
    replan_freq = 1
    replan_counter = 0
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.env = habitat.Env(config = config)
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
        
        rospy.init_node("sim", anonymous=False)
        sim_cfg = config['habitat']['simulator']
        self.control_frequency = int(np.floor(sim_cfg['ctrl_freq']/sim_cfg['ac_freq_ratio']))
        time_step = 1.0 / (self.control_frequency)
        self.control_frequency = 10
        self._r = rospy.Rate(self._sensor_rate)
        self._r_control = rospy.Rate(self.control_frequency)
        
        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_rgb_2 = rospy.Publisher("~rgb2", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)
        self._robot_pose = rospy.Publisher("~robot_pose", PoseStamped, queue_size = 1)
        self._pub_all_agents = rospy.Publisher("~agent_poses", PoseArray, queue_size = 1)
        self._pub_door = rospy.Publisher("~door", MarkerArray, queue_size = 1)
        self._pub_goal_marker = rospy.Publisher("~goal", Marker, queue_size = 1)
        self._pub_human_goal_marker = rospy.Publisher("~human_goal", Marker, queue_size = 1)
        self.cloud_pub = rospy.Publisher("top_down_img", PointCloud2, queue_size=2)
        self._pub_img_res = rospy.Publisher("img_res", Float64, queue_size= 1)
        self._reload_map_server = rospy.Publisher("reload_map_server", Bool, queue_size= 1)
        self._sub_wait = rospy.Subscriber("wait_for_traj", Bool, self.start_wait, queue_size = 1)
        self._sub_path = rospy.Subscriber("irl_path", Path, self.get_path,  queue_size=1)
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
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            agents_goal_pos_3d = [self.env.current_episode.info['robot_goal'], self.env.current_episode.info['human_goal']]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
        self.linear_velocity = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.grid_size_in_m = GRID_SIZE
        self.grid_resolution = 0.15
        self.grid_dimension = self.grid_size_in_m/self.grid_resolution
        sim_config = habitat.get_config("/habitat-lab/habitat-baselines/habitat_baselines/config/social_nav/social_nav_fetch_test.yaml")
        agent_config = sim_config.habitat_baselines
        ppo = baseline_registry.get_trainer(agent_config.trainer_name)
        checkpoint_path = agent_config.eval_ckpt_path_dir
        
        self.ppo = ppo(sim_config)
        ckpt_dict = self.ppo.load_checkpoint(
                checkpoint_path, map_location="cpu"
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
        self.im_array = []
        self.new_dataset = RearrangeDatasetV0()
        self.human_start_point = None
        self.robot_start_point = None
        self.human_goal_point = None 
        self.robot_goal_point = None
        with open(DATASET_PATH, 'r') as j:
            self.all_episodes = json.loads(j.read())
        # with open(PREV_OUTPUT_DATSET_PATH, 'r') as j:
        #     eps = json.loads(j.read())["episodes"]
        # for ep in eps:
        #     self.new_dataset.episodes.append(ep)
    def reset(self):
        #### Save the results of the previous episode ####
        # metrics = self.env.get_metrics()
        # results_dict = {}
        # results_dict["num_steps"] = metrics["num_steps"]
        # results_dict["did_collide"] = metrics["did_collide"]
        # results_dict["robot_scene_collision"] = metrics["robot_collisions"]["robot_scene_colls"]
        # results_dict["social_nav_to_pos_success"] = metrics["social_nav_to_pos_success"]
        # results_dict["social_dist_to_goal"] = metrics["social_dist_to_goal"]
        # results_dict["avg_robot_to_human_dis_over_epi"] = metrics["social_nav_stats"]["avg_robot_to_human_dis_over_epi"]
        # results_dict["social_nav_reward"] = metrics["social_nav_reward"]
        
        # with open(CSV_PATH, "a", newline="") as fp:
        # # Create a writer object
        #     writer = csv.DictWriter(fp, fieldnames=metrics.keys())
        #     if self._current_episode == 0:
        #         writer.writeheader()
        #     writer.writerow(metrics)
        # #### Finish writing csv file ####
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
        self._reload_map_server.publish(True)
        self.start_ep = False
        ### Saving images for video ####
        # self.im_array[0].save(SAVE_VIDEO_DIR + "/episode_" + str(self.env.current_episode.episode_id) + ".gif", save_all=True, append_images=self.im_array[1:], duration=100, loop=0)
        self.im_array = []
        self.human_start_point = None
        self.robot_start_point = None
        self.human_goal_point = None
        self.robot_goal_point = None
        self._current_episode +=1
        self.initial_state = []
        self.number_of_agents = len(self.env.sim.agents_mgr)
        self.objs = []
        for i in range(self.number_of_agents):
            agent_pos = self.env.sim.agents_mgr[i].articulated_agent.base_pos
            start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
            initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
            agents_goal_pos_3d = [self.env.current_episode.info['robot_goal'], self.env.current_episode.info['human_goal']]
            agents_initial_velocity = [0.5,0.0]
            goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
            self.initial_state.append(initial_pos+agents_initial_velocity+goal_pos)
            self.objs.append(self.env.sim.agents_mgr[i].articulated_agent)
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
            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["agent_0_third_rgb"][:,:,:3].ravel()),
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
            for i in range(self.number_of_agents):
                agent_pos = self.objs[i].base_pos
                start_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
                initial_pos = list(to_grid(self.env._sim.pathfinder, start_pos, self.grid_dimensions))
                agents_goal_pos_3d = [self.env._task.my_nav_to_info.robot_info.nav_goal_pos, self.env._task.my_nav_to_info.human_info.nav_goal_pos]
                agents_initial_velocity = [0.5,0.0]
                goal_pos = list(to_grid(self.env._sim.pathfinder, agents_goal_pos_3d[i], self.grid_dimensions))
                self.initial_state[i] = initial_pos+agents_initial_velocity+goal_pos
                # if self.current_point is None:
            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_rgb_2.publish(np.float32(rgb2_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            self.map_to_base_link({'x': initial_pos[0], 'y': initial_pos[1], 'theta': self.get_object_heading(self.env.sim.agents_mgr[0].articulated_agent.base_transformation)})
            lock.release()
            self._r.sleep()

    def start_wait(self, msg):
        self.waiting_for_traj = msg.data
        print("Waiting for traj? ", self.waiting_for_traj)

    def get_path(self,msg):
        
        traj = []
        traj_2d = []
        norm_list = []
        for pose in msg.poses:
            point_map = [pose.pose.position.x, pose.pose.position.y]
            p = [point_map[0]+1, point_map[1]+1]
            traj_2d.append(p)
            point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
            traj.append(point_3d)
            norm_list.append([np.linalg.norm(np.array(p) - np.array(self.initial_state[0][0:2]))])   
        start_index = np.argmin(norm_list)
        traj = traj[start_index:]
        traj_2d = traj_2d[start_index:]
        
        if len(traj)<10:
            self.initial_state[0][4:6] = traj_2d[-1]
            self.current_point = np.array([traj[-1][0], traj[-1][1], traj[-1][2]])
            
        else:
            self.initial_state[0][4:6] = traj_2d[9]
            self.current_point = np.array([traj[9][0], traj[9][1], traj[9][2]])
        self.waiting_for_traj = False
        robot_final_goal = self.env.current_episode.info['robot_goal']
        dist_to_goal = np.linalg.norm((robot_final_goal-self.current_point)[[0, 2]])
        print("Distance to final goal is ", dist_to_goal)
        if dist_to_goal <0.8:
            self.current_point = robot_final_goal
            print("Setting to robot final goal")
        print("Current point is ", self.current_point)
        
        


    def update_agent_pos_vel(self):
        if (self.env._episode_over):
            print("Done with episode and starting a new one")
            self.reset()
            return
            
        if not self.start_ep:
            base_vel = [0.0, 0.0]
            # print("Caught here ", not self.start_ep,  self.waiting_for_traj , self.current_point is None)
            self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
            return
        self.wait_counter +=1
        dist_human_moved = np.linalg.norm(self.objs[1].base_pos -  self.env.current_episode.info['human_start'])
        print("Human has moved ", dist_human_moved)
        # if (self.wait_counter < HUMAN_HEAD_START):
        while (dist_human_moved<0.2):
            print("Giving the human a head start")
            k = 'agent_1_oracle_nav_randcoord_action'
            for i in range(1):
                self.observations.update(self.env.step({"action":k, "action_args":{}}))
            dist_human_moved = np.linalg.norm(self.objs[1].base_pos -  self.env.current_episode.info['human_start'])
            return
        self.reset()
        if self.current_point is None:
            base_vel = [0.0, 0.0]
            print("Caught here ", not self.start_ep,  self.waiting_for_traj , self.current_point is None)
            self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
            return
        ### When driving agent ####
        lin_vel = self.linear_velocity[2]
        ang_vel = self.angular_velocity[1]
        base_vel = [lin_vel, ang_vel]
        # self.env._episode_over = False
        
        ### When RL drives agent ####
        # base_vel = self.act()
        
        k = 'agent_1_oracle_nav_randcoord_action'
        # # my_env.env.task.actions[k].coord_nav = self.observations['agent_0_localization_sensor'][:3]
        self.env.task.actions[k].step()

        k = 'agent_0_oracle_nav_randcoord_action'
        # self.env.task.actions[k].coord_nav = self.current_point### Read the point from the traj
        
        coord_nav = self.current_point
        # # print("Coord nav here is ", coord_nav)
        self.observations.update(self.env.step({"action":k, "action_args":{"agent_0_oracle_nav_randcoord_action":coord_nav}}))
        # self.observations.update(self.env.step({"action": 'agent_0_base_velocity', "action_args":{"agent_0_base_vel":base_vel}}))
        #### Saving images for video ####
        # self.im_array.append(Image.fromarray(self.observations["agent_1_third_rgb"].astype(np.uint8)))


        #### Goal Follower agent ####
        # print("chosen action is ", self.goal_agent.act(self.observations))
        # print("Measurement is ", self.observations["dist_to_goal"])
        # discrete_action = self.goal_agent.act(self.observations)['action']
        # if discrete_action == 0:
        #     lin_vel = 0.0
        #     ang_vel = 0.0
        # elif discrete_action == 1:
        #     lin_vel = 0.5
        #     ang_vel = 0.0
        # elif discrete_action == 2:
        #     lin_vel = 0.0
        #     ang_vel = -1.0
        # elif discrete_action == 3:
        #     lin_vel = 0.0
        #     ang_vel = 1.0
        # else:
        #     lin_vel = 0.0
        #     ang_vel = 0.0
        # base_vel = [lin_vel, ang_vel]
        # self.observations.update(self.env.step({"action":"BASE_VELOCITY", "action_args":{"base_vel":base_vel}}))
        #### Goal Follower agent ####

        # print(self.env.task.actions[k].coord_nav)
        # self.env.task.actions[k].step()
        # self.observations.update(self.env.sim.get_sensor_observations())
        # base_vel = [0.0, 0.0]
        # self.observations.update()
        if self.replan_counter % int(self.control_frequency/self.replan_freq):
            # self.img_to_grid()
            self.replan_counter = 0
        self.replan_counter +=1

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
        self._pub_all_agents.publish(poseArrayMsg)

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
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        self._pub_goal_marker.publish(goal_marker)
        goal_marker = Marker()
        goal_marker.header.frame_id = "my_map_frame"
        goal_marker.type = 2
        goal_marker.pose.position.x = self.initial_state[1][4]-1
        goal_marker.pose.position.y = self.initial_state[1][5]-1
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
        self._pub_human_goal_marker.publish(goal_marker)
        

    def point_callback(self, msg):
        
        point_map = [msg.point.x, msg.point.y, msg.point.z]
        p = [point_map[0]+1, point_map[1]+1]
        point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
        if self.robot_goal_point is None:
            k = 'agent_0_oracle_nav_randcoord_action'
            # chance = np.random.randn(1)
            # if chance>0.5:
            self.robot_goal_point = np.array(point_3d)
            self.env.task.actions[k].coord_nav = np.array([point_3d[0], point_3d[1], point_3d[2]])
            self.env._task.my_nav_to_info.robot_info.nav_goal_pos = np.array([point_3d[0], point_3d[1], point_3d[2]])
            self.env.current_episode.info['robot_goal'] = np.array([point_3d[0], point_3d[1], point_3d[2]])
            print("setting robot goal to ",point_3d)
            return
        elif self.human_goal_point is None:
            k = 'agent_1_oracle_nav_randcoord_action'
            # chance = np.random.randn(1)
            # if chance>0.5:
            self.human_goal_point = np.array(point_3d)
            self.env.task.actions[k].coord_nav = np.array([point_3d[0], point_3d[1], point_3d[2]])
            self.env._task.my_nav_to_info.human_info.nav_goal_pos = np.array([point_3d[0], point_3d[1], point_3d[2]])
            self.env.current_episode.info['human_goal'] = np.array([point_3d[0], point_3d[1], point_3d[2]])
            print("setting human goal to ",point_3d)
            return
        
        if self.robot_goal_point is not None and self.human_goal_point is not None:
            if self.robot_start_point is not None and self.human_start_point is not None:     
                for eps in self.all_episodes["episodes"]:
                    ep = copy.deepcopy(eps)
                    if ep["scene_id"] == self.env.current_episode.scene_id and ep["info"]["door_start"] == self.env.current_episode.info["door_start"]:
                        ep["start_position"] = self.env.current_episode.start_position
                        ep["start_rotation"] = np.array(self.env.current_episode.start_rotation)
                        new_ep_info = copy.deepcopy(ep["info"])
                        new_ep_info["human_start"] = self.env.current_episode.info["human_start"]
                        new_ep_info["robot_goal"] = self.robot_goal_point
                        new_ep_info["human_goal"] = self.human_goal_point
                        new_ep_info["human_rot"] = self.env.current_episode.info["human_rot"]
                        ep["info"] = copy.deepcopy(new_ep_info)
                        ep["episode_id"] = len(self.new_dataset.episodes)+1
                        self.new_dataset.episodes.append(ep)
                        break
            # a = input("Finished entering episodes? ")
            # if a == 'y':
                with gzip.open(OUTPUT_DATSET_PATH, "wt") as f:
                    f.write(self.new_dataset.to_json())
                print("Great, select the next episode!")
                self.start_ep = True
    
        
    def pose_callback(self, msg):
        point_map = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        p = [point_map[0]+1, point_map[1]+1]
        point_3d = from_grid(self.env._sim.pathfinder, [p[0]/0.025, p[1]/0.025], self.grid_dimensions)
        if self.robot_start_point is None:
            self.robot_start_point = np.array(point_3d)
            self.env.current_episode.start_position = self.robot_start_point
            self.env.sim.agents_mgr[0].articulated_agent.base_pos = mn.Vector3(self.robot_start_point)
            self.env.sim.agents_mgr[0].articulated_agent.base_rot = -euler_from_quaternion(orientation)[2]
            agent_root = self.env.sim.agents_mgr[0].articulated_agent.sim_obj.transformation
            rotation_quat = mn.Quaternion.from_matrix(agent_root.rotation())
            self.env.current_episode.start_rotation = np.array([0,0,self.env.sim.agents_mgr[0].articulated_agent.base_rot,1])  
            print("Placing robot at ",point_3d)
            return
        elif self.human_start_point is None:
            self.human_start_point = np.array(point_3d)
            self.env.current_episode.info['human_start'] = self.human_start_point
            self.env.sim.agents_mgr[1].articulated_agent.base_pos = mn.Vector3(self.human_start_point)
            self.env.sim.agents_mgr[1].articulated_agent.base_rot = -euler_from_quaternion(orientation)[2]
            agent_root = self.env.sim.agents_mgr[1].articulated_agent.sim_obj.transformation
            rotation_quat = mn.Quaternion.from_matrix(agent_root.rotation())
            self.env.current_episode.info['human_rot'] = np.array([0,0,self.env.sim.agents_mgr[1].articulated_agent.base_rot,1])  
            print("Placing human at ",point_3d)
            return
        
        
        if self.robot_goal_point is not None and self.human_goal_point is not None:
            if self.robot_start_point is not None and self.human_start_point is not None:                
                for eps in self.all_episodes["episodes"]:
                    ep = eps.copy()
                    if ep["scene_id"] == self.env.current_episode.scene_id and ep["info"]["door_start"] == self.env.current_episode.info["door_start"]:
                        ep["start_position"] = self.env.current_episode.start_position
                        ep["start_rotation"] = np.array(self.env.current_episode.start_rotation)
                        ep["info"].update(self.env.current_episode.info)
                        ep["episode_id"] = len(self.new_dataset.episodes)+1
                        self.new_dataset.episodes.append(ep)
                        break
            # a = input("Finished entering episodes? ")
            # if a == 'y':
                with gzip.open(OUTPUT_DATSET_PATH, "wt") as f:
                    f.write(self.new_dataset.to_json())
                self.start_ep = True

def callback(vel, my_env):
    #### Robot Control ####
    my_env.linear_velocity = np.array([(1.0 * vel.linear.y), 0.0, (1.0 * vel.linear.x)])
    my_env.angular_velocity = np.array([0, vel.angular.z, 0])
    my_env.current_point = [0,0,0]

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
    my_env.start()
    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)
    while not rospy.is_shutdown():
   
        my_env.update_agent_pos_vel()
        # rospy.spin()
        my_env._r_control.sleep()
