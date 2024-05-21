import cv2
import numpy as np
import torch
import scipy
import math
import time
import json
import logging, os
import coloring
import random
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path

import argparse
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from scipy.spatial.transform.rotation import Rotation
from typing import Dict, Tuple, List, Union
import skimage.morphology

from explorer.transforms.camera import CameraTransformer
from explorer.transforms.voxel import splat_feature
from explorer.visual import TopDownMap
from explorer_prelude import *
from model import SemanticMoudle
import copy 

from core.interfaces import Observations
import utils.pose as pu
from perception.detection.detic.detic_perception import (
    DeticPerception,
)
from mapping.semantic.instance_tracking_modules import InstanceMemory
from mapping.semantic.instance_map import InstanceMap, InstanceMapState
from mapping.semantic.goat_matching import GoatMatching
# from instance_map import InstanceMap, InstanceMapState
from navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)
from navigation_planner.LPAstarModel import LPAstar
from navigation_planner.nav_planner import DiscretePlanner

from explorer.environ.utils import inspect_environ

from explorer.visual.constants import TOP_DOWN_MAP_COLOR_MAP
TOP_DOWN_MAP_COLOR_MAP['unexplorered'] = [255, 255, 255]

WINDOW_NAME = 'view'
TOPDOWN_MAP_NAME = 'bird-eye-view'
CENTIMETER = int
METER = float
DEGREE = float; RADIUS = float

Scalar = Union[int, float]

# For testing
# intermediate_pts = [
#     np.array([70,162]),
#     np.array([95,157]),
#     np.array([113,150]),
#     np.array([137,142]),
#     np.array([159,143]),
#     np.array([169,170]),
#     np.array([167,184]),
#     np.array([154,190]),
# ]
# intermediate_pts = [
#     np.array([37,166]),
#     np.array([38,170])
# ]

intermediate_pts = [(55, 176), (60, 176), (65, 176), (70, 176), (75, 176), (80, 176), (85, 176), (90, 176), (95, 176), (100, 176), (105, 176), (110, 177), (115, 182), (120, 187), (125, 188), (130, 190), (135, 190), (140, 190), (145, 190), (150, 190), (155, 190), (160, 190), (165, 190), (170, 185), (175, 180), (180, 180)]
target_goal_pt = np.array([180,180])

log_format = '%(asctime)s - %(levelname)s - %(funcName)s : %(message)s'
os.makedirs('datadump/logs', exist_ok=True)

if os.path.isfile('datadump/logs/actor.log'):
    os.remove('datadump/logs/actor.log')
logging.basicConfig(filename='datadump/logs/actor.log',format= log_format,level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

@dataclass
class Memory:
    voxel_map: np.ndarray
    # scene_pixel_features: np.ndarray
    def update_voxel_map(self, new_voxel_map):
        self.voxel_map = np.maximum(self.voxel_map, new_voxel_map) 
            

def quat_to_rotvec(quat) -> Tuple[np.ndarray, DEGREE]:
    # hacky approach to keep the rotation axis as -y, i.e [0, -1, 0]
    rotvec = Rotation.from_quat(quat).as_rotvec(degrees=True)
    rot_angle = np.linalg.norm(rotvec)
    rotvec = rotvec / rot_angle
    if rotvec[1] > 0:
        rotvec *= -1; rot_angle *= -1
    return rotvec, rot_angle

def standardize(v, min_val, max_val):
    '''
    standardize value to range [-1, 1] through linear mapping
    '''
    center = (min_val + max_val) / 2
    return (v - center) / ((max_val - min_val) / 2)

def hook_keystroke() -> str:
    '''
    capture keystroke and outputs corresponding action string
    '''
    while True:

        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # if window is closed by hand
            return 'terminate'

        key_stroke = cv2.waitKey(0)
        if key_stroke == ord('w'):
            return 'move_forward'
        elif key_stroke == ord('a'):
            return 'turn_left'
        elif key_stroke == ord('d'):
            return 'turn_right'
        elif key_stroke == ord('q'):
            cv2.destroyAllWindows()
            return 'terminate'

image_size = (256, 256)
# sensor_pos = 0.25 * SensorPos.LEFT + 1.5 * SensorPos.UP
sensor_pos = 1.5 * SensorPos.UP
rgb_camera = RGBCamera(name='rgb', resolution=image_size, position=sensor_pos)
depth_camera = DepthCamera(name='depth', resolution=image_size, position=sensor_pos)

@register_moves('turn_left', 'turn_right', 'move_forward')
@register_sensors(rgb_camera, depth_camera)

class MyActor(Actor):
    def __init__(self,
                 config,
                 scene_bbox: List[List[METER]],
                 map_resolution: CENTIMETER,
                 local_map_size: List[METER],
                 device_id: int = 0):
        super().__init__()
        self.device_id = device_id
        self.device = torch.device(f"cuda:{self.device_id}")
        self.map_resol = map_resolution
        self.max_steps = [500, 500, 500, 500, 500]

        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 /
                                            config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        self.panorama_rotate_steps = int(360 / config.ENVIRONMENT.turn_angle)

        # TODO: merge the two perception modules
        self.segmentation = DeticPerception(vocabulary="coco", sem_gpu_id=device_id)
        self.semantic_module = SemanticMoudle(vocab='coco', device_id=device_id)  
        self.cate_id_to_name = self.segmentation.category_id_to_name                  
        
        self.num_environments = config.NUM_ENVIRONMENTS
        self.num_sem_categories = len(self.segmentation.categories_mapping)
        self.current_task_idx = None

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.loop = 0
        self.max_instances = 0

        self.record_instance_ids = config.AGENT.SEMANTIC_MAP.record_instance_ids
        # print(self.record_instance_ids)
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
                category_id_to_category_name=self.cate_id_to_name)

        self.goal_policy_config = config.AGENT.SUPERGLUE
        self.matching = GoatMatching(
            device=device_id,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=self.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self.scene_bbox = scene_bbox
        # scene_size in unit of map resolution
        self.scene_size = [
            int(100 * (max_val - min_val) / map_resolution)
            for (min_val, max_val) in scene_bbox
        ]

        self.instance_map = InstanceMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=self.num_sem_categories,
            scene_size=self.scene_size,
            instance_memory=self.instance_memory,
            max_instances=self.max_instances)

        self.instance_map_module = InstanceMap(
            scene_size=self.scene_size,
            num_sem_categories=self.num_sem_categories,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            max_instances=self.max_instances)

        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(np.ceil(agent_radius_cm / self.map_resol))
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.
            obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.
            goal_dilation_selem_radius,
            scene_size=self.scene_size,
            map_resolution=self.map_resol,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.
            min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            scene_bbox = self.scene_bbox
        )

        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device)

        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal = torch.zeros(self.num_environments,
                                      1,
                                      dtype=bool,
                                      device=self.device)
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            self.scene_size[0],
            self.scene_size[2],
            device=self.device,
        )
        self.goal_pose = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering

        self.task_list = []
        ### Get language tasks
        for k, v in config.LANGUAGE_TASKS.items():
            self.task_list.append(v)
        self.key_list = []
        # print(self.segmentation.categories_mapping)
        for k, v in self.segmentation.categories_mapping.items():
            self.key_list.append(k)
        # print(self.key_list)

        ''' from actor.py '''
        self.global_map = np.zeros(
            (self.scene_size[0], self.scene_size[2], self.segmentation.num_sem_categories+1)
        )
        initial_voxel_map = np.zeros((*self.scene_size, self.num_sem_categories+1))
        self.memory_module = Memory(initial_voxel_map)

        # a mutable reference pointing to the global map
        self.local_map = None
        self.local_map_size = local_map_size
        self.map_resol = map_resolution

        # heading convention: -z
        self._spirit_init_rot_correction: DEGREE = -90
        self._prev_pos_index, self._prev_rot = None, None
        self.topdown_map_vis = TopDownMap(
            self.global_map.shape[:2],
            track_trajactory=False
        )
        # query each pixel feature
        self.querier = {
            'semantic': slice(0, self.num_sem_categories),
            'top_down': slice(0, 3, 2)
        }
        self.vis_color_array = np.array([np.array([255, 255, 255], dtype=np.uint8)] + coloring.COCO_COLORING)

        '''Parameters from module'''
        self.instance_goal_found = False
        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy,
            device_id=self.device_id)
        self.goal_update_steps = self.policy.goal_update_steps
        self.goal_policy_config = config.AGENT.SUPERGLUE
        self.goal_inst = None
        '''Parameters from map'''
        self.loop = 0
        self.last_pose = None

        ''' Parameters for path planning '''
        self.stgs = [] # list for short term goals
        self.explored_frontier_goals = []
        self.stop = None
        self.target_goal = None
        self.short_term_goal = None
        self.navigating_to_goal = False
        self.goal_targets = []


    def round_to_one(self, x):
        if x > 0:
            return 1
        elif x == 0:
            return x
        else:
            raise NotImplementedError

    def register_semantic(self, key):
        if key not in self.key_list:
            return 0
        else:
            return key

    def preprocess_obs(self,
                       obs: Observations,
                       task_type: str = "languagenav"):
        rgb = torch.from_numpy(obs.rgb.copy()).to(self.device)
        ### Do not transform the depth
        # depth = (
        #     torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        # )
        depth = (torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device))

        current_task = obs.task_observations["tasks"][
            self.current_task_idx]  # Dict containing task info
        current_goal_semantic_id = current_task["semantic_id"]

        semantic = obs.semantic
        instance_ids = None
        # print(semantic.shape)
        # TODO: key points and matches
        (
            matches,
            confidences,
            keypoints,
            local_instance_ids,
            all_matches,
            all_confidences,
            all_rgb_keypoints,
            instance_ids,
        ) = (None, None, None, None, [], [], [], [])

        # TODO self._module
        if not self.instance_goal_found:
            if task_type == "imagenav":
                if self.goal_image is None:
                    img_goal = obs.task_observations["tasks"][
                        self.current_task_idx]["image"]
                    (
                        self.goal_image,
                        self.goal_image_keypoints,
                    ) = self.matching.get_goal_image_keypoints(img_goal)
                    # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.image_matching_function,
                    self.total_timesteps[0],
                    image_goal=self.goal_image,
                    goal_image_keypoints=self.goal_image_keypoints,
                    categories=[current_task["semantic_id"]],
                    use_full_image=False,
                )

            elif task_type == "languagenav":
                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.matching.match_language_to_image,
                    self.total_timesteps[0],
                    language_goal=current_task["description"],
                    categories=[current_task["semantic_id"]],
                    use_full_image=True,
                )
        # print("matches:", matches)
        # print("confidences:", confidences)
        ### Map semantic to (0,16)
        # print("semantic:", np.unique(semantic))
        semantic = np.vectorize(self.register_semantic)(semantic)
        semantic = np.vectorize(
            self.segmentation.categories_mapping.get)(semantic)
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(
            self.device)]
        # print(semantic.shape)

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        # print(obs_preprocessed.shape)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]

            instance_ids = np.unique(instances)
            # print("instance ids:", instance_ids)
            instance_id_to_idx = {
                instance_id: idx
                for idx, instance_id in enumerate(instance_ids)
            }

            # print("="*20)
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)).to(
                    self.device)
            # print(instances)
            # np.savetxt("instances.txt",instances.cpu().numpy())
            ### 1st channel: background, 2...: instances
            instances = torch.eye(len(instance_ids),
                                  device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)
            # print(obs_preprocessed.shape)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)
        # print(obs_preprocessed.shape)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])).unsqueeze(0)
        self.last_poses[0] = curr_pose

        object_goal_category = torch.tensor(
            current_goal_semantic_id).unsqueeze(0)

        # TODO: camara position
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        if (task_type in ["languagenav", "imagenav"]
                and self.record_instance_ids
                and (self.sub_task_timesteps[0][self.current_task_idx] == 0
                     or self.force_match_against_memory)):
            if self.force_match_against_memory:
                print("Force a match against the memory")
            self.force_match_against_memory = False
            (all_rgb_keypoints, all_matches, all_confidences,
             instance_ids) = self._match_against_memory(
                 task_type, current_task)

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            self.goal_image,
            camera_pose,
            keypoints,
            matches,
            confidences,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        )

    @torch.no_grad()
    def update_instance_memory(
        self,
        obs: torch.Tensor,
        pose: torch.Tensor,
        point_cloud: torch.Tensor,
        top_down_instance_map: torch.Tensor,
        save_3DSG = False
    ):
        '''Update the instance map given the newly observed map (top_down_instance_map)'''
        dones = torch.tensor([False] * self.num_environments)
        # update_global = torch.tensor(
        #     [
        #         self.timesteps_before_goal_update[e] == 0
        #         for e in range(self.num_environments)
        #     ]
        # )

        (
            self.instance_map.instance_map_2d,
            seq_local_pose,
        ) = self.instance_map_module(obs.unsqueeze(1), pose.unsqueeze(1),
                                     dones.unsqueeze(1), point_cloud,
                                     self.instance_map.instance_map_2d,
                                     top_down_instance_map)

        # TODO: update 3DSG
        if save_3DSG:
            instance_list, write_path = self.instance_memory.save_to_3DSG(env_id=0)
            # update pose since the original pose is robot's pose instead of object's pose
            for instance in instance_list:
                updated_pose = self.instance_map_module.compute_coordinates(
                    self.instance_map.instance_map_2d.clone(),
                    env_id = 0,
                    category_id = instance['category_id'],
                    global_instance_id = instance['instance_id']
                )
                instance['position'] = (updated_pose.tolist() if updated_pose.any().item() else None)
            # save to json file
            with open(write_path, 'w', encoding='utf-8') as f:
                json.dump(instance_list, f, ensure_ascii=False, indent=4)

    def module_forward(
        self,
        category_map: torch.Tensor,  # (H, W)
        explored_map: torch.Tensor,  # (H, W)
        seq_found_goal: torch.Tensor,  # (num_env, 1)
        seq_goal_map: torch.Tensor,  # (num_env, 1, H, W)
        seq_object_goal_category=None,  # (1, 1)
        reject_visited_targets=False,
        blacklist_target=False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        seq_obstacle_locations=None,
        seq_free_locations=None,
        batch_size = 1,
        sequence_length = 1
    ):
        # TODO: the map is not stale enough, need to fix this issue
        # stale_local_id_to_global_id_map = self.instance_memory.local_id_to_global_id_map.copy()
        stale_local_id_to_global_id_map = self.instance_memory.last_local_id_to_global_id_map.copy(
        )
        print("local to global:", stale_local_id_to_global_id_map)

        frontier_map = self.policy.get_frontier_map_from_explored(explored_map)
        # plt.subplot(111)
        # plt.imshow(frontier_map[0,0].cpu().numpy(), cmap='Greys',  interpolation='nearest')
        # plt.savefig('datadump/images/debug/frontier.png')

        if seq_found_goal[0, 0] == 0:  # Goal not found
            # print("seq_found_goal:", seq_found_goal)
            seq_goal_map = frontier_map
        seq_goal_pose = None

        instance_map = self.instance_map.instance_map_2d

        if len(all_matches) > 0 or matches is not None or self.instance_goal_found:
            # print("=== select and localize instance ===")
            (
                seq_goal_map,
                seq_found_goal,
                seq_goal_pose,
                self.instance_goal_found,
                self.goal_inst,
            ) = self.matching.select_and_localize_instance(
                seq_goal_map,
                seq_found_goal,
                instance_map,
                matches,
                confidence,
                local_instance_ids,
                stale_local_id_to_global_id_map,
                self.instance_goal_found,
                self.goal_inst,
                all_matches=all_matches,
                all_confidences=all_confidences,
                instance_ids=instance_ids,
                score_thresh=score_thresh,
            )
            # TODO: reshape goal map
            seq_goal_map = seq_goal_map.view(batch_size, sequence_length, *seq_goal_map.shape[-2:])
        else:
            # Predict high-level goals from map features
            # batched across sequence length x num environments
            if seq_object_goal_category is not None:
                seq_object_goal_category = seq_object_goal_category.flatten(
                    0, 1)

            # Compute the goal map
            print("=== Compute goal map from policy ===")
            goal_map, found_goal = self.policy(
                category_map,
                explored_map,
                seq_object_goal_category,
                reject_visited_targets=reject_visited_targets,
            )

            ### TODO: reshape goal map and found goal to batch size 1
            # assert goal_map.shape[0] == 1
            assert goal_map.shape[0] == self.scene_size[0]
            assert goal_map.shape[1] == self.scene_size[2]
            # seq_goal_map = goal_map
            # seq_found_goal = found_goal
            seq_goal_map = goal_map.view(
                batch_size, sequence_length, *goal_map.shape[-2:]
            )
            seq_found_goal = found_goal.view(batch_size, sequence_length)

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )
        # seq_frontier_map = frontier_map
        assert seq_frontier_map.shape[0] == 1 and seq_frontier_map.shape[1] == 1
        assert seq_frontier_map.shape[2] == self.scene_size[0]
        assert seq_frontier_map.shape[3] == self.scene_size[2]
        # print("seq_goal_map shape:", seq_goal_map.shape)
        # print("seq_found_goal shape:", seq_found_goal.shape)
        # print("seq_goal_pose:", seq_goal_pose)
        # print("seq_frontier_map shape:", seq_frontier_map.shape)

        return (seq_goal_map, seq_found_goal, seq_goal_pose, seq_frontier_map)

    def get_planner_inputs(
        self,
        pose: torch.Tensor,
        explored_map: torch.Tensor,
        obstacle_map: torch.Tensor,
        category_map: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
    ):
        dones = torch.tensor([False] * self.num_environments)

        if obstacle_locations is not None:
            obstacle_locations = obstacle_locations.unsqueeze(1)
        if free_locations is not None:
            free_locations = free_locations.unsqueeze(1)
        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
            # print("object_goal_category:", object_goal_category)
        (
            self.goal_map,
            self.found_goal,
            self.goal_pose,
            frontier_map,
        ) = self.module_forward(
            category_map,
            explored_map,
            self.found_goal,
            self.goal_map,
            seq_object_goal_category=object_goal_category,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            seq_obstacle_locations=obstacle_locations,
            seq_free_locations=free_locations,
        )

        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if self.found_goal[0].item():  ### TODO: This is where we stop
            goal_map = self._prep_goal_map_input()  # Optional clustering

        # found_goal = self.found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            # if frontier_map is not None:
            #     self.semantic_map.update_frontier_map(
            #         e, frontier_map[e][0].cpu().numpy()
            #     )
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                # self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[
                        e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            self.sub_task_timesteps[e][self.current_task_idx] += 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1)

        # TODO:
        # print("goal_map shape:", goal_map.shape)
        assert frontier_map.shape[0] == 1 and goal_map.shape[
            0] == 1 and self.found_goal.shape[0] == 1
        assert frontier_map.shape[1] == 1 and self.found_goal.shape[1] == 1
        assert frontier_map.shape[2] == self.scene_size[0] and goal_map.shape[
            1] == self.scene_size[0]
        assert frontier_map.shape[3] == self.scene_size[2] and goal_map.shape[
            2] == self.scene_size[2]
        # print("obstacle_map shape:", obstacle_map.shape)
        print("found goal:", self.found_goal)
        logging.debug(f'found goal: {self.found_goal}')
        planner_inputs = [
            {
                "obstacle_map":
                obstacle_map.cpu().numpy(),  # Need to specify env
                "goal_map":
                goal_map[0],
                "frontier_map":
                frontier_map[e][0].cpu().numpy(),
                "sensor_pose":
                pose.cpu().numpy(),
                "found_goal":
                self.found_goal[e].item(),
                "goal_pose":
                self.goal_pose[e] if self.goal_pose is not None else None
            } for e in range(self.num_environments)
        ]
        vis_inputs = None
        # if self.visualize:
        #     vis_inputs = [
        #         {
        #             "explored_map": self.semantic_map.get_explored_map(e),
        #             "semantic_map": self.semantic_map.get_semantic_map(e),
        #             "been_close_map": self.semantic_map.get_been_close_map(e),
        #             "timestep": self.total_timesteps[e],
        #         }
        #         for e in range(self.num_environments)
        #     ]
        #     if self.record_instance_ids:
        #         for e in range(self.num_environments):
        #             vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
        #                 e
        #             )
        # else:
        #     vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

    def check_scattered_property(
            self, 
            point_coord: list, 
            goal_list: list,
            scalar = -1,
            step_size = 5):
        ''' 
        This function is for checking whether the point of interest in to close to the explored point in frontier exploration.
        It will return True if the point to too close. Otherwise return False.
        '''
        if scalar == -1:
            distance = step_size*5
        else:
            distance = scalar
        for pt in goal_list:
            if self.mahantan_dist(*pt, *point_coord, distance):
                return True
        return False

    def find_path(
            self, 
            goal_map: np.ndarray, 
            traversible_map: np.ndarray,
            start_point: np.ndarray,
            found_goal = False,
            step_size = 5):
        '''
        Input:
            goal_map: binary array of shape (H x W) containing the goal information (there might be multiple goal targets)
            traversible_map: binary array of shape (H x W) containing information of all traversible points (pixel value equals 1 if traversible)
            start_point: the starting position [x1, y1, o1] of the robot agent
        Output:
            short_term_goals: a list of stg
            target_goal: [x_g, y_g] the chosen goal point to navigate to
            stop: a binary flag indicating the stop signal for robot
            no_path: a binary flag indicating no path was found
        '''
        no_path=False
        stop_flag=False
        next_point=tuple(start_point)
        target_goal=tuple(start_point)
        list_target_goal=[]
        list_path=[]
        indices = np.argwhere(goal_map == 1)
        indices_list = [ list(index) for index in indices ]
        short_term_goals = []
        assert len(indices) == len(indices_list)

        # frontier exploration: we would like to explore points scattered around
        # FIXME: time complexity O(n) too slow!
        # TODO: we could further optimize the indices_list
        indices_list = indices_list[::step_size]
        if not found_goal:
            for element in indices_list:
                # TODO: we could incoporate object location into the list for checking
                if self.check_scattered_property(element, self.explored_frontier_goals):
                    indices_list.remove(element)
        elif len(self.goal_targets) == 0:
            self.goal_targets = indices_list
            
        if len(indices) > 3:
            list_target_goal = random.sample(indices_list,3)
        elif len(indices) > 1:
            list_target_goal = copy.deepcopy(indices_list)
        elif len(indices) == 1:
            list_target_goal.append(indices_list[0])
        # LPAstarlite=None
        # print("list_target_goal:", list_target_goal)
        for point in list_target_goal:
            # print(point)
            # print(start_point)
            if self.mahantan_dist(*point, *start_point, 2):
                continue
            try:
                LPAstarlite = LPAstar.LPAStar(tuple(start_point),tuple(point),traversible_map, "manhattan")
                list_path.append(copy.deepcopy(LPAstarlite.run()))
            except:
                list_path.append([])
        # print(list_path)
        filtered_list = list(filter(bool, list_path))
        # print(filtered_list)
        if len(filtered_list)==0 or len(min(filtered_list, key=len))==1:
            no_path=True
            stop_flag=True
        else:
            # TODO: need better optimization     
            short_term_goals = min(filtered_list, key=len)
            next_point=tuple(min(filtered_list, key=len)[1])
            target_goal=tuple(min(filtered_list, key=len)[len(min(filtered_list, key=len))-1])
            self.explored_frontier_goals += short_term_goals[::step_size*2] + [target_goal]
        # plt.subplot(1,1,1)
        # LPAstarlite.Plot.plot_grid("Lifelong Planning A*")
        # LPAstarlite.Plot.plot_path(min(filtered_list, key=len))
        # plt.savefig('datadump/images/Astar.png')
        # plt.show()
        return next_point, target_goal, short_term_goals[::step_size*2] + [target_goal], no_path, stop_flag

    def compute_action(
            self,
            start_pose,  # [x1, y1, o1]
            next_point,  # [x2, y2]
            target_point,  # [x_g, y_g]
            found_goal: bool,
            stop: bool,
            goal_pose=None,
            debug=False):
        # Normalize agent angle
        angle_agent = pu.normalize_angle(start_pose[2])

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = next_point
        relative_stg_x, relative_stg_y = stg_x - start_pose[0], stg_y - start_pose[1]
        angle_st_goal = 180 - math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(-(angle_agent - angle_st_goal))
        
        logging.debug(f"start: {start_pose}")
        logging.debug(f"next: {next_point}")
        logging.debug(f'angle_st_goal: {angle_st_goal}')
        logging.debug(f'relative_angle_to_stg: {relative_angle_to_stg}')
        if debug:
            print("start:",start_pose)
            print("next:", next_point)
            # print('angle_agent:', angle_agent)
            print('angle_st_goal:', angle_st_goal)
            print('relative_angle_to_stg:', relative_angle_to_stg)
        # Compute angle to the final goal
        goal_x, goal_y = target_point
        angle_goal = 180 - math.degrees(math.atan2(goal_x - start_pose[0], goal_y - start_pose[1]))

        if goal_pose is None:
            # Compute angle to the final goal
            relative_angle_to_closest_goal = pu.normalize_angle(-(angle_agent - angle_goal))
            logging.debug(f'angle_goal: {angle_goal}')
            logging.debug(f'relative_angle_to_closest_goal: {relative_angle_to_closest_goal}')
            if debug:
                print('angle_goal:', angle_goal)
                print('relative_angle_to_closest_goal:', relative_angle_to_closest_goal)
        else:
            relative_angle_to_closest_goal = pu.normalize_angle(angle_agent - goal_pose)

        action = self.planner.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            start_pose[2],
            found_goal,
            stop,
            debug,
        )
        return action

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [[0] * self.max_num_sub_task_episodes
                                   ] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.instance_map.init_instance_map()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False

        # if self.imagenav_visualizer is not None:
        #     self.imagenav_visualizer.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()
        ''' module reset '''
        # self._module.reset()
        self.instance_goal_found = False
        self.goal_inst = None

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

        self.instance_goal_found = False
        self.goal_inst = None

    def _match_against_memory(self, task_type: str, current_task: Dict):
        print("--------Matching against memory!--------")
        if task_type == "languagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.matching.match_language_to_image,
                self.total_timesteps[0],
                language_goal=current_task["description"],
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "all": scores.flatten().tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                    f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_language_stats.json",
                    "w",
            ) as f:
                json.dump(stats, f, indent=4)

        elif task_type == "imagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.image_matching_function,
                self.sub_task_timesteps[0][self.current_task_idx],
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.sum(axis=1).mean()),
                    "median": float(np.median(scores.sum(axis=1))),
                    "max": float(scores.sum(axis=1).max()),
                    "min": float(scores.sum(axis=1).min()),
                    "all": scores.sum(axis=1).tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                    f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_image_stats.json",
                    "w",
            ) as f:
                json.dump(stats, f, indent=4)

        return all_rgb_keypoints, all_matches, all_confidences, instance_ids

    def _pos2index(self, coord: Union[str, List[str]],
                   value: Union[Scalar, np.ndarray]):
        '''
        discretize coordinate in unit of meter to integer coordinate at a given granularity (map_resolution)
        the possible range of value of each axis is known forehead via inspect_environ API provided by explorer
        '''
        coord_str2min = {
            s: v
            for (s, (v, _)) in zip(['x', 'y', 'z'], self.scene_bbox)
        }
        if type(value) == np.ndarray:
            coord_min = np.array([
                coord_str2min[axis]
                for axis in (coord if type(coord) == list else list(coord))
            ])
            return (100 * (value - coord_min) / self.map_resol).astype(
                np.int32)
        else:
            return int(100 * (value - coord_str2min[coord]) / self.map_resol)

    def _splat_pixel_features(self, pixel_features: np.ndarray,
                              coords: np.ndarray,
                              map_size: List[int]) -> np.ndarray:
        '''
        splat features associated with the image the agent currently sees to a global 3D voxel
        say the image the agent extracts is of size (h, w), global map is of size (X, Y, Z)
        Args:
            pixel_features: (h, w, F) each pixel associates with F feature
            coords: (h, w, 3) coordinate associated with each pixel of the voxel map it will be splatted to, value in unit of meter
            map_size: (X, Y, Z) size of voxel map feature will be splatted to 
        
        Return:
            global feature grid (X, Y, Z, F)
        '''
        # normalize pixel coordinate in unit of meter to range [-1, 1] to accommondate the prototype of explorer's splat_feature func
        coord_min, coord_max = map(np.array, zip(*self.scene_bbox))
        normed_coords = standardize(coords, coord_min, coord_max)

        F = pixel_features.shape[-1]
        voxel_map = np.zeros((1, *map_size, F))

        # operation is taken in place
        splat_feature(pixel_features.reshape(1, -1, F),
                      normed_coords.reshape(1, -1, 3), voxel_map)
        return voxel_map.squeeze(axis=0)

    def score_thresh(self, task_type):
        # If we have fully explored the environment, set the matching threshold to 0.0
        # to go to the highest scoring instance
        if self.fully_explored[0]:
            return 0.0

        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def _update_topdown_map_vis(self,
                                agent_pos: List[METER],
                                agent_rot: DEGREE):
        # project 3D voxel map along height(y) axis to form a 2D top down map
        voxel_topdown_projection = np.mean(self.memory_module.voxel_map, axis=1)
        topdown_map_pixel_cate = np.argmax(voxel_topdown_projection, axis=-1)
        # print('pixel classes:', np.unique(topdown_map_pixel_cate))
        # print('voxel shape:', voxel_topdown_projection.shape)
        # print('topdown shape:', topdown_map_pixel_cate.shape)
        # print(topdown_map_pixel_cate)

        # TODO update topdown map pixel category
        self.instance_map.topdown_category[0] = self.instance_map_module.update_topdown_cat(
            self.instance_map.topdown_category[0], 
            torch.tensor(topdown_map_pixel_cate, device=self.device))
        # print(self.instance_map.topdown_category[0].cpu().numpy())
        pixel_coloring = self.vis_color_array[self.instance_map.topdown_category[0].cpu().numpy().astype(int)] 

        agent_pos_index = np.array([self._pos2index(axis, v) for v, axis in zip(agent_pos, ['x', 'z'])])
        agent_rot = self._spirit_init_rot_correction + agent_rot

        # we'd better change this api of subsitituting raw_map afterwards
        self.topdown_map_vis.raw_map = pixel_coloring[:,:,::-1]

        if self._prev_pos_index is None:
           self.topdown_map_vis.init_sprite_at(agent_pos_index, agent_rot) 
           self._prev_pos_index, self._prev_rot = agent_pos_index, agent_rot
        else:
            dxy, dtheta = agent_pos_index - self._prev_pos_index, agent_rot - self._prev_rot
        #    if np.sum(abs(dxy)) > 1e-2 or abs(dtheta)> 1e-2:
            self.topdown_map_vis.move_sprite(dxy, dtheta)
            self._prev_pos_index, self._prev_rot = agent_pos_index, agent_rot
    
    def _compute_local_voxel_map_index(self, location: List[METER]) -> Tuple[np.ndarray, np.ndarray]:
        # location in unit of meters
        (x_min, x_max), (y_min, _), (z_min, z_max)= self.scene_bbox
        x_range, z_range = self.local_map_size

        cur_x, cur_z = location[self.querier['top_down']]
        origin_x, origin_z = max(x_min, cur_x - x_range/2), max(z_min, cur_z - z_range/2)
        x_span, z_span = min(x_range, x_max - origin_x), min(z_range, z_max - origin_z)

        # statically configure local range along y axis, ranging from y_min and stretches over 2.5m
        origin_y, y_span = y_min, 2.5
        return np.array([origin_x, origin_y, origin_z]), np.array([x_span, y_span, z_span])

    def step(self, observes: Dict) -> str:
        depth_image = np.clip(observes['depth'], 0, 10) / 10 * 255
        rgb_image = observes['rgb'][..., :3][..., ::-1]

        depth_image = np.repeat(depth_image[..., None].astype(np.uint8),
                                3,
                                axis=-1)

        agent_view = np.concatenate([rgb_image, depth_image], axis=1)

        obs = Observations(
            gps=[7, 15],  ### TODO
            compass=[60],  ### TODO
            rgb=rgb_image,
            depth=observes['depth'],
            task_observations={"tasks": self.task_list},
            camera_pose=None,
            third_person_image=None)

        obs = self.segmentation.predict(obs)
        seg_res = self.semantic_module.predict(rgb_image)
        # print("instance_classes",obs.task_observations["instance_classes"])
        # print("instance_classes",obs.task_observations["instance_scores"])
        agent_view = np.concatenate(
            [depth_image, obs.task_observations["semantic_frame"]], axis=1)

        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]

        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            img_goal,
            camera_pose,
            keypoints,
            matches,
            confidence,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        ) = self.preprocess_obs(obs, task_type)

        depth_camera_state = self.get_sensors_state()['depth']

        camera_trans = CameraTransformer.from_instance(
            depth_camera, depth_camera_state.position,
            depth_camera_state.rotation)
        rotvec, rot_in_deg = quat_to_rotvec(depth_camera_state.rotation)
        # print('rotvec: ', rotvec)
        # print('angle: ', rot_in_deg)

        point_cloud = camera_trans.depth2point_cloud(observes['depth'],
                                                     heading_axis='-z')
        point_cloud_world_frame = camera_trans.camera2world(point_cloud)
        point_cloud_xz_index = self._pos2index(
            ['x', 'z'], point_cloud_world_frame[:,:,self.querier['top_down']]
        )
        point_cloud_x, point_cloud_z = point_cloud_xz_index[:,:,0], point_cloud_xz_index[:,:,1]
        # print("depth camera position:", depth_camera_state.position)
        agent_pos_index = np.array([
            self._pos2index(axis, v) for v, axis in zip(
                depth_camera_state.position[self.querier['top_down']],
                ['x', 'z'])
        ])
        # print("Agent pose:", agent_pos_index)
        agent_pose = np.array([*agent_pos_index, rot_in_deg, *depth_camera_state.position[self.querier['top_down']]])
        logging.debug(f"Agent pose with angle: {agent_pose}")
        print("Agent pose with angle:", agent_pose)

        batch_size, obs_channels, h, w = obs_preprocessed.size()
        num_instance_channels = obs_channels - 4 - self.num_sem_categories
        # print("obs:",obs_preprocessed.shape)
        # print("num_instance_channels:", num_instance_channels)
        instance_global_map = np.zeros(
            (self.scene_size[0], self.scene_size[2], num_instance_channels))
        instance_channels = obs_preprocessed[
            :,
            4 + self.num_sem_categories:4 + self.num_sem_categories +
            num_instance_channels,
            :,
            :,
        ][0].permute(1, 2, 0).cpu().numpy()

        # Voxel map for instance map
        # instance_channels: a binary matrix with shape [256, 256, num_instance_channels + 1]
        cur_frame_voxel_map = self._splat_pixel_features(instance_channels, point_cloud_world_frame, self.scene_size)
        # print('voxel shape:',cur_frame_voxel_map.shape)

        voxel_topdown_projection = np.mean(cur_frame_voxel_map, axis=1)
        voxel_topdown_projection = np.clip(voxel_topdown_projection, 0, 1)
        voxel_topdown_projection_one_hot = np.vectorize(self.round_to_one)(voxel_topdown_projection)
        # TODO: need to better process it
        voxel_topdown_projection_aggre = np.argmax(voxel_topdown_projection, axis=-1)
        
        # Voxel map for category map
        cur_frame_voxel_map_cat = self._splat_pixel_features(seg_res['pixel_features'], point_cloud_world_frame, self.scene_size)
        # print(cur_frame_voxel_map_cat.shape)
        # self.memory_module.update_voxel_map(cur_frame_voxel_map_cat)
        self.memory_module.voxel_map = cur_frame_voxel_map_cat

        # Update topdown vis
        top_down_pos = depth_camera_state.position[self.querier['top_down']]
        self._update_topdown_map_vis(top_down_pos, rot_in_deg)

        # Update obstacle map
        self.instance_map.obstacle_map_2d[0] = self.instance_map_module.update_obstacle_map(
            self.instance_map.obstacle_map_2d[0],
            torch.tensor(np.vectorize(self.round_to_one)(voxel_topdown_projection_aggre), device=self.device)
        )

        # plt.subplot(1,1,1)
        # plt.imshow(self.instance_map.obstacle_map_2d[0].cpu().numpy(), cmap='Greys', interpolation='nearest')
        # plt.savefig('datadump/images/obstacle.png')

        # TODO: leave out those detected but unprojected instances
        # print("topdown one hot shape:", voxel_topdown_projection_one_hot.shape)
        # print("topdown elements:", np.unique(voxel_topdown_projection_aggre))
            
        self.update_instance_memory(obs_preprocessed, torch.tensor(agent_pose),
                                    torch.tensor(point_cloud),
                                    torch.tensor(voxel_topdown_projection_one_hot),
                                    save_3DSG = True)

        ### TODO: category map: need to -1 when using to keep everything consistent
        category_map_tmp = self.instance_map.topdown_category[0].clone()
        # print('category_map_tmp shape:', category_map_tmp.shape)
        planner_inputs, vis_inputs = self.get_planner_inputs(
            torch.tensor(agent_pose),
            torch.tensor(self.planner.visited_map, device=self.device),
            self.instance_map.obstacle_map_2d[0].clone(),
            category_map_tmp.clone(),
            object_goal_category=object_goal_category,
            reject_visited_targets=self.reject_visited_targets,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
        )
        # Planning
        closest_goal_map = None
        dilated_obstacle_map = None
        # short_term_goal = None
        could_not_find_path = False
        if planner_inputs[0]["found_goal"]:  # If we found goal in the first few steps, then no need to turn around
            self.episode_panorama_start_steps = 0
        if self.total_timesteps[0] < self.episode_panorama_start_steps:
            print('panaroma starts')
            action = "turn_right"
        else:
            # print("=== prepare goal inputs ===")
            (traversible_map, goal_map, start_pose,
             dilated_obstacle_map) = self.planner.prepare_goal_inputs(
                 **planner_inputs[0],
                 use_dilation_for_stg=self.use_dilation_for_stg,
                 timestep=self.sub_task_timesteps[0][self.current_task_idx],
                 debug=False)
            ### TODO: if we found the goal, store the goal map via a class parameter
            ### TODO: to check if we reached the goal: 
            ### TODO: 1. get all indices 2. if the current pose is close to any of the indices, we stop 3. set that index to the current target goal
            # plt.subplot(2,1,1)
            # plt.imshow(traversible_map, cmap='Greys', interpolation='nearest')

            # plt.subplot(2,1,2)
            # plt.imshow(goal_map, cmap='Greys', interpolation='nearest')
            # plt.savefig('datadump/images/traversible_goal.png')

            # if we found a stg and a path, we do not need to calculate the shortest path again
            # we only need to re-find the path if: 1) we reached the stg 2) a collision occurred
            # TODO: how to deal with collided situation???
            logging.debug('collided info: {}'.format(observes['collided']))
            logging.debug(f'self.stgs: {self.stgs}')
            logging.debug(f'stop: {self.stop}')
            logging.debug(f'short term goal: {self.short_term_goal}')
            logging.debug(f'target goal: {self.target_goal}')

            if (len(self.stgs) > 0) and not self.stop and not observes['collided']:
                self.short_term_goal = self.stgs[0]
                if self.mahantan_dist(*start_pose[:2], *self.stgs[0], 2):
                    self.stgs.remove(self.stgs[0]) 
                # If we found the goal in the middle of the our way, we redirect the path to the goal
                if planner_inputs[0]["found_goal"] and not self.navigating_to_goal:
                    # print('[...redirecting...]')
                    self.navigating_to_goal = True
                    (
                        next_point,
                        self.target_goal,
                        self.stgs,
                        could_not_find_path,
                        self.stop
                    ) = self.find_path( 
                        goal_map,
                        traversible_map,
                        start_pose[:2],
                        planner_inputs[0]["found_goal"]
                    )
                    self.short_term_goal = self.stgs[0]
                elif planner_inputs[0]["found_goal"] and self.navigating_to_goal:
                    ### TODO: we need to check the property under collision
                    # TODO: If we are navigating to a found goal, we would like to check if we reach any of the goal index
                    # by checking the mahattan distance
                    # print('[...checking close to goal...]')
                    if self.check_scattered_property(start_pose[:2], self.goal_targets, 3):
                        self.stop = True 

                # print("self.stgs:",self.stgs)
                # print('short term goal:', self.short_term_goal)
            elif planner_inputs[0]["found_goal"] and self.navigating_to_goal:
                ### TODO: We need to add extra collision situation
                # print('[...debugging elif...]')
                # if we found the goal and the lenghth of short term goal list is equal to 0, we stop (we reach the goal)
                if len(self.stgs) == 0:
                    self.stop = True
                elif self.mahantan_dist(*start_pose[:2], *self.short_term_goal, 2):
                    self.stgs.remove(self.stgs[0])

            else: # TODO
                logging.debug('Finding a path to a goal.')
                (
                    next_point,
                    self.target_goal,
                    self.stgs,
                    could_not_find_path,
                    self.stop
                ) = self.find_path( 
                    goal_map,
                    traversible_map,
                    start_pose[:2],
                    planner_inputs[0]["found_goal"]
                )
                self.short_term_goal = self.stgs[0]

            
            # TODO!: bugs have been reported here: the robot moves around without navigating to the goal
            if len(self.stgs) > 0 and self.mahantan_dist(*start_pose[:2], *self.short_term_goal, 2):
                logging.debug('The agent has reached a short term goal.')
                action = 'turn_right'
             ### FIXME: If we could not find a path to a frontier goal, we need to repeat the process in find_path
            elif could_not_find_path:
                if planner_inputs[0]["found_goal"]:
                    logging.warning('Could not find a path to object goals!')
                    logging.warning('Navigation terminated!')
                    action = 'terminate'
                else:
                    logging.warning('Could not find a path to frontier goals!')
                    logging.warning('Refinding a path to frontier goals...')
                    cnt = 0
                    while True: # We repeatedly find a traversible frontier goal
                        (
                            next_point,
                            self.target_goal,
                            self.stgs,
                            could_not_find_path,
                            self.stop
                        ) = self.find_path( 
                            goal_map,
                            traversible_map,
                            start_pose[:2],
                            planner_inputs[0]["found_goal"]
                        )
                        self.short_term_goal = self.stgs[0]
                        if self.stgs is not None:
                            break
                        elif cnt > 15:
                            logging.warning('Map has been fully explored?!')
                            action = 'Terminate'
                            break
                        cnt += 1
            else:
                action = self.compute_action(
                    start_pose,
                    self.short_term_goal,
                    self.target_goal,
                    found_goal=planner_inputs[0]["found_goal"],
                    stop=self.stop,
                    goal_pose=planner_inputs[0]["goal_pose"],
                    debug=False
                )
            self.planner.last_action = action

        # action = None
        if (self.sub_task_timesteps[0][self.current_task_idx]
                >= self.max_steps[self.current_task_idx]):
            print("Reached max number of steps for subgoal, calling STOP")
            action = "terminate"

        if could_not_find_path and not self.stop and action != "terminate":
            # TODO: is this accurate?
            loggine.warning("Can't find a path. Map fully explored.")
            print("Can't find a path. Map fully explored.")
            self.fully_explored[0] = True
            self.force_match_against_memory = True
            ## TODO: need to terminate after matching against memory

        if action == "terminate":
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:  # There are more than one tasks
                logging.debug('Task: [{}] is completed.'.format(obs.task_observations["tasks"][self.current_task_idx]['description']))
                print('Task: [{}] is completed.'.format(obs.task_observations["tasks"][self.current_task_idx]['description']))
                self.current_task_idx += 1
                logging.debug(f'Resetting config for task {self.current_task_idx}...')
                print(f'Resetting config for task {self.current_task_idx}...')
                time.sleep(10)
                self.force_match_against_memory = False
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(self.num_environments,
                                              1,
                                              dtype=bool,
                                              device=self.device)
                self.reset_sub_episode()
                # keep everything ative
                action = 'turn_right'
            else:
                print('=== Episode terminated ===')

        self.prev_task_type = task_type
        logging.info("========== total timestep: {} ==========".format(self.total_timesteps))
        print("======= total timestep: {} =======".format(self.total_timesteps))
        # print("collided info:", observes['collided'])
        cv2.imshow('view', agent_view)
        cv2.imshow(TOPDOWN_MAP_NAME, self.topdown_map_vis.raw_map)
        cv2.waitKey(10)
        time.sleep(0.5)
        # action = hook_keystroke()
        # start_po = np.array([*agent_pos_index, rot_in_deg])
        # if self.last_pose is not None and self.mahantan_dist(*start_po[:2], *intermediate_pts[self.loop], 3) and self.loop < len(intermediate_pts):
        #     print("=== LOOP +1: we reach intermediate point ===")
        #     self.loop += 1

        # if self.loop == len(intermediate_pts):
        #     action = 'terminate'
        # else:
        #     action = self.compute_action(
        #         start_pose=start_po,
        #         next_point=intermediate_pts[self.loop],
        #         target_point=target_goal_pt,
        #         found_goal=True,
        #         stop=False,
        #         goal_pose=None,
        #         debug=False
        #     )
        # # TODO: last action
        # self.planner.last_action = action
        # self.last_pose = start_po
    
        return action

    def mahantan_dist(self, x1, y1, x2, y2, threshold=1):
        # print(abs(x1-x2) <= threshold and (y1-y2) <= threshold)
        return abs(x1-x2) <= threshold and abs(y1-y2) <= threshold

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        # print("self.goal_map shape:", self.goal_map.shape)
        # print("self.found_goal", self.found_goal)
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map
        # print("goal_map shape[0] in [_pre_goal]", goal_map.shape[0])
        for e in range(goal_map.shape[0]):
            # print("e:", e)
            if not self.found_goal[e]:
                continue

            # cluster goal points
            try:
                c = DBSCAN(eps=4, min_samples=1)
                data = np.array(goal_map[e].nonzero()).T
                c.fit(data)

                # mask all points not in the largest cluster
                mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
                mode_mask = (c.labels_ != mode).nonzero()
                x = data[mode_mask]
                goal_map_ = np.copy(goal_map[e])
                goal_map_[x] = 0.0

                # adopt masked map if non-empty
                if goal_map_.sum() > 0:
                    goal_map[e] = goal_map_
            except Exception as e:
                print(e)
                return goal_map

        return goal_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="nav_config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()
    baseline_config = OmegaConf.load(args.baseline_config_path)
    # record_instance_id = getattr(baseline_config.AGENT.SEMANTIC_MAP, "record_instance_id", False)

    env_path = '/home/planner/dataset/MP3D/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'
    scene_info = inspect_environ(env_path)
    # print('scene_info:',scene_info)
    logging.info(f'scene_info: {scene_info}')
    actor = MyActor(baseline_config, scene_info.nav_bbox, 5, (1, 1.5), 3)
    actor.reset()
    actors = [actor]
    envs = [Env(env_path)]
    run_simulation(actors, envs)
