import cv2
import numpy as np
import torch
import json
from pathlib import Path

import argparse
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from scipy.spatial.transform.rotation import Rotation
from typing import Dict, Tuple, List, Union


from explorer.transforms.camera import CameraTransformer
from explorer.transforms.voxel import splat_feature
from explorer.visual import TopDownMap
from explorer_prelude import *

from core.interfaces import Observations
import utils.pose as pu
from perception.detection.detic.detic_perception import (
            DeticPerception,
        )
from mapping.semantic.instance_tracking_modules import InstanceMemory
from goat_matching import GoatMatching
from instance_map import InstanceMap, InstanceMapState

from explorer.environ.utils import inspect_environ

from explorer.transforms.camera import CameraTransformer
from explorer.transforms.voxel import splat_feature
from explorer.visual import TopDownMap
from explorer.visual.constants import TOP_DOWN_MAP_COLOR_MAP 
TOP_DOWN_MAP_COLOR_MAP['unexplorered'] = [255, 255, 255]

WINDOW_NAME = 'view'
TOPDOWN_MAP_NAME = 'bird-eye-view'
CENTIMETER = int
METER = float
DEGREE = float; RADIUS = float

Scalar = Union[int, float]

@dataclass
class Memory:
    voxel_map: np.ndarray
    # scene_pixel_features: np.ndarray
    def update_voxel_map(self, new_voxel_map):
        np.maximum(self.voxel_map, new_voxel_map, self.voxel_map) 


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
sensor_pos = 0.25 * SensorPos.LEFT + 1.5 * SensorPos.UP
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
        self.segmentation = DeticPerception(vocabulary="coco", sem_gpu_id=device_id)
        self.num_environments = config.NUM_ENVIRONMENTS
        self.num_sem_categories = len(self.segmentation.categories_mapping)
        self.current_task_idx = None

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)
        
        self.instance_goal_found = False
        
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
                padding_cropped_instances=200
            )

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
        self.scene_size = [int(100*(max_val - min_val)/map_resolution) for (min_val, max_val) in scene_bbox]

        self.instance_map = InstanceMapState(
            device = self.device,
            num_environments = self.num_environments,
            num_sem_categories = self.num_sem_categories,
            scene_size = self.scene_size,
            instance_memory = self.instance_memory,
            max_instances = self.max_instances
        )

        self.instance_map_module = InstanceMap(
            scene_size = self.scene_size,
            num_sem_categories = self.num_sem_categories,
            du_scale = config.AGENT.SEMANTIC_MAP.du_scale,
            record_instance_ids = self.record_instance_ids,
            instance_memory = self.instance_memory,
            max_instances = self.max_instances
        )

        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.total_timesteps = None
        self.sub_task_timesteps = None
        self.last_poses = None
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.task_list = []
        ### Get language tasks
        for k,v in config.LANGUAGE_TASKS.items():
            self.task_list.append(v)
        self.key_list = []
        # print(self.segmentation.categories_mapping)
        for k,v in self.segmentation.categories_mapping.items():
            self.key_list.append(k)
        # print(self.key_list)
        self.map_resol = map_resolution

        # query each pixel feature
        self.querier = {
            'semantic': slice(0, self.num_sem_categories),
            'top_down': slice(0, 3, 2)
        }
    
    def register_semantic(self, key):
        if key not in self.key_list:
            return 0
        else:
            return key
          
    def preprocess_obs(self, obs: Observations, task_type: str = "languagenav"):
        rgb = torch.from_numpy(obs.rgb.copy()).to(self.device)
        ### Do not transform the depth
        # depth = (
        #     torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        # ) 
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device)
        ) 

        current_task = obs.task_observations["tasks"][self.current_task_idx] # Dict containing task info
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
                    img_goal = obs.task_observations["tasks"][self.current_task_idx][
                        "image"
                    ]
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

        ### Map semantic to (0,16)
        semantic = np.vectorize(self.register_semantic)(semantic)
        semantic = np.vectorize(self.segmentation.categories_mapping.get)(semantic)
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        # print(semantic.shape)

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # print(instances)
            instance_ids = np.unique(instances)
            # print("instance ids:",instance_ids)
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # print(instances)
            # print(instance_id_to_idx)
            # print("="*20)
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # print(instances)
            ### 1st channel: background, 2...: instances
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)
            # print(obs_preprocessed.shape)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)
        # print(obs_preprocessed.shape)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose

        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0)

        # TODO: camara position
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        
        if (
            task_type in ["languagenav", "imagenav"]
            and self.record_instance_ids
            and (
                self.sub_task_timesteps[0][self.current_task_idx] == 0
                or self.force_match_against_memory
            )
        ):
            if self.force_match_against_memory:
                print("Force a match against the memory")
            self.force_match_against_memory = False
            (all_rgb_keypoints, all_matches, all_confidences, instance_ids) = self._match_against_memory(
                task_type, current_task
            )

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
        object_goal_category: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
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
        # update_global = torch.tensor(
        #     [
        #         self.timesteps_before_goal_update[e] == 0
        #         for e in range(self.num_environments)
        #     ]
        # )

        (
            self.instance_map.instance_map_2d,
            seq_local_pose,
        ) = self.instance_map_module(
            obs.unsqueeze(1),
            pose.unsqueeze(1),
            dones.unsqueeze(1),
            point_cloud,
            self.instance_map.instance_map_2d,
            top_down_instance_map
        )
        for e in range(self.num_environments):
            self.sub_task_timesteps[e][self.current_task_idx] += 1

    
    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [
            [0] * self.max_num_sub_task_episodes
        ] * self.num_environments
        # self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.instance_map.init_instance_map()
        # self.episode_panorama_start_steps = self.panorama_start_steps
        # self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        # self.reject_visited_targets = False
        # self.blacklist_target = False
        self.current_task_idx = 0
        # self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False

        # if self.imagenav_visualizer is not None:
        #     self.imagenav_visualizer.reset()

        # self.goal_image = None
        # self.goal_mask = None
        # self.goal_image_keypoints = None

        # self.found_goal[:] = False
        # self.goal_map[:] *= 0
        # self.prev_task_type = None
        # self.planner.reset()
        # self._module.reset()
    
    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
    
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

    def _pos2index(self, coord: Union[str, List[str]], value: Union[Scalar, np.ndarray]):
        '''
        discretize coordinate in unit of meter to integer coordinate at a given granularity (map_resolution)
        the possible range of value of each axis is known forehead via inspect_environ API provided by explorer
        '''
        coord_str2min = {s:v for (s, (v, _)) in zip(['x', 'y', 'z'], self.scene_bbox)}
        if type(value) == np.ndarray:
            coord_min = np.array([coord_str2min[axis] for axis in (coord if type(coord) == list else list(coord))])
            return (100*(value - coord_min) / self.map_resol).astype(np.int32)
        else:
            return int(100*(value - coord_str2min[coord]) / self.map_resol)
    
    def _splat_pixel_features(
            self,
            pixel_features: np.ndarray,
            coords: np.ndarray,
            map_size: List[int]
        ) -> np.ndarray:
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
        splat_feature(
            pixel_features.reshape(1, -1, F), normed_coords.reshape(1, -1, 3), voxel_map
        )
        return voxel_map.squeeze(axis=0)

    
    def step(self, observes: Dict) -> str:
        depth_image = np.clip(observes['depth'], 0, 10) / 10 * 255
        rgb_image = observes['rgb'][...,:3][...,::-1]

        depth_image = np.repeat(depth_image[...,None].astype(np.uint8), 3, axis=-1)

        # agent_view = np.concatenate([rgb_image, depth_image], axis=1)

        obs = Observations(
        gps=[7,15], ### TODO
        compass=[60], ### TODO
        rgb=rgb_image, 
        depth=observes['depth'],
        task_observations={
            "tasks": self.task_list
        },
        camera_pose=None,
        third_person_image=None)

        obs = self.segmentation.predict(obs)
        agent_view = np.concatenate([obs.task_observations["semantic_frame"], depth_image], axis=1)

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
            depth_camera, depth_camera_state.position, depth_camera_state.rotation
        )
        rotvec, rot_in_deg = quat_to_rotvec(depth_camera_state.rotation)
        print('rotvec: ', rotvec)
        print('angle: ', rot_in_deg)

        point_cloud = camera_trans.depth2point_cloud(observes['depth'], heading_axis='-z')
        point_cloud_world_frame = camera_trans.camera2world(point_cloud)

        agent_pos_index = np.array(
            [self._pos2index(axis, v) for v, axis in zip(depth_camera_state.position[self.querier['top_down']], ['x', 'z'])]
        )
        # print("Agent pose:", agent_pos_index)
        agent_pose = np.array([*agent_pos_index, rot_in_deg])
        print("Agent pose with angle",agent_pose)

        batch_size, obs_channels, h, w = obs_preprocessed.size()
        num_instance_channels = obs_channels - 4 - self.num_sem_categories
        # print("obs:",obs_preprocessed.shape)
        # print(num_instance_channels)
        instance_global_map = np.zeros(
            (self.scene_size[0], self.scene_size[2], num_instance_channels)
        )
        # initial_voxel_map = np.zeros((*self.scene_size, self.num_sem_categories))
        instance_channels = obs_preprocessed[
                :,
                4
                + self.num_sem_categories : 4
                + self.num_sem_categories
                + num_instance_channels,
                :,
                :,
            ][0].permute(1,2,0).cpu().numpy()
        cur_frame_voxel_map = self._splat_pixel_features(instance_channels, point_cloud_world_frame, self.scene_size)
        voxel_topdown_projection = np.mean(cur_frame_voxel_map, axis=1)

        self.update_instance_memory(
            obs_preprocessed,
            torch.tensor(agent_pose),
            torch.tensor(point_cloud),
            torch.tensor(voxel_topdown_projection)
        )

        cv2.imshow('view', agent_view)
        action = hook_keystroke()
        return action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="memory_config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()
    baseline_config = OmegaConf.load(args.baseline_config_path)
    # record_instance_id = getattr(baseline_config.AGENT.SEMANTIC_MAP, "record_instance_id", False)

    env_path = '/home/planner/dataset/MP3D/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'
    scene_info = inspect_environ(env_path)
    actor = MyActor(baseline_config, scene_info.nav_bbox, 5, (1, 1.5), 1)
    actor.reset()
    actors = [actor]
    envs = [Env(env_path)]
    run_simulation(actors, envs)
