import cv2
import numpy as np


from dataclasses import dataclass
from scipy.spatial.transform.rotation import Rotation
from typing import Dict, Tuple, List, Union


import coloring

from explorer.transforms.camera import CameraTransformer
from explorer.transforms.voxel import splat_feature
from explorer.visual import TopDownMap
from explorer_prelude import *

from model import SemanticMoudle


from explorer.visual.constants import TOP_DOWN_MAP_COLOR_MAP 
# reset color mapping for TopDownMap
TOP_DOWN_MAP_COLOR_MAP['unexplorered'] = [255, 255, 255]

WINDOW_NAME = 'view'
TOPDOWN_MAP_NAME = 'bird-eye-view'
CENTIMETER = int
METER = float
DEGREE = float; RADIUS = float

# type alias
Scalar = Union[int, float]



def feature_blur(feature: np.ndarray, keep_shape: bool=True):
    pass

def standardize(v, min_val, max_val):
    '''
    standardize value to range [-1, 1] through linear mapping
    '''
    center = (min_val + max_val) / 2
    return (v - center) / ((max_val - min_val) / 2)

def quat_to_rotvec(quat) -> Tuple[np.ndarray, DEGREE]:
    # hacky approach to keep the rotation axis as -y, i.e [0, -1, 0]
    rotvec = Rotation.from_quat(quat).as_rotvec(degrees=True)
    rot_angle = np.linalg.norm(rotvec)
    rotvec = rotvec / rot_angle
    if rotvec[1] > 0:
        rotvec *= -1; rot_angle *= -1
    return rotvec, rot_angle

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


@dataclass
class Memory:
    voxel_map: np.ndarray
    # scene_pixel_features: np.ndarray
    def update_voxel_map_globally(self, new_voxel_map):
        Memory._voxel_update_policy(self.voxel_map, new_voxel_map)

    def update_voxel_map_locally(self,
                                 new_voxel_map: np.ndarray,
                                 local_map_origin: np.ndarray[np.int32],
                                 local_map_span: np.ndarray[np.int32]):
        '''
        this function only operates on a slice of entire global voxel map for the sake of 
        better memory/computation efficiency
        '''
        local_map_slice = tuple(slice(i, i+s, None) for (i, s) in zip(local_map_origin, local_map_span))
        map_ref = self.voxel_map[local_map_slice]
        Memory._voxel_update_policy(map_ref, new_voxel_map)


    @staticmethod 
    def _voxel_update_policy(old_map, new_map):
        '''
        upating strategy:
        1. take maximum over object channel
        2. updating 0'th channel, which denotes whether there exists object of interest according to 
           the confidence of other channel
           policy: if there is one category with confidence over 50%, we set 0'th channel to 0

        this function operates in place of old_map
        '''
        np.maximum(old_map[..., 1:], new_map[..., 1:], old_map[..., 1:])
        mask = np.max(old_map[..., 1:], axis=-1) > 0.5
        old_map[..., 0][mask] = 0

    
    

             

# image_size = (512, 512)
image_size = (256, 256)
sensor_pos = 0.25 * SensorPos.LEFT + 1.5 * SensorPos.UP
rgb_camera = RGBCamera(name='rgb', resolution=image_size, position=sensor_pos)
depth_camera = DepthCamera(name='depth', resolution=image_size, position=sensor_pos)

@register_moves('turn_left', 'turn_right', 'move_forward')
@register_sensors(rgb_camera, depth_camera)
class DeticActor(Actor):


    def __init__(self,
                 scene_bbox: List[List[METER]],
                 map_resolution: CENTIMETER,
                 local_map_size: List[METER]):
        '''
        Convention:
            +x points RIGHT, +y points UP, -z points FORWARD
        Args:
            scene_size: size of scene in unit of meter, convention (x, y, z)
            map_resolution: in unit of centimeter
            local_map_size: convention (width, depth) in unit of meter
            agent is placed at (width/2, 0)
        '''
        super().__init__()
        self.semantic_module = SemanticMoudle(vocab='coco')
        self.scene_bbox = scene_bbox

        # scene_size in unit of map resolution 
        self.scene_size = [int(100*(max_val - min_val)/map_resolution) for (min_val, max_val) in scene_bbox]

        # currently, only track semantic info with each pixel
        # 2D map squeezes out height dimension, which is y axis
        self.global_map = np.zeros(
            (self.scene_size[0], self.scene_size[2], self.semantic_module.num_sem_categories+1)
        )

        initial_voxel_map = np.zeros((*self.scene_size, self.semantic_module.num_sem_categories+1))
        self.memory_module = Memory(initial_voxel_map)

        # a mutable reference pointing to the global map
        self.local_map = None
        self.local_map_size = local_map_size
        self.map_resol = map_resolution


        # used for maintain visualization of topdown map

        # heading convention: -z 
        self._spirit_init_rot_correction: DEGREE = -90
        self._prev_pos_index, self._prev_rot = None, None
        self.topdown_map_vis = TopDownMap(
            self.global_map.shape[:2],
            track_trajactory=False
        )

        # query each pixel feature
        self.querier = {
            'semantic': slice(0, self.semantic_module.num_sem_categories+1),
            'top_down': slice(0, 3, 2)
        }

        # coloring for each class for visualizing top_down_map
        self.vis_color_array = np.array([np.array([255, 255, 255], dtype=np.uint8)] + coloring.COCO_COLORING)

    
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

    def step(self, observes: Dict) -> str:
        depth_image = np.clip(observes['depth'], 0, 10) / 10 * 255
        # print('observe keys: ', observes.keys())
        # print('collided', observes['collided'])
        depth_image = np.repeat(depth_image[...,None].astype(np.uint8), 3, axis=-1)
        rgb_image = observes['rgb'][...,:3][...,::-1]

        seg_res = self.semantic_module.predict(rgb_image)

        agent_view = np.concatenate([depth_image, seg_res['annotated_frame']], axis=1)


        # TODO: consider displaying this in multiple thread for better perf
        # agent_pos, agent_rot = self.get_agent_state()
        # agent_view = np.concatenate([rgb_image], axis=1)
        # agent_view = np.concatenate([rgb_image, depth_image, seg_res['annotated_frame']], axis=1)
        # agent_pos, agent_rot = self.get_agent_state()

        depth_camera_state = self.get_sensors_state()['depth']
        camera_trans = CameraTransformer.from_instance(
            depth_camera, depth_camera_state.position, depth_camera_state.rotation
        )
        rotvec, rot_in_deg = quat_to_rotvec(depth_camera_state.rotation)
        print('rotvec: ', rotvec)
        print('angle: ', rot_in_deg)

        # point cloud in camera's frame in unit of meter
        # TODO: extrace a squared area centered around agent
        # transfrom point cloud in agent's heading frame to squared frame
        print('depth', np.max(observes['depth']))
        print(depth_camera_state.position)
        print('y range', self.scene_bbox[1])
        print(self.memory_module.voxel_map.shape)

        point_cloud = camera_trans.depth2point_cloud(observes['depth'], heading_axis='-z')
        point_cloud_world_frame = camera_trans.camera2world(point_cloud)
        point_cloud_xz_index = self._pos2index(
            ['x', 'z'], point_cloud_world_frame[:,:,self.querier['top_down']]
        )
        point_cloud_x, point_cloud_z = point_cloud_xz_index[:,:,0], point_cloud_xz_index[:,:,1]

        agent_pos_in_meter = depth_camera_state.position
        (local_map_origin, local_map_span) = self._compute_local_voxel_map_index(agent_pos_in_meter)
        point_cloud_rotated_local_frame = point_cloud_world_frame - local_map_origin

        print(local_map_origin, local_map_span)
        agent_pos_index = np.array(
            [self._pos2index(axis, v) for v, axis in zip(depth_camera_state.position[self.querier['top_down']], ['x', 'z'])]
        )



        # TODO: better impl splat & update function for better memory efficiency
        # self._splat_pixel_features_locally(local_map_ref, seg_res['pixel_features'])
        origin_pixel_features = seg_res['pixel_features']
        # print('un-splatted pixel feature', np.unique(np.argmax(origin_pixel_features, axis=-1)))


        print('prior to filtering', point_cloud_rotated_local_frame.shape)
        filter_mask = np.all(np.logical_and(
           point_cloud_rotated_local_frame > 0, point_cloud_rotated_local_frame < local_map_span 
        ), axis=-1)
        print(filter_mask.shape)
        stdized_filtered_point_cloud_local_frame = standardize(
            point_cloud_rotated_local_frame[filter_mask, :],
            np.zeros(3), local_map_span
        )
        print('after filtering', stdized_filtered_point_cloud_local_frame.shape)
        # stdized_filtered_point_cloud_local_frame = standardize(filtered_point_cloud_local_frame, np.zeros(3), local_map_span)

        local_map_size = np.array([int(i*100/self.map_resol) for i in local_map_span])

        cur_frame_local_voxel_map = self._splat_pixel_features(
            seg_res['pixel_features'][filter_mask, :], stdized_filtered_point_cloud_local_frame, 
            local_map_size
        )

        # stdized_point_cloud_world_frame = standardize(point_cloud_world_frame, *map(np.array, zip(*self.scene_bbox)))
        # cur_frame_voxel_map = self._splat_pixel_features(seg_res['pixel_features'], point_cloud_world_frame, self.scene_size)
        # cur_frame_voxel_map = self._splat_pixel_features(seg_res['pixel_features'], stdized_point_cloud_world_frame, self.scene_size)
        # self.memory_module.voxel_map = cur_frame_voxel_map
        # self.memory_module.update_voxel_map(cur_frame_voxel_map)
        self.memory_module.update_voxel_map_locally(
            cur_frame_local_voxel_map,
            self._pos2index(['x', 'y', 'z'], local_map_origin),
            local_map_size
        )

        cv2.imshow(WINDOW_NAME, agent_view)

        # update topdown map visualization
        top_down_pos = depth_camera_state.position[self.querier['top_down']]
        _, rot_in_deg = quat_to_rotvec(depth_camera_state.rotation)

        self._update_topdown_map_vis(top_down_pos, rot_in_deg)
        cv2.imshow(TOPDOWN_MAP_NAME, self.topdown_map_vis.raw_map)

        action = hook_keystroke()
        return action
    



    def _splat_pixel_features(
            self,
            pixel_features: np.ndarray,
            stdized_coords: np.ndarray,
            map_size: List[int]
        ) -> np.ndarray:
        '''
        splat features associated with the image the agent currently sees to a global 3D voxel
        say the image the agent extracts is of size (h, w), global map is of size (X, Y, Z)
        Args:
            pixel_features: (h, w, F) each pixel associates with F feature
            stdized_coords: (h, w, 3) coordinate associated with each pixel of the voxel map it will be splatted to, which should be standarized into
                            range [-1, 1]
            map_size: (X, Y, Z) size of voxel map feature will be splatted to 
        
        Return:
            global feature grid (X, Y, Z, F)
        '''
        # normalize pixel coordinate in unit of meter to range [-1, 1] to accommondate the prototype of explorer's splat_feature func
        # coord_min, coord_max = map(np.array, zip(*self.scene_bbox))    
        # normed_coords = standardize(coords, coord_min, coord_max)

        F = pixel_features.shape[-1]
        voxel_map = np.zeros((1, *map_size, F))

        # operation is taken in place
        splat_feature(
            pixel_features.reshape(1, -1, F), stdized_coords.reshape(1, -1, 3), voxel_map
        )
        return voxel_map.squeeze(axis=0)



    def _update_topdown_map_vis(self,
                                agent_pos: List[METER],
                                agent_rot: DEGREE):
        # project 3D voxel map along height(y) axis to form a 2D top down map
        voxel_topdown_projection = np.mean(self.memory_module.voxel_map, axis=1)
        topdown_map_pixel_cate = np.argmax(voxel_topdown_projection, axis=-1)
        pixel_coloring = self.vis_color_array[topdown_map_pixel_cate] 

        agent_pos_index = np.array([self._pos2index(axis, v) for v, axis in zip(agent_pos, ['x', 'z'])])
        agent_rot = self._spirit_init_rot_correction + agent_rot

        # we'd better change this api of subsitituting raw_map afterwards
        self.topdown_map_vis.raw_map = pixel_coloring[:,:,::-1]

        if self._prev_pos_index is None:
           self.topdown_map_vis.init_sprite_at(agent_pos_index, agent_rot) 
           self._prev_pos_index, self._prev_rot = agent_pos_index, agent_rot
        else:
           dxy, dtheta = agent_pos_index - self._prev_pos_index, agent_rot - self._prev_rot
           if np.sum(abs(dxy)) > 1e-2 or abs(dtheta)> 1e-2:
                self.topdown_map_vis.move_sprite(dxy, dtheta)
                self._prev_pos_index, self._prev_rot = agent_pos_index, agent_rot

    
    def _get_local_map_as_ref(self, location: List[float], read_only=False):
        # compute local map
        xx, zz, w, d = self._compute_local_map_index(location)
        local_map_ref = self.global_map[xx:xx+w, zz:zz+d]
        local_map_ref.flags.writable = (not read_only)

        return local_map_ref
    
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

        # lower left corner of local map
        # local_map_origin = self._pos2index(['x', 'y', 'z'], np.array([origin_x, y_min, origin_z]))
        # local_map_span = np.array([int(i*100/self.map_resol) for i in [x_span, y_span, z_span]])
        # return local_map_origin, local_map_span


