from collections import defaultdict
from typing import Optional, Tuple, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import torch
import torch.nn as nn
from torch import IntTensor, Tensor
from torch.nn import functional as F

from mapping.semantic.constants import MapConstants as MC
from mapping.semantic.instance_tracking_modules import InstanceMemory

class InstanceMapState:
    def __init__(
        self,
        device: torch.device,
        num_environments: int,
        num_sem_categories: int,
        scene_size: List[int],
        instance_memory: Optional[InstanceMemory] = None,
        max_instances: int = 0,
    ):  
        self.device = device
        self.num_environments = num_environments
        self.num_sem_categories = num_sem_categories

        self.scene_size_x = scene_size[0]
        self.scene_size_y = scene_size[2]

        num_channels = self.num_sem_categories

        # if evaluate_instance_tracking:
        #     num_channels += max_instances + 1
        self.instance_map_2d = torch.zeros(
            self.num_environments,
            num_channels,
            self.scene_size_x,
            self.scene_size_y,
            device=self.device,
        )

    def init_instance_map(self):
        for e in range(self.num_environments):
            self.instance_map_2d[e].fill_(0.0)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_instance_map(self, e) -> np.ndarray:
        return np.copy(self.instance_map[e].cpu().numpy())

class InstanceMap(nn.Module):
    def __init__(self,
                 scene_size: List[int],
                 num_sem_categories: int,
                 du_scale: int,
                 record_instance_ids: bool,
                 instance_memory: Optional[InstanceMemory] = None,
                 max_instances: int = 0,
                 dilation_for_instances: int = 5,
                 padding_for_instance_overlap: int = 5):
        
        super().__init__()
        # self.screen_h = frame_height
        # self.screen_w = frame_width
        self.num_sem_categories = num_sem_categories

        # self.global_downscaling = global_downscaling
        self.du_scale = du_scale
        self.scene_size = scene_size

        self.record_instance_ids = record_instance_ids
        self.padding_for_instance_overlap = padding_for_instance_overlap
        self.dilation_for_instances = dilation_for_instances
        self.instance_memory = instance_memory
        self.max_instances = max_instances
    
    
    @torch.no_grad()
    def forward(
        self,
        seq_obs: Tensor,
        seq_pose: Tensor,
        seq_dones: Tensor,
        point_cloud: Tensor,
        init_instance_map: Tensor, # [h, w, num_sem_categories]
        update_instance_map: Tensor
    ):
        batch_size, sequence_length = seq_obs.shape[:2]
        device, dtype = seq_obs.device, seq_obs.dtype

        pose = seq_pose.clone()
        instance_map = init_instance_map # [1, num_sem, scene_size, scene_size]
        update_map = update_instance_map # [331, 222, 3]
        update_map = update_map.unsqueeze(-1).permute(3,2,0,1)
        # print("update_map",update_map.shape)

        # print(init_instance_map.shape)
        # print(update_instance_map.shape)
        ### By default, the batch size and seq length is 1
        obs = seq_obs[0, :, :, :, :] # [1, 4 + num_sem + num_ins, 256, 256]
        semantic_channels = obs[:, 4 : 4 + self.num_sem_categories, :, :]
        num_instance_channels = obs.shape[1] - 4 - self.num_sem_categories
        # print("num_instance", num_instance_channels)
        # print(obs.shape)
        instance_channels = obs[
                :,
                4
                + self.num_sem_categories : 4
                + self.num_sem_categories
                + num_instance_channels,
                :,
                :,
            ]
        # print(point_cloud.unsqueeze(-1).permute(3,0,1,2).shape)
        if num_instance_channels > 0:
            self.instance_memory.process_instances(
                semantic_channels,
                instance_channels,
                point_cloud.unsqueeze(-1).permute(3,0,1,2), # TODO
                pose.unsqueeze(-1).permute(2,0,1),  # store the global pose
                image=obs[:, :3, :, :],
            )
        
        update_instance_map = self._aggregate_instance_map_channels_per_category(update_map, num_instance_channels)
        # print("aggregate_map,",update_instance_map.shape)

        for e in range(batch_size):
            # TODO if seq_update_global[e, t]:
            self._update_global_map_for_env(
                e, update_instance_map, instance_map
            )

        return init_instance_map, seq_pose

    def _aggregate_instance_map_channels_per_category(
        self, curr_map, num_instance_channels
    ):
        """Aggregate map channels for instances (input: one binary channel per instance in [0, 1])
        by category (output: one channel per category containing instance IDs)."""

        top_down_instance_one_hot = curr_map.clone()
        # now we convert the top down instance map to get a map for storing instances per channel
        top_down_instances_per_category = torch.zeros(
            curr_map.shape[0], # num_env
            self.num_sem_categories,
            curr_map.shape[2],
            curr_map.shape[3],
            device=curr_map.device,
            dtype=curr_map.dtype,
        )

        if num_instance_channels > 0:
            # loop over envs
            # TODO Can we vectorize this across envs? (Only needed if we use multiple envs)
            for i in range(top_down_instance_one_hot.shape[0]):
                # create category id to instance id list mapping
                category_id_to_instance_id_list = defaultdict(list)
                # retrieve unprocessed instances
                unprocessed_instances = (
                    self.instance_memory.get_unprocessed_instances_per_env(i)
                )
                # loop over unprocessed instances
                ### instance_id should be > 0
                for instance_id, instance in unprocessed_instances.items():
                    category_id_to_instance_id_list[instance.category_id].append(
                        instance_id
                    )

                # loop over categories
                # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
                for category_id in category_id_to_instance_id_list.keys():
                    if len(category_id_to_instance_id_list[category_id]) == 0:
                        continue
                    # get the instance ids for this category
                    instance_ids = category_id_to_instance_id_list[category_id]
                    # create a tensor by slicing top_down_instance_one_hot using the instance ids
                    instance_one_hot = top_down_instance_one_hot[i, instance_ids]
                    # add a channel with all values equal to 1e-5 as the first channel
                    instance_one_hot = torch.cat(
                        (
                            1e-5 * torch.ones_like(instance_one_hot[:1]),
                            instance_one_hot,
                        ),
                        dim=0,
                    )
                    # get the instance id map using argmax
                    instance_id_map = instance_one_hot.argmax(dim=0)
                    # print(instance_id_map[220,220])
                    # add a zero to start of instance ids
                    instance_id = [0] + instance_ids # [0, 2, 4]
                    # update the ids using the list of instance ids
                    instance_id_map = torch.tensor(
                        instance_id, device=instance_id_map.device
                    )[instance_id_map]
                    # update the per category instance map
                    top_down_instances_per_category[i, category_id] = instance_id_map

        return top_down_instances_per_category
    
    def _update_global_map_for_env(
        self,
        e: int,
        updated_instance_map: Tensor,
        init_instance_map: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """
        global_instance_map = self._update_global_map_instances(
                e, init_instance_map, updated_instance_map
            )
        return global_instance_map

    def _update_global_map_instances(
        self, e: int, init_instance_map: Tensor, updated_instance_map: Tensor,
    ) -> Tensor:
        # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
        for i in range(self.num_sem_categories):
            if (
                torch.sum(
                    updated_instance_map[e, i]
                )
                > 0
            ):
                max_instance_id = (
                    torch.max(
                        init_instance_map[e]
                    )
                    .int()
                    .item()
                )
                # if the local map has any object instances, update the global map with instance ids
                instances = self._update_global_map_instances_for_one_channel(
                    e,
                    init_instance_map[e, i],
                    updated_instance_map[e, i],
                    max_instance_id,
                )
                init_instance_map[
                    e, i
                ] = instances

        return init_instance_map

    def _update_global_map_instances_for_one_channel(
        self,
        env_id: int,
        global_instances: Tensor,
        local_map: Tensor,
        max_instance_id: int,
    ) -> Tensor:
        ### TODO: needs to pad for a smaller local map
        # p = self.padding_for_instance_overlap  # default: 1
        # d = self.dilation_for_instances  # default: 0

        # H = global_instances.shape[0]
        # W = global_instances.shape[1]

        # x1, x2 = x_range
        # y1, y2 = y_range

        # # padding added on each side
        # t_p = min(x1, p)
        # b_p = min(H - x2, p)
        # l_p = min(y1, p)
        # r_p = min(W - y2, p)

        # # the indices of the padded local_map in the global map
        # x_start = x1 - t_p
        # x_end = x2 + b_p
        # y_start = y1 - l_p
        # y_end = y2 + r_p

        # local_map = torch.round(local_map)

        # # pad the local map
        # extended_local_map = F.pad(local_map.float(), (l_p, r_p), mode="replicate")
        # extended_local_map = F.pad(
        #     extended_local_map.transpose(1, 0), (t_p, b_p), mode="replicate"
        # ).transpose(1, 0)

        # self.instance_dilation_selem = skimage.morphology.disk(d)
        # # dilate the extended local map
        # if d > 0:
        #     extended_dilated_local_map = torch.round(
        #         torch.tensor(
        #             cv2.dilate(
        #                 extended_local_map.cpu().numpy(),
        #                 self.instance_dilation_selem,
        #                 iterations=1,
        #             ),
        #             device=local_map.device,
        #             dtype=local_map.dtype,
        #         )
        #     )
        # else:
        #     extended_dilated_local_map = torch.clone(extended_local_map)
        # # Get the instances from the global map within the local map's region
        # global_instances_within_local = global_instances[x_start:x_end, y_start:y_end]
        extended_dilated_local_map = extended_local_map = local_map.clone()
        global_instances_within_local = global_instances

        instance_mapping = self._get_local_to_global_instance_mapping( ### For a particular channel
            env_id,
            extended_dilated_local_map,
            global_instances_within_local,
            max_instance_id,
            torch.unique(extended_local_map),
        )

        # Update the global map with the associated instances from the local map
        global_instances_in_local = np.vectorize(instance_mapping.get)(
            local_map.cpu().numpy()
        )
        global_instances = torch.maximum(
            global_instances,
            torch.tensor(
                global_instances_in_local,
                dtype=torch.int64,
                device=global_instances.device,
            ),
        )
        return global_instances       

    def _get_local_to_global_instance_mapping(
        self,
        env_id: int,
        extended_local_labels: Tensor,
        global_instances_within_local: Tensor,
        max_instance_id: int,
        local_instance_ids: Tensor,
    ) -> dict:
        """
        Creates a mapping of local instance IDs to global instance IDs.

        Args:
            extended_local_labels: Labels of instances in the extended local map.
            global_instances_within_local: Instances from the global map within the local map's region.

        Returns:
            A mapping of local instance IDs to global instance IDs.
        """
        instance_mapping = {}

        # Associate instances in the local map with corresponding instances in the global map
        for local_instance_id in local_instance_ids:
            if local_instance_id == 0:
                # ignore 0 as it does not correspond to an instance
                continue
            # pixels corresponding to
            local_instance_pixels = extended_local_labels == local_instance_id

            # Check for overlapping instances in the global map
            overlapping_instances = global_instances_within_local[local_instance_pixels]
            unique_overlapping_instances = torch.unique(overlapping_instances)

            unique_overlapping_instances = unique_overlapping_instances[
                unique_overlapping_instances != 0
            ]
            if len(unique_overlapping_instances) >= 1:
                # If there is a corresponding instance in the global map, pick the first one and associate it
                global_instance_id = int(unique_overlapping_instances[0].item())
                instance_mapping[local_instance_id.item()] = global_instance_id
            else:
                # If there are no corresponding instances, create a new instance
                global_instance_id = max_instance_id + 1
                instance_mapping[local_instance_id.item()] = global_instance_id
                max_instance_id += 1
            # update the id in instance memory
            self.instance_memory.update_instance_id(
                env_id, int(local_instance_id.item()), global_instance_id
            )
        instance_mapping[0.0] = 0
        return instance_mapping
