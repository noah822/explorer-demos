"""
Env 2D
@author: huiming zhou
"""
import numpy as np

class Env:
    def __init__(self,map:np.array):
        self.map=np.array(map)
        num_rows, num_cols = self.map.shape
        self.x_range = num_rows  # size of background
        self.y_range = num_cols
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        obs = set()
        try:
            obs = set(map(tuple, np.argwhere(self.map == 0)))
        except:
            pass
        return obs
