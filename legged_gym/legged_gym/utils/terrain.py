import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
       
        self.slope_max = cfg.slope_max
        self.uniform_terrain_height_max = cfg.uniform_terrain_height_max
        self.step_height_max = cfg.step_height_max
        self.wave_terrain_amplitude_max = cfg.wave_terrain_amplitude_max
        self.discrete_height = cfg.discrete_height

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.terrain_proportions = np.array(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        


    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        
        slope = difficulty * self.slope_max
        uniform_terrain_height = difficulty * self.uniform_terrain_height_max
        step_height = difficulty * self.step_height_max
        wave_terrain_amplitude_max = 0.1 + difficulty * self.wave_terrain_amplitude_max
        discrete_height = difficulty * self.discrete_height

        terrain_utils.pyramid_sloped_terrain(terrain, slope=0.0, platform_size=1.0)
        # terrain_utils.pyramid_sloped_terrain(terrain, slope=0.1, platform_size=1.0)
        # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-0.1, platform_size=1.5)
        # terrain_utils.sloped_terrain(terrain, slope=0.52598)
        # terrain_utils.random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.005, downsampled_scale=0.2)
        # terrain_utils.discrete_obstacles_terrain(terrain, max_height=0.02, min_size=1.0, max_size=1.5, num_rects=50)

### -----------------   plane  ------------------------
        # if choice < self.proportions[0]:
        #     terrain_utils.discrete_obstacles_terrain(terrain, max_height=discrete_height, min_size=1.0, max_size=1.5, num_rects=50)
        # elif choice < self.proportions[1]:
        #     if choice < self.proportions[1]-self.terrain_proportions[1]/2:
        #         slope *= -1
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.0)
        # elif choice < self.proportions[2]:
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-uniform_terrain_height, max_height=uniform_terrain_height, step=0.005, downsampled_scale=0.2)
        # # elif choice < self.proportions[3]:
        # #     if choice<self.proportions[3] - self.terrain_proportions[3]/2:
        # #         step_height *= -1
        # #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=step_height, platform_size=1.5)
        # else:
        #     terrain_utils.discrete_obstacles_terrain(terrain, max_height=discrete_height, min_size=1.0, max_size=2.0, num_rects=30)


### -----------------   terrain  ------------------------
        # if choice < self.proportions[0]:
        #     terrain_utils.discrete_obstacles_terrain(terrain, max_height=discrete_height, min_size=1.0, max_size=1.5, num_rects=50)
        # elif choice < self.proportions[1]:
        #     if choice < self.proportions[1]-self.terrain_proportions[1]/2:
        #         slope *= -1
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.0)
        # elif choice < self.proportions[2]:
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-uniform_terrain_height, max_height=uniform_terrain_height, step=0.005, downsampled_scale=0.2)
        # elif choice < self.proportions[3]:
        #     if choice<self.proportions[3] - self.terrain_proportions[3]/2:
        #         step_height *= -1
        #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=step_height, platform_size=1.5)
        # else:
        #     terrain_utils.discrete_obstacles_terrain(terrain, max_height=discrete_height, min_size=1.0, max_size=2.0, num_rects=30)

       
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0
    
    return terrain

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    
    return terrain
