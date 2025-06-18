"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Terrain examples
-------------------------
Demonstrates the use terrain meshes.
Press 'R' to reset the  simulation
"""

import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt

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


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()
print("======================")
print("args:",args)
print("======================")
# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# load ball asset
asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  "resources/assets")
asset_file = "urdf/cube.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

rigid_shape_properties = gym.get_asset_rigid_shape_properties(asset)
print("len(rigid_shape_properties):",len(rigid_shape_properties))
rigid_shape_properties[0].friction = 0.01
# rigid_shape_properties[0].rolling_friction=10.0
# rigid_shape_properties[0].torsion_friction=10.0
print("friction:",rigid_shape_properties[0].friction)
print("rolling_friction:",rigid_shape_properties[0].rolling_friction)
print("torsion_friction:",rigid_shape_properties[0].torsion_friction)
print("restitution:",rigid_shape_properties[0].restitution)

# gym.set_asset_rigid_shape_properties(asset, )

# set up the env grid
num_envs = 10
num_per_row = 10
env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(1, 1, 0)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 0.5
pose.p.x = 0.
# print("pose.p:",pose.p) ### 3 0 1
# print("pose.r:",pose.r) ##  0 0 0 1
envs = []
# set random seed
np.random.seed(1)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 5)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])
    if i == 0:
        color = gymapi.Vec3(1,0,0)

    actor_handle = gym.create_actor(env, asset, pose, 'ball', i, 0)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)





# root_state = gym.acquire_actor_root_state_tensor(sim)
# print("shape:",root_state.shape)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# create all available terrain types
num_terains = 1
terrain_width = 8. ### 10 m
terrain_length = 8. ### 10 m
horizontal_scale = 0.1  # [m]
vertical_scale = 0.005  # [m]
num_rows = int(terrain_width/horizontal_scale)
print("num_rows",num_rows)
num_cols = int(terrain_length/horizontal_scale)
print("num_cols",num_cols)
heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)



# plane_params = gymapi.PlaneParams()
# plane_params.distance = 1 
# plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
# plane_params.static_friction = 1.0
# plane_params.dynamic_friction = 1.0
# plane_params.restitution = 0
# gym.add_ground(sim, plane_params)



def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
# heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=0.0, max_height=0.5, step=0.02, downsampled_scale=1.0).height_field_raw
print("heightfield.shape:",heightfield.shape)
terrain  = new_sub_terrain()
# heightfield[0:num_rows, :] = gap_terrain(new_sub_terrain(), gap_size=0.5, platform_size=5.0).height_field_raw
# heightfield[0:num_rows, :] = pit_terrain(new_sub_terrain(), depth=1.0, platform_size=4.0).height_field_raw
# heightfield[0:num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.1).height_field_raw
# heightfield[0:num_rows, :] = pyramid_sloped_terrain(terrain, slope=0, platform_size=0).height_field_raw
pyramid_sloped_terrain(terrain, slope=0.0, platform_size=3.)
heightfield[0:num_rows, :] = random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2).height_field_raw
# heightfield[0:num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.2, min_size=2., max_size=5., num_rects=20).height_field_raw
# heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=5., amplitude=0.25).height_field_raw
# heightfield[0:num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=1, step_height=-0.5).height_field_raw
# heightfield[0:num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.3, step_height=-0.2, platform_size=1.0).height_field_raw
# heightfield[0:num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.0,
                                                            # stone_distance=1., max_height=0.5, platform_size=0.0, depth=-1).height_field_raw

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
print("vertices.shape:",vertices.shape)
print("triangles.shape:",triangles.shape)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
tm_params.dynamic_friction = 1.0
tm_params.static_friction = 1.0
print("tm_params.dynamic_friction:",tm_params.dynamic_friction)
print("tm_params.static_friction:",tm_params.static_friction)
# print("tm_params.restitution:",tm_params.restitution)
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# p1 = gymapi.Vec3(0,0,0)
# p2 = gymapi.Vec3(2,0,0)
# color_line = gymapi.Vec3(1,0,0)
# gymutil.draw_line(p1,p2,color_line, gym, viewer, envs[0])
# color_line = gymapi.Vec3(0,1,0)
# gymutil.draw_line(p1,p2,color_line, gym, viewer, envs[1])


geom = gymutil.AxesGeometry()
pos_trans = gymapi.Transform()
pos_trans.p = gymapi.Vec3(0, 0, 0)
pos_trans.r = gymapi.Quat(0, 0, 0, 1)
gymutil.draw_lines(geom,gym, viewer, envs[0], pos_trans)


# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
