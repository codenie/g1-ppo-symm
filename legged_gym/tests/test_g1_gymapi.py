import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *
from math import sqrt


device = 'cuda:0'
# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()
print("======================")
print("args:",args)
print("======================")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.substeps = 1
sim_params.use_gpu_pipeline = True

sim_params.physx.num_threads = 10
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.contact_offset = 0.001
sim_params.physx.friction_offset_threshold = 0.001  ## zy add
sim_params.physx.rest_offset = 0.0   # [m]
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
sim_params.physx.default_buffer_size_multiplier = 5
sim_params.physx.contact_collection = gymapi.ContactCollection(2) # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
sim_params.physx.use_gpu = args.use_gpu
sim_params.physx.num_subscenes = args.subscenes


sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# load ball asset
asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  "resources/robots/g1_description")
asset_file = "g1_27dof-zy.urdf"


asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = 3
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = True
asset_options.fix_base_link = False
asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 1000
asset_options.max_linear_velocity = 1000
asset_options.armature = 0
asset_options.thickness = 0.01
asset_options.disable_gravity = False
asset_options.flip_visual_attachments=False  #### NOTE Useful


robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
num_dofs = gym.get_asset_dof_count(robot_asset)
print("num_dofs:",num_dofs)
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
print("num_bodies:",num_bodies)
dof_props_asset = gym.get_asset_dof_properties(robot_asset)
rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)


# save body names from the asset
body_names = gym.get_asset_rigid_body_names(robot_asset)
print("body_names:",body_names)
dof_names = gym.get_asset_dof_names(robot_asset)
print("dof_names:",dof_names)
num_bodies = len(body_names)
num_dofs = len(dof_names)
print("num_dofs:",num_dofs)
print("num_bodies:",num_bodies)



foot_name = "ankle_roll"
penalize_contacts_on = ["hip", "knee", "shoulder_yaw", "elbow", 'wrist']
terminate_after_contacts_on = ["waist", "shoulder_pitch"]
feet_names = [s for s in body_names if foot_name in s]
penalized_contact_names = []
for name in penalize_contacts_on:
    penalized_contact_names.extend([s for s in body_names if name in s])
termination_contact_names = []
for name in terminate_after_contacts_on:
    termination_contact_names.extend([s for s in body_names if name in s])


print("feet_names:",feet_names)
print("penalized_contact_names:",penalized_contact_names)
print("termination_contact_names:",termination_contact_names)

pos = [0.0, 0.0, 1.8] # x,y,z [m]
rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]


###  
# default_joint_angles = { 
#             'left_hip_pitch_joint': -0.2, 
#             'left_hip_roll_joint': -0.0, 
#             'left_hip_yaw_joint': 0.0, 
#             'left_knee_joint': 0.42, 
#             'left_ankle_pitch_joint': -0.23,
#             'left_ankle_roll_joint': 0.0, 
#             'right_hip_pitch_joint': -0.2, 
#             'right_hip_roll_joint': 0.0, 
#             'right_hip_yaw_joint': 0.0, 
#             'right_knee_joint': 0.42, 
#             'right_ankle_pitch_joint': -0.23, 
#             'right_ankle_roll_joint': 0.0, 
#             'waist_yaw_joint': 0.0,
#             'left_shoulder_pitch_joint': 0.0,
#             'left_shoulder_roll_joint': 0.0,
#             'left_shoulder_yaw_joint': 0.0,
#             'left_elbow_joint': 0.87,
#             'left_wrist_roll_joint': 0.0,
#             'left_wrist_pitch_joint': 0.0,
#             'left_wrist_yaw_joint': 0.0,
#             'right_shoulder_pitch_joint': 0.0,
#             'right_shoulder_roll_joint': 0.0,
#             'right_shoulder_yaw_joint': 0.0,
#             'right_elbow_joint': 0.87,
#             'right_wrist_roll_joint': 0.0,
#             'right_wrist_pitch_joint': 0.0,
#             'right_wrist_yaw_joint': 0.0,
#         }  

default_joint_angles = {  ### height = 0.7429
    'left_hip_pitch_joint': -0.2, 
    'left_hip_roll_joint': -0.0, 
    'left_hip_yaw_joint': 0.0, 
    'left_knee_joint': 0.42, 
    'left_ankle_pitch_joint': -0.23,
    'left_ankle_roll_joint': 0.0, 
    'right_hip_pitch_joint': -0.2, 
    'right_hip_roll_joint': 0.0, 
    'right_hip_yaw_joint': 0.0, 
    'right_knee_joint': 0.42, 
    'right_ankle_pitch_joint': -0.23, 
    'right_ankle_roll_joint': 0.0, 
    'waist_yaw_joint': 0.0,
    # 'left_shoulder_pitch_joint': 0.25,
    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.2,
    'left_shoulder_yaw_joint': 0.15,
    # 'left_elbow_joint': 0.85,
    'left_elbow_joint': 1.2,
    'left_wrist_roll_joint': 0.0,
    'left_wrist_pitch_joint': 0.0,
    'left_wrist_yaw_joint': 0.0,
    # 'right_shoulder_pitch_joint': 0.25,
    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint': -0.2,
    'right_shoulder_yaw_joint': -0.15,
    # 'right_elbow_joint': 0.85,
    'right_elbow_joint': 1.2,
    'right_wrist_roll_joint': 0.0,
    'right_wrist_pitch_joint': 0.0,
    'right_wrist_yaw_joint': 0.0,
}


default_dof_pos = torch.zeros(num_dofs, dtype=torch.float, device=device, requires_grad=False)
for i in range(num_dofs):
    name = dof_names[i]
    angle = default_joint_angles[name]
    default_dof_pos[i] = angle
    
default_dof_pos = default_dof_pos.unsqueeze(0)
print("default_dof_pos:",default_dof_pos)



base_init_state_list = pos + rot + lin_vel + ang_vel
base_init_state = to_torch(base_init_state_list, device=device, requires_grad=False)
start_pose = gymapi.Transform()
print("base_init_state[:3]:",base_init_state[:3])
start_pose.p = gymapi.Vec3(*base_init_state[:3])
start_pose.r = gymapi.Quat(0, 0, 0, 1)
env_lower = gymapi.Vec3(0., 0., 0.)
env_upper = gymapi.Vec3(0., 0., 0.)

# create env instance
env_handle = gym.create_env(sim, env_lower, env_upper, 1)
actor_handle = gym.create_actor(env_handle, robot_asset, start_pose, 'g1', 0, 1, 0)



feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=device, requires_grad=False)
for i in range(len(feet_names)):
    feet_indices[i] = gym.find_actor_rigid_body_handle(env_handle, actor_handle, feet_names[i])
penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=device, requires_grad=False)
for i in range(len(penalized_contact_names)):
    penalised_contact_indices[i] = gym.find_actor_rigid_body_handle(env_handle, actor_handle, penalized_contact_names[i])

termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=device, requires_grad=False)
for i in range(len(termination_contact_names)):
    termination_contact_indices[i] = gym.find_actor_rigid_body_handle(env_handle, actor_handle, termination_contact_names[i])


print("feet_indices:",feet_indices)
print("penalised_contact_indices:",penalised_contact_indices)
print("termination_contact_indices:",termination_contact_indices)


success_ = gym.prepare_sim(sim)
print("success_:",success_)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)


    # create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


gym.simulate(sim)
gym.fetch_results(sim, True)




# get gym GPU state tensors buffer
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
net_contact_force = gym.acquire_net_contact_force_tensor(sim)
rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)

# create some wrapper tensors for different slices
root_state = gymtorch.wrap_tensor(root_state_tensor)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)
contact_forces = gymtorch.wrap_tensor(net_contact_force).view(1, -1, 3) # shape: num_envs, num_bodies, xyz axis

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_net_contact_force_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

dof_pos = dof_state.view(1, num_dofs, 2)[..., 0]
dof_vel = dof_state.view(1, num_dofs, 2)[..., 1]
base_pos = root_state[:, 0:3]
base_quat = root_state[:, 3:7]
## feet_pos,  feet_vel,    pos 足端 平地上 有 正的 0.02cm， 偏置 
feet_pos = rigid_body_state.view(1, num_bodies, 13)[:,feet_indices,0:3]
feet_vel = rigid_body_state.view(1, num_bodies, 13)[:,feet_indices,7:10]

gym.simulate(sim)
gym.fetch_results(sim, True)

root_state_init = gym.acquire_actor_root_state_tensor(sim)
dof_state_init = gym.acquire_dof_state_tensor(sim)

print("before_init----------------")
print("base_pos:",base_pos)
print("base_quat:",base_quat)
print("dof_pos:",dof_pos)
print("dof_vel:",dof_vel)
print("feet_pos:",feet_pos)
print("feet_vel:",feet_vel)

# print("==================================")

env_ids = torch.zeros(1, device=device, dtype=torch.int32)
dof_pos[0] = default_dof_pos
# dof_pos[0] = default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), num_dofs), device=device)
dof_vel[0] = 0.
env_ids_int32 = env_ids.to(dtype=torch.int32)
gym.set_dof_state_tensor_indexed(sim,
                            gymtorch.unwrap_tensor(dof_state),
                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

gym.simulate(sim)
gym.fetch_results(sim, True)

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_net_contact_force_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)



## feet_pos,  feet_vel,    pos 足端 平地上 有 正的 0.02cm， 偏置 
feet_pos = rigid_body_state.view(1, num_bodies, 13)[:,feet_indices,0:3]
feet_vel = rigid_body_state.view(1, num_bodies, 13)[:,feet_indices,7:10]

upper_body_pos = rigid_body_state.view(1, num_bodies, 13)[:,13,0:3]

print("after_init--------------")
print("base_pos:",base_pos)
print("base_quat:",base_quat)
print("dof_pos:",dof_pos)
print("dof_vel:",dof_vel)
print("feet_pos:",feet_pos)
print("feet_vel:",feet_vel)

print("upper_body_pos:",upper_body_pos)


body_props = gym.get_actor_rigid_body_properties(env_handle, actor_handle)
print("body_props[13].com:",body_props[13].com)
print("body_props[13].mass:",body_props[13].mass)





geom = gymutil.AxesGeometry()
pos_trans = gymapi.Transform()
pos_trans.p = gymapi.Vec3(0, 0, 2)
pos_trans.r = gymapi.Quat(0, 0, 0, 1)
gymutil.draw_lines(geom,gym, viewer, env_handle, pos_trans)


# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")


while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    # for evt in gym.query_viewer_action_events(viewer):
    #     if evt.action == "reset" and evt.value > 0:
            

    # gym.set_actor_root_state_tensor(sim, root_state_init)
    # gym.set_dof_state_tensor(sim, dof_state_init)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    
    # print("base_pos:",base_pos)
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

