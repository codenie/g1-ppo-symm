from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import math
import cv2
from datetime import datetime
from rsl_rl.utils.symm_utils import add_repr_to_gspace, SimpleEMLP, get_symm_tensor


import escnn
import escnn.group
from escnn.group import CyclicGroup
import ast



G = CyclicGroup(2)
gspace = escnn.gspaces.no_base_space(G)
# 添加需要的变换函数
# obs transition functions for actor & critic
add_repr_to_gspace(G, [0, 1, 2], [-1, 1, -1], 'base_ang_vel')
add_repr_to_gspace(G, [0, 1, 2], [1, -1, 1], 'projected_gravity')
add_repr_to_gspace(G, [0, 1, 2], [1, -1, -1], 'commands')
add_repr_to_gspace(G, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                              [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1],
                            'dof_pos_vel_action')
add_repr_to_gspace(G, [0, 1], [-1, -1], 'phase')

# obs transition functions for critic only
add_repr_to_gspace(G, [0, 1, 2], [-1, 1, 1], 'base_lin_vel')
add_repr_to_gspace(G, [176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 
                               154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 
                               132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 
                               110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                               88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 66, 67, 68, 69, 70, 
                               71, 72, 73, 74, 75, 76, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                               54, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 11, 12, 13, 14, 
                               15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.ones(187), 'heights_obs')
add_repr_to_gspace(G, [0], [1], 'scalar')


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)


    # fix_cmd = cycle_shape_commands[args.fix_cmd]
    # print('now,the command is:', fix_cmd)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    
    # env_cfg.external_forces.env_external_force_proportion = 1.0


    env_cfg.commands.resampling_time = 20.0

    # env_cfg.commands.ranges.lin_vel_x = [2.0, 2.0]
    # env_cfg.commands.ranges.lin_vel_x = [1.5, 1.5]
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    # env_cfg.commands.ranges.lin_vel_x = [-1.0, -1.0]
    # env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    
    env_cfg.terrain.play = True
    # env_cfg.terrain.play = False


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    robot_index = 0 # which robot is used for logging
    joint_index = 9 # which joint is used for logging
    _, _= env.reset() 
    if FIX_command:
        env.commands[robot_index, :] = [0.5,0,0]
    obs, _ = env.get_observations()
    obs_history, base_vel = env.get_extra_info()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    
    logger = Logger(env.dt) ### env dt  is  control dt
    



    start_state_log = 0
    stop_state_log = 2000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        args.run_name = "g1"
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for i in range(int(2010)): ### 10 * 20s
        actions = policy(obs.detach(), obs_history.detach())
        if FIX_command:
            env.commands[robot_index, :] = [0.5, 0, A * np.sin(omega * i)]
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        gap = torch.abs((env.feet_pos_in_body_frame[0,0,1] - env.feet_pos_in_body_frame[0,1,1]))
        obs_history, base_vel = env.get_extra_info()

         #get symmetry actions

        representations = ['base_ang_vel', 'projected_gravity', 'commands', 
                                   *('dof_pos_vel_action',)*3, 'phase'] * 5
        obs_history_symmetry = get_symm_tensor(obs_history, G, representations)
        actions_symmetry =  policy(obs.detach(), obs_history_symmetry.detach())

        if i > start_state_log and i< stop_state_log:
            logger.log_states(
                {

                    #'base_vel_est':base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'base_height_est':base_height_est[robot_index,:].detach().cpu().numpy(),

                    'gap': gap.cpu().numpy(),

                    'base_rpy': env.base_rpy[robot_index,:].detach().cpu().numpy(),
                    'base_euler': env.base_pos[robot_index,:].detach().cpu().numpy(),

                    'leg_phase':env.leg_phase[robot_index,:].detach().cpu().numpy(),

                    'base_height':(env.base_pos[robot_index,2]- env.measured_heights[robot_index,93]).detach().cpu().numpy(),
                    # 'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.25 + env.default_dof_pos[0].detach().cpu().numpy(),
                    'dof_pos':env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    
                    # 'base_vel_est': base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'contact_est': foot_contact_est[robot_index,:].detach().cpu().numpy(),
                    'contact_flag': env.contact_flag[robot_index, :].detach().cpu().numpy(),

                    'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.5 + env.default_dof_pos[0].detach().cpu().numpy(),
                    'actions_symmetry':actions_symmetry.detach().cpu().numpy() * 0.5 + env.default_dof_pos[0].detach().cpu().numpy(),
                    # 'actions_dof':env.actions[robot_index,:].detach().cpu().numpy(),

                    'dof_pos':env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, :].detach().cpu().numpy(),
                    'dof_trq': env.torques[robot_index, :].detach().cpu().numpy(),

                    'commands':env.commands[robot_index, :].detach().cpu().numpy(),
                    'base_vel': env.base_lin_vel[robot_index, :3].detach().cpu().numpy(),
                    'base_ang_vel': env.base_ang_vel[robot_index, :3].detach().cpu().numpy(),

                    'feet_pos': env.feet_pos[robot_index, :].detach().cpu().numpy(),
                    'feet_vel': env.feet_vel[robot_index, :].detach().cpu().numpy(),
                    'contact_force': env.contact_forces[robot_index, :].detach().cpu().numpy(),
                    

                }
            )
        elif i==stop_state_log:
            # logger.plot_states()
            # 定义保存路径（可以是绝对路径或相对路径）
            save_path = f"/home/jinrongjun/g1-ppo-symm-main/legged_gym/logs/{ train_cfg.runner.experiment_name}/play_data_n/{fix_cmd[0]}_{fix_cmd[1]}_{fix_cmd[2]}_state_log.npy"  # 替换为你的目标路
            # 保存 state_log 到指定路径
            np.save(save_path, logger.state_log)
            print("Data is saved")

        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item() ### num  traj
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()
        

        #video recording
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
    if RENDER:
        video.release()

if __name__ == '__main__':
    RENDER  =  False
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_command = True
    mi_shape_commands = torch.tensor([
                                      [0.5, 0.0, 0.0],
                                      [0.5, 0.5, 0.0],
                                      [0.0, 0.5, 0.0],
                                      [-0.5, 0.0, 0.0],
                                      [-0.5, 0.5, 0.0],
                                      [-0.5, -0.5, 0.0],
                                      [0.5, -0.5, 0.0],
                                      [0.0, -0.5, 0.0]
                                      ])
    cycle_shape_commands = torch.tensor([[1.0, 0.0, 0.25],
                                         [1.0, 0.0, -0.25],
                                         [1.0, 0.0, 0.5],
                                         [1.0, 0.0, -0.5],
                                         [1.0, 0.0, 0.75],
                                         [1.0, 0.0, -0.75],
                                         [1.0, 0.0, 1.0],
                                         [1.0, 0.0, -1.0],
                                      ])
    A = 1.0
    omega = 0.5
    args = get_args()
    play(args)
