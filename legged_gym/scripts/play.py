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


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)
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
    

    _, _= env.reset() 

    obs, _ = env.get_observations()
    obs_history, base_vel = env.get_extra_info()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    
    logger = Logger(env.dt) ### env dt  is  control dt
    
    robot_index = 0 # which robot is used for logging
    joint_index = 9 # which joint is used for logging


    start_state_log = 0
    stop_state_log = 400 # number of steps before plotting states
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

    for i in range(int(env.max_episode_length)): ### 10 * 20s
        actions = policy(obs.detach(), obs_history.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        gap = torch.abs((env.feet_pos_in_body_frame[0,0,1] - env.feet_pos_in_body_frame[0,1,1]))
        obs_history, base_vel = env.get_extra_info()


        if i > start_state_log and i< stop_state_log:
            logger.log_states(
                {

                    #'base_vel_est':base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'base_height_est':base_height_est[robot_index,:].detach().cpu().numpy(),

                    'gap': gap.cpu().numpy(),

                    'base_rpy': env.base_rpy[robot_index,:].detach().cpu().numpy(),

                    'leg_phase':env.leg_phase[robot_index,:].detach().cpu().numpy(),

                    'base_height':(env.base_pos[robot_index,2]- env.measured_heights[robot_index,93]).detach().cpu().numpy(),
                    # 'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.25 + env.default_dof_pos[0].detach().cpu().numpy(),
                    'dof_pos':env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    
                    # 'base_vel_est': base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'contact_est': foot_contact_est[robot_index,:].detach().cpu().numpy(),
                    'contact_flag': env.contact_flag[robot_index, :].detach().cpu().numpy(),

                    'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.5 + env.default_dof_pos[0].detach().cpu().numpy(),
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
            logger.plot_states()
            # 定义保存路径（可以是绝对路径或相对路径）
            save_path = f"/home/jinrongjun/g1-ppo-symm-main/legged_gym/logs/G1_PPO_EMLP/play_data/{datetime.now().strftime('%b%d_%H-%M-%S')}_state_log.npy"  # 替换为你的目标路
            # 保存 state_log 到指定路径
            np.save(save_path, logger.state_log)

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item() ### num  traj
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        

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
    RENDER = True
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
