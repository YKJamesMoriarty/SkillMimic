from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class SkillMimicBallPlay(HumanoidWholeBodyWithObject): 
    """
    SkillMimicBallPlay类用于实现基于模仿学习的机器人球类运动技能训练
    继承自HumanoidWholeBodyWithObject类，扩展了模仿学习相关功能
    """
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """
        初始化函数
        
        参数：
        - cfg: 配置字典，包含环境设置
        - sim_params: 仿真参数
        - physics_engine: 物理引擎类型
        - device_type: 设备类型 (CPU/GPU)
        - device_id: 设备ID
        - headless: 是否无头模式运行
        """
        # 初始化状态设置
        state_init = str(cfg["env"]["stateInit"])
        if state_init.lower() == "random":
            self._state_init = -1  # 随机参考状态初始化
            print("Random Reference State Init (RRSI)")
        else:
            self._state_init = int(state_init)  # 确定性参考状态初始化
            print(f"Deterministic Reference State Init from {self._state_init}")

        # 从配置中读取参数
        self.motion_file = cfg['env']['motion_file']  # 运动数据文件路径
        self.play_dataset = cfg['env']['playdataset']  # 是否播放数据集
        self.robot_type = cfg["env"]["asset"]["assetFileName"]  # 机器人模型文件名
        self.reward_weights_default = cfg["env"]["rewardWeights"]  # 默认奖励权重
        self.save_images = cfg['env']['saveImages']  # 是否保存图像
        self.save_images_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 图像保存时间戳
        self.init_vel = cfg['env']['initVel']  # 初始速度
        self.isTest = cfg['args'].test  # 是否测试模式

        self.condition_size = 64  # 条件向量大小

        # 调用父类初始化
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        # 参考HOI(人机交互)观测大小
        self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6 #V1
        
        self._load_motion(self.motion_file)  # 加载运动数据

        # 初始化各种观测张量
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)  # 当前参考观测
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)  # 历史参考观测
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)  # 当前实际观测
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)  # 历史实际观测
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)  # 目标位置
        # 初始化HOI数据标签批次
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)
        # 订阅键盘事件用于改变条件
        self._subscribe_events_for_change_condition()

        self.show_motion_test = False  # 是否显示运动测试
        self.motion_id_test = 0  # 运动ID测试
        self.succ_pos = []  # 成功位置记录
        self.fail_pos = []  # 失败位置记录
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)  # 到达目标标记

        self.show_abnorm = [0] * self.num_envs  # 异常显示计数

        return

    def post_physics_step(self):
        """
        物理仿真步骤后执行的函数
        更新条件、计算观测、调用父类post_physics_step、更新历史观测
        """
        self._update_condition()  # 更新条件
        
        # 计算当前观测，用于模仿奖励
        self._compute_hoi_observations()

        super().post_physics_step()  # 调用父类的物理步骤后处理

        self._update_hist_hoi_obs()  # 更新历史HOI观测

        return

    def _update_hist_hoi_obs(self, env_ids=None):
        """
        更新历史HOI观测
        
        参数：
        - env_ids: 可选，环境ID列表，默认为None表示所有环境
        """
        self._hist_obs = self._curr_obs.clone()  # 将当前观测复制到历史观测
        return

    def get_obs_size(self):
        """
        获取观测大小
        
        返回：
        - 观测向量的总大小
        """
        obs_size = super().get_obs_size()  # 获取父类观测大小
        
        obs_size += self.condition_size  # 添加条件向量大小
        return obs_size

    def get_task_obs_size(self):
        """
        获取任务观测大小
        
        返回：
        - 任务观测向量大小，此处为0
        """
        return 0
    
    def _compute_observations(self, env_ids=None):
        """
        计算观测向量
        在重置和每个物理步骤后调用
        
        参数：
        - env_ids: 可选，环境ID列表，默认为None表示所有环境
        """
        obs = None
        # 计算人形机器人观测
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        # 计算物体观测并拼接
        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        # 如果启用任务观测，计算并拼接任务观测
        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)
        
        # 处理所有环境或特定环境
        if (env_ids is None):
            # 所有环境，添加条件向量
            textemb_batch = self.hoi_data_label_batch
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[:] = obs  # 更新观测缓冲区
            
            # 获取所有环境ID和时间步
            env_ids = torch.arange(self.num_envs)
            ts = self.progress_buf.clone()
            # 获取当前参考观测
            self._curr_ref_obs = self.hoi_data_batch[env_ids, ts].clone()

        else:
            # 特定环境，添加条件向量
            textemb_batch = self.hoi_data_label_batch[env_ids]
            obs = torch.cat((obs, textemb_batch), dim=-1)
            self.obs_buf[env_ids] = obs  # 更新特定环境的观测缓冲区

            # 获取特定环境的时间步和参考观测
            ts = self.progress_buf[env_ids].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()

        return

    def _compute_reset(self):
        """
        计算环境重置条件
        
        调用compute_humanoid_reset函数计算哪些环境需要重置
        """
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf,
            self._contact_forces,
            self._rigid_body_pos, self.max_episode_length,
            self._enable_early_termination, self._termination_heights, 
            self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
            self.isTest, self.cfg["env"]["episodeLength"]
        )
        return
    
    def _compute_reward(self, actions):
        """
        计算奖励
        
        参数：
        - actions: 机器人动作
        
        调用compute_humanoid_reward函数计算奖励值
        """
        self.rew_buf[:] = compute_humanoid_reward(
            self._curr_ref_obs,  # 当前参考观测
            self._curr_obs,       # 当前实际观测
            self._hist_obs,       # 历史观测
            self._contact_forces, # 接触力
            self._tar_contact_forces, # 目标接触力
            len(self._key_body_ids),  # 关键身体部位数量
            self._motion_data.reward_weights  # 奖励权重
        )
        return
    
    def _load_motion(self, motion_file):
        """
        加载运动数据
        
        参数：
        - motion_file: 运动数据文件路径
        """
        self.skill_name = os.path.basename(motion_file)  # 提取技能名称
        self.max_episode_length = 60  # 默认最大 episode 长度
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length = self.cfg["env"]["episodeLength"]  # 从配置中读取

        # 初始化运动数据处理器
        self._motion_data = MotionDataHandler(
            motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset
        )
        
        # 初始化HOI数据批次张量
        self.hoi_data_batch = torch.zeros(
            [self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], 
            device=self.device, dtype=torch.float
        )
        
        return
    
    def _subscribe_events_for_change_condition(self):
        """
        订阅键盘事件以改变条件
        用于在可视化界面中手动切换不同的技能动作
        """
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "011")  # 向左运球
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "012")  # 向右运球
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "013")  # 向前运球
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "001")  # 拾取
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "002")  # 投篮
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "031")  # 上篮
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "032")  # 预留动作
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "033")  # 预留动作
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "034")  # 转身上篮
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "035")  # 预留动作
        
        return
    
    def _reset_envs(self, env_ids):
        """
        重置环境
        
        参数：
        - env_ids: 需要重置的环境ID列表
        """
        if len(env_ids) > 0:
            self.reached_target[env_ids] = 0  # 重置到达目标标记
        
        super()._reset_envs(env_ids)  # 调用父类重置环境

        return

    def _reset_actors(self, env_ids):
        """
        重置演员
        
        参数：
        - env_ids: 需要重置的环境ID列表
        """
        if self._state_init == -1:
            # 随机参考状态初始化
            self._reset_random_ref_state_init(env_ids)
        elif self._state_init >= 2:
            # 确定性参考状态初始化
            self._reset_deterministic_ref_state_init(env_ids)
        else:
            # 不支持的初始化类型
            assert(False), f"Unsupported state initialization from: {self._state_init}"

        super()._reset_actors(env_ids)  # 调用父类重置演员

        return

    def _reset_humanoid(self, env_ids):
        """
        重置人形机器人
        
        参数：
        - env_ids: 需要重置的环境ID列表
        """
        # 设置人形机器人根状态
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]  # 根位置
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]  # 根旋转
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]  # 根线速度
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]  # 根角速度
        
        # 设置自由度位置和速度
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]  # 自由度位置
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]  # 自由度速度
        return

    def _reset_random_ref_state_init(self, env_ids):
        """
        随机参考状态初始化
        
        参数：
        - env_ids: 需要重置的环境ID列表
        """
        num_envs = env_ids.shape[0]  # 环境数量

        # 采样运动ID和时间
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)

        # 获取技能标签并转换为one-hot编码
        skill_label = self._motion_data.motion_class[motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(
            torch.tensor(skill_label, device=self.device), 
            num_classes=self.condition_size
        ).float()

        # 获取初始状态
        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  \
        self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], \
        self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        """
        确定性参考状态初始化
        
        参数：
        - env_ids: 需要重置的环境ID列表
        """
        num_envs = env_ids.shape[0]  # 环境数量

        # 采样运动ID，时间固定为配置值
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        # 获取技能标签并转换为one-hot编码
        skill_label = self._motion_data.motion_class[motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(
            torch.tensor(skill_label, device=self.device), 
            num_classes=self.condition_size
        ).float()

        # 获取初始状态
        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  \
        self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], \
        self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return

    def _compute_hoi_observations(self, env_ids=None):
        """
        计算HOI(人机交互)观测
        
        参数：
        - env_ids: 可选，环境ID列表，默认为None表示所有环境
        """
        # 获取关键身体部位位置
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            # 计算所有环境的HOI观测
            self._curr_obs[:] = build_hoi_observations(
                self._rigid_body_pos[:, 0, :],  # 根位置
                self._rigid_body_rot[:, 0, :],  # 根旋转
                self._rigid_body_vel[:, 0, :],  # 根线速度
                self._rigid_body_ang_vel[:, 0, :],  # 根角速度
                self._dof_pos, self._dof_vel,  # 自由度位置和速度
                key_body_pos,  # 关键身体部位位置
                self._local_root_obs, self._root_height_obs,  # 局部根观测和根高度观测
                self._dof_obs_size, self._target_states,  # 自由度观测大小和目标状态
                self._hist_obs,  # 历史观测
                self.progress_buf  # 进度缓冲区
            )
        else:
            # 计算特定环境的HOI观测
            self._curr_obs[env_ids] = build_hoi_observations(
                self._rigid_body_pos[env_ids][:, 0, :],  # 根位置
                self._rigid_body_rot[env_ids][:, 0, :],  # 根旋转
                self._rigid_body_vel[env_ids][:, 0, :],  # 根线速度
                self._rigid_body_ang_vel[env_ids][:, 0, :],  # 根角速度
                self._dof_pos[env_ids], self._dof_vel[env_ids],  # 自由度位置和速度
                key_body_pos[env_ids],  # 关键身体部位位置
                self._local_root_obs, self._root_height_obs,  # 局部根观测和根高度观测
                self._dof_obs_size, self._target_states[env_ids],  # 自由度观测大小和目标状态
                self._hist_obs[env_ids],  # 历史观测
                self.progress_buf[env_ids]  # 进度缓冲区
            )
        
        return
    
    def _update_condition(self):
        """
        更新条件
        处理键盘事件，切换不同的技能动作
        """
        for evt in self.evts:
            # 检查是否为数字动作且值大于0
            if evt.action.isdigit() and evt.value > 0:
                # 更新条件向量为one-hot编码
                self.hoi_data_label_batch = torch.nn.functional.one_hot(
                    torch.tensor(int(evt.action), device=self.device), 
                    num_classes=self.condition_size
                ).float().repeat(self.hoi_data_label_batch.shape[0], 1)
                print(evt.action)  # 打印动作ID

    def play_dataset_step(self, time):
        """
        播放数据集步骤
        
        参数：
        - time: 当前时间步
        
        返回：
        - 观测缓冲区
        """
        t = time

        for env_id, env_ptr in enumerate(self.envs):
            # 获取当前环境的运动ID和归一化时间
            motid = self._motion_data.envid2motid[env_id].item()
            t = t % self._motion_data.motion_lengths[motid]

            ### 更新物体状态 ###
            self._target_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t, :]
            self._target_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][t, :]
            self._target_states[env_id, 7:10] = torch.zeros_like(self._target_states[env_id, 7:10])
            self._target_states[env_id, 10:13] = torch.zeros_like(self._target_states[env_id, 10:13])

            ### 更新主体状态 ###
            _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][t, :].clone()
            _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][t, :].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot
            self._humanoid_root_states[env_id, 7:10] = torch.zeros_like(self._humanoid_root_states[env_id, 7:10])
            self._humanoid_root_states[env_id, 10:13] = torch.zeros_like(self._humanoid_root_states[env_id, 10:13])
            
            self._dof_pos[env_id] = self._motion_data.hoi_data_dict[motid]['dof_pos'][t, :].clone()
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            # 检查接触和异常
            contact = self._motion_data.hoi_data_dict[motid]['contact'][t, :]
            obj_contact = torch.any(contact > 0.1, dim=-1)
            root_rot_vel = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t, :]
            angle = torch.norm(root_rot_vel)  # 计算角速度大小
            abnormal = torch.any(torch.abs(angle) > 5.)  # 检查是否异常

            if abnormal == True:
                print("frame:", t, "abnormal:", abnormal, "angle", angle)
                self.show_abnorm[env_id] = 10  # 设置异常显示计数

            # 更新物体颜色：接触时红色，否则绿色
            handle = self._target_handles[env_id]
            if obj_contact == True:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            
            # 更新人形机器人颜色：异常时蓝色
            if abnormal == True or self.show_abnorm[env_id] > 0:
                for j in range(self.num_bodies):
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1  # 减少异常显示计数
            else:
                for j in range(self.num_bodies):
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
      
        # 更新仿真状态
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()     

        # 渲染和仿真
        self.render(t=time)
        self.gym.simulate(self.sim)

        # 计算观测
        self._compute_observations()

        return self.obs_buf
    
    def _draw_task_play(self, t):
        """
        绘制任务播放
        
        参数：
        - t: 当前时间步
        """
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # 红色

        self.gym.clear_lines(self.viewer)  # 清除之前的线条

        starts = self._motion_data.hoi_data_dict[0]['hoi_data'][t, :3]  # 起始位置

        # 绘制每个环境的关键身体部位连线
        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self._motion_data.hoi_data_dict[0]['key_body_pos'][t, j*3:j*3+3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

        return

    def render(self, sync_frame_time=False, t=0):
        """
        渲染函数
        
        参数：
        - sync_frame_time: 是否同步帧时间
        - t: 当前时间步
        """
        super().render(sync_frame_time)  # 调用父类渲染

        if self.viewer:
            self._draw_task()  # 绘制任务
            self.play_dataset  # 播放数据集标志
            if self.save_images:
                env_ids = 0  # 保存第一个环境的图像
                frame_id = t if self.play_dataset else self.progress_buf[env_ids]  # 帧ID
                # 构建图像保存路径
                rgb_filename = "skillmimic/data/images/" + self.save_images_timestamp + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("skillmimic/data/images/" + self.save_images_timestamp, exist_ok=True)  # 创建目录
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)  # 保存图像
        return

    def _draw_task(self):
        """
        绘制任务
        当前为空实现，可根据需要添加绘制逻辑
        """
        # # 绘制物体接触
        # obj_contact = torch.any(torch.abs(self._tar_contact_forces[..., 0:2]) > 0.1, dim=-1)
        # for env_id, env_ptr in enumerate(self.envs):
        #     env_ptr = self.envs[env_id]
        #     handle = self._target_handles[env_id]

        #     if obj_contact[env_id] == True:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(1., 0., 0.))
        #     else:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(0., 1., 0.))

        return

    def get_num_amp_obs(self):
        """
        获取AMP(Adaptive Motion Priors)观测数量
        
        返回：
        - 参考HOI观测大小
        """
        return self.ref_hoi_obs_size



#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, hist_obs, progress_buf):
    """
    构建HOI(人机交互)观测向量
    
    参数：
    - root_pos: 根位置
    - root_rot: 根旋转
    - root_vel: 根线速度
    - root_ang_vel: 根角速度
    - dof_pos: 自由度位置
    - dof_vel: 自由度速度
    - key_body_pos: 关键身体部位位置
    - local_root_obs: 局部根观测
    - root_height_obs: 根高度观测
    - dof_obs_size: 自由度观测大小
    - target_states: 目标状态
    - hist_obs: 历史观测
    - progress_buf: 进度缓冲区
    
    返回：
    - 构建好的HOI观测向量
    """
    # 速度处理：第一帧速度设为0
    dof_vel = dof_vel * (progress_buf != 1).unsqueeze(dim=-1)

    # 接触状态初始化
    contact = torch.zeros(key_body_pos.shape[0], 1, device=dof_vel.device)
    
    # 构建观测向量：根位置、旋转(转为指数映射)、自由度位置和速度、目标状态、关键身体部位位置、接触状态
    obs = torch.cat((
        root_pos, 
        torch_utils.quat_to_exp_map(root_rot), 
        dof_pos, 
        dof_vel, 
        target_states[:, :10], 
        key_body_pos.contiguous().view(-1, key_body_pos.shape[1] * key_body_pos.shape[2]), 
        contact
    ), dim=-1)
    return obs

# @torch.jit.script
def compute_humanoid_reward(hoi_ref, hoi_obs, hoi_obs_hist, contact_buf, tar_contact_forces, len_keypos, w):
    """
    计算人形机器人模仿奖励
    
    参数：
    - hoi_ref: 参考HOI(人机交互)观测
    - hoi_obs: 当前HOI观测
    - hoi_obs_hist: 历史HOI观测
    - contact_buf: 接触力缓冲区
    - tar_contact_forces: 目标接触力
    - len_keypos: 关键位置数量
    - w: 奖励权重字典
    
    返回：
    - 计算得到的奖励值
    """
    # 数据预处理

    # 模拟状态解析
    root_pos = hoi_obs[:,:3]  # 根位置
    root_rot = hoi_obs[:,3:3+3]  # 根旋转(指数映射)
    dof_pos = hoi_obs[:,6:6+52*3]  # 自由度位置
    dof_pos_vel = hoi_obs[:,162:162+52*3]  # 自由度速度
    obj_pos = hoi_obs[:,318:318+3]  # 物体位置
    obj_rot = hoi_obs[:,321:321+4]  # 物体旋转
    obj_pos_vel = hoi_obs[:,325:325+3]  # 物体速度
    key_pos = hoi_obs[:,328:328+len_keypos*3]  # 关键身体部位位置
    contact = hoi_obs[:,-1:]  # 接触状态(模拟值)
    key_pos = torch.cat((root_pos, key_pos), dim=-1)  # 合并根位置和关键位置
    body_rot = torch.cat((root_rot, dof_pos), dim=-1)  # 合并根旋转和自由度位置
    
    # 构建交互图(IG): 关键位置与物体位置的相对关系
    ig = key_pos.view(-1, len_keypos+1, 3).transpose(0, 1) - obj_pos[:,:3]
    ig_wrist = ig.transpose(0, 1)[:,0:7+1,:].view(-1, (7+1)*3)  # 手腕部分的交互图
    ig = ig.transpose(0, 1).view(-1, (len_keypos+1)*3)  # 完整交互图

    # 历史自由度速度
    dof_pos_vel_hist = hoi_obs_hist[:,162:162+52*3]

    # 参考状态解析
    ref_root_pos = hoi_ref[:,:3]  # 参考根位置
    ref_root_rot = hoi_ref[:,3:3+3]  # 参考根旋转
    ref_dof_pos = hoi_ref[:,6:6+52*3]  # 参考自由度位置
    ref_dof_pos_vel = hoi_ref[:,162:162+52*3]  # 参考自由度速度
    ref_obj_pos = hoi_ref[:,318:318+3]  # 参考物体位置
    ref_obj_rot = hoi_ref[:,321:321+4]  # 参考物体旋转
    ref_obj_pos_vel = hoi_ref[:,325:325+3]  # 参考物体速度
    ref_key_pos = hoi_ref[:,328:328+len_keypos*3]  # 参考关键位置
    ref_obj_contact = hoi_ref[:,-1:]  # 参考物体接触
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos), dim=-1)  # 合并参考根位置和关键位置
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos), dim=-1)  # 合并参考根旋转和自由度位置
    
    # 构建参考交互图
    ref_ig = ref_key_pos.view(-1, len_keypos+1, 3).transpose(0, 1) - ref_obj_pos[:,:3]
    ref_ig_wrist = ref_ig.transpose(0, 1)[:,0:7+1,:].view(-1, (7+1)*3)  # 参考手腕交互图
    ref_ig = ref_ig.transpose(0, 1).view(-1, (len_keypos+1)*3)  # 完整参考交互图

    # 身体模仿奖励计算
    
    # 身体位置奖励
    ep = torch.mean((ref_key_pos - key_pos)**2, dim=-1)  # 位置误差
    rp = torch.exp(-ep * w['p'])  # 指数衰减奖励

    # 身体旋转奖励
    er = torch.mean((ref_body_rot - body_rot)**2, dim=-1)  # 旋转误差
    rr = torch.exp(-er * w['r'])  # 指数衰减奖励

    # 身体位置速度奖励 (当前设为0)
    epv = torch.zeros_like(ep)
    rpv = torch.exp(-epv * w['pv'])

    # 身体旋转速度奖励
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2, dim=-1)  # 速度误差
    rrv = torch.exp(-erv * w['rv'])  # 指数衰减奖励

    # 身体速度平滑度奖励
    e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2 / (((ref_dof_pos_vel**2) + 1e-12)*1e12), dim=-1)
    r_vel_diff = torch.exp(-e_vel_diff * 0.1)  # 速度变化平滑度奖励

    # 综合身体奖励
    rb = rp * rr * rpv * rrv * r_vel_diff

    # 物体模仿奖励计算
    
    # 物体位置奖励
    eop = torch.mean((ref_obj_pos - obj_pos)**2, dim=-1)  # 物体位置误差
    rop = torch.exp(-eop * w['op'])  # 指数衰减奖励

    # 物体旋转奖励 (当前设为0)
    eor = torch.zeros_like(ep)
    ror = torch.exp(-eor * w['or'])

    # 物体位置速度奖励
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2, dim=-1)  # 物体速度误差
    ropv = torch.exp(-eopv * w['opv'])  # 指数衰减奖励

    # 物体旋转速度奖励 (当前设为0)
    eorv = torch.zeros_like(ep)
    rorv = torch.exp(-eorv * w['orv'])

    # 综合物体奖励
    ro = rop * ror * ropv * rorv

    # 交互图奖励计算
    eig = torch.mean((ref_ig - ig)**2, dim=-1)  # 交互图误差
    rig = torch.exp(-eig * w['ig'])  # 指数衰减奖励

    # 简化接触图奖励计算
    # 由于Isaac Gym尚未提供GPU管道中的详细碰撞检测API，使用力检测来近似接触状态
    # 使用接触图节点而非边进行模仿

    # 身体接触检测
    contact_body_ids = [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 34, 35, 36]  # 关键身体部位ID
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone()  # 提取关键部位接触力
    body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)  # 判断是否接触 (力小于阈值)
    body_contact = 1. - torch.all(body_contact, dim=-1).to(float)  # 0表示无接触，1表示有接触

    # 物体接触检测
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(float)  # 物体接触状态

    # 计算接触图奖励
    ref_body_contact = torch.zeros_like(ref_obj_contact)  # 参考身体接触始终为0
    ecg1 = torch.abs(body_contact - ref_body_contact[:, 0])  # 身体接触误差
    rcg1 = torch.exp(-ecg1 * w['cg1'])  # 身体接触奖励
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:, 0])  # 物体接触误差
    rcg2 = torch.exp(-ecg2 * w['cg2'])  # 物体接触奖励

    # 综合接触图奖励
    rcg = rcg1 * rcg2

    # 任务无关的HOI模仿奖励
    reward = rb * ro * rig * rcg

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength):
    """
    计算人形机器人重置条件
    
    参数：
    - reset_buf: 重置缓冲区
    - progress_buf: 进度缓冲区
    - contact_buf: 接触力缓冲区
    - rigid_body_pos: 刚体位置
    - max_episode_length: 最大episode长度
    - enable_early_termination: 是否启用提前终止
    - termination_heights: 终止高度阈值
    - hoi_ref: 参考HOI观测
    - hoi_obs: 当前HOI观测
    - envid2episode_lengths: 环境ID到episode长度的映射
    - isTest: 是否测试模式
    - maxEpisodeLength: 最大episode长度配置
    
    返回：
    - reset: 需要重置的环境
    - terminated: 已终止的环境
    """
    terminated = torch.zeros_like(reset_buf)  # 初始化终止缓冲区

    # 提前终止条件检查
    if enable_early_termination:
        body_height = rigid_body_pos[:, 0, 2]  # 根高度
        body_fall = body_height < termination_heights  # 判断是否摔倒
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)  # 仅在第二帧之后才检查摔倒
        
        # 更新终止缓冲区
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    # 检查是否达到最大episode长度
    if isTest and maxEpisodeLength > 0:
        # 测试模式下使用配置的最大长度
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    else:
        # 训练模式下使用每个环境的episode长度
        reset = torch.where(progress_buf >= envid2episode_lengths - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated