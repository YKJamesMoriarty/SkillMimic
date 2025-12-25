# -*- coding: utf-8 -*-
"""
运动数据处理脚本

该脚本用于加载、处理和管理人类-物体交互(HOI)运动数据，主要用于强化学习环境中的运动模仿学习。

主要功能：
1. 加载运动数据文件(.pt格式)
2. 处理和转换运动序列
3. 计算运动速度和加速度
4. 四元数平滑和转换
5. 运动数据采样
6. 生成初始状态
7. 管理奖励权重

依赖库：
- os: 文件和目录操作
- glob: 文件路径匹配
- torch: 深度学习框架
- numpy: 数值计算
- re: 正则表达式
- utils.torch_utils: 自定义PyTorch工具函数
"""

# 导入必要的库
import os  # 用于文件和目录操作
import glob  # 用于文件路径匹配
import torch  # PyTorch深度学习框架
import numpy as np  # 用于数值计算
import torch.nn.functional as F  # PyTorch的功能函数库
import re  # 用于正则表达式操作
from utils import torch_utils  # 自定义的PyTorch工具函数

# 运动数据处理类，负责加载、处理和管理运动数据
class MotionDataHandler:
    """
    运动数据处理类
    
    该类负责加载、处理和管理人类-物体交互(HOI)运动数据，为强化学习环境提供运动参考。
    """
    
    # 初始化方法，设置运动数据处理器的基本参数
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False):
        """
        初始化运动数据处理器
        
        参数：
        motion_file (str): 运动数据文件或目录路径
        device (torch.device): 计算设备(CPU或GPU)
        key_body_ids (list): 关键身体部位的ID列表
        cfg (dict): 配置信息字典
        num_envs (int): 环境数量
        max_episode_length (int): 最大episode长度
        reward_weights_default (dict): 默认奖励权重字典
        init_vel (bool): 是否初始化速度
        play_dataset (bool): 是否播放数据集模式
        """
        self.device = device  # 计算设备（CPU或GPU）
        self._key_body_ids = key_body_ids  # 关键身体部位的ID列表
        self.cfg = cfg  # 配置信息
        self.init_vel = init_vel  # 是否初始化速度
        self.play_dataset = play_dataset  # 是否播放数据集模式
        self.max_episode_length = max_episode_length  # 最大 episode 长度
        
        self.hoi_data_dict = {}  # 存储人类-物体交互数据的字典
        self.hoi_data_label_batch = None  # 人类-物体交互数据标签批次
        self.motion_lengths = None  # 各运动序列的长度
        self.load_motion(motion_file)  # 加载运动数据

        self.num_envs = num_envs  # 环境数量
        # 环境ID到运动ID的映射张量
        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 各环境的episode长度张量
        self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.reward_weights_default = reward_weights_default  # 默认奖励权重
        self.reward_weights = {}  # 奖励权重字典
        # 初始化各奖励权重为默认值
        self.reward_weights["p"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["p"])
        self.reward_weights["r"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["r"])
        self.reward_weights["pv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["pv"])
        self.reward_weights["rv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["rv"])
        # 移除与物体相关的奖励权重，因为拳击数据没有物体
        for key in ["op", "ig", "cg1", "cg2", "or", "opv", "orv"]:
            if key in self.reward_weights_default:
                self.reward_weights[key] = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

    # 加载运动数据的方法
    def load_motion(self, motion_file):
        """
        加载运动数据
        
        参数：
        motion_file (str): 运动数据文件或目录路径
        """
        self.skill_name = os.path.basename(motion_file)  # 获取技能名称
        # 如果motion_file是文件，则使用该文件；否则，递归查找所有.pt文件
        all_seqs = [motion_file] if os.path.isfile(motion_file) \
            else glob.glob(os.path.join(motion_file, '**', '*.pt'), recursive=True)
        self.num_motions = len(all_seqs)  # 运动序列数量
        # 初始化运动长度张量，记录每个运动的帧数
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long)
        self.motion_class = np.zeros(len(all_seqs), dtype=int)  # 初始化运动类别数组
        # 上篮目标位置张量，用于篮球运动
        self.layup_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        # 根节点目标位置张量，用于篮球运动
        self.root_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)

        all_seqs.sort(key=self._sort_key)  # 按照文件名排序
        # 遍历所有运动序列 
        for i, seq_path in enumerate(all_seqs):
            loaded_dict = self._process_sequence(seq_path)  # 处理单个序列
            self.hoi_data_dict[i] = loaded_dict  # 存储处理后的序列数据
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0]  # 记录序列长度
            self.motion_class[i] = int(loaded_dict['hoi_data_text'])  # 记录序列类别
            # 特殊处理上篮和投篮技能，记录目标位置
            if self.skill_name in ['layup', "SHOT_up"]:
                layup_target_ind = torch.argmax(loaded_dict['obj_pos'][:, 2])  # 找到物体最高点
                self.layup_target[i] = loaded_dict['obj_pos'][layup_target_ind]  # 记录目标位置
                self.root_target[i] = loaded_dict['root_pos'][layup_target_ind]  # 记录根节点位置
        self._compute_motion_weights(self.motion_class)  # 计算运动权重
        print(f"--------已加载 {len(all_seqs)} 个运动序列--------")  # 打印加载信息
    
    # 文件名排序的关键字函数
    def _sort_key(self, filename):
        """
        文件名排序的关键字函数
        
        参数：
        filename (str): 文件名
        
        返回：
        int: 用于排序的关键字
        """
        match = re.search(r'\d+\.pt$', filename)  # 查找文件名末尾的数字
        return int(match.group().replace('.pt', '')) if match else -1  # 返回数字作为排序关键字

    # 处理单个运动序列的方法
    def _process_sequence(self, seq_path):
        """
        处理单个运动序列
        
        参数：
        seq_path (str): 运动序列文件路径
        
        返回：
        dict: 处理后的运动序列数据
        """
        loaded_dict = {}  # 存储处理后的序列数据
        hoi_data = torch.load(seq_path)  # 加载.pt文件数据
        loaded_dict['hoi_data_text'] = os.path.basename(seq_path)[0:3]  # 从文件名获取数据文本标签
        # loaded_dict['hoi_data']的形状 ： [帧数, 总特征维度]
        loaded_dict['hoi_data'] = hoi_data.detach().to(self.device)  # 将数据转移到指定设备
        data_frames_scale = self.cfg["env"]["dataFramesScale"]  # 获取数据帧缩放比例
        fps_data = self.cfg["env"]["dataFPS"] * data_frames_scale  # 计算数据帧率

        # 提取根节点位置数据 [帧数, 3]，” ：冒号“ 表示选择所有帧，即保留第一维的所有数据
        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        # 计算根节点位置速度 [帧数, 3]
        loaded_dict['root_pos_vel'] = self._compute_velocity(loaded_dict['root_pos'], fps_data)

        # 提取根节点3D旋转数据（指数映射）[帧数, 3]
        loaded_dict['root_rot_3d'] = loaded_dict['hoi_data'][:, 3:6].clone()
        # 将旋转向量（指数映射）转换为四元数 [帧数, 4]
        loaded_dict['root_rot'] = torch_utils.exp_map_to_quat(loaded_dict['root_rot_3d']).clone()
        # 平滑四元数序列，确保相邻四元数的连续性
        self.smooth_quat_seq(loaded_dict['root_rot'])
        # 计算相邻四元数的差分，得到相对旋转
        q_diff = torch_utils.quat_multiply(
            torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1, :].clone()), 
            loaded_dict['root_rot'][1:, :].clone()
        )
        # 将四元数差分转换为角度和轴 [帧数-1, 1], [帧数-1, 3]
        angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        # 将角度轴转换为指数映射 [帧数-1, 3]
        exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        # 计算根节点旋转速度，乘以帧率转换为每秒弧度 [帧数-1, 3]
        loaded_dict['root_rot_vel'] = exp_map*fps_data
        # 补全初始帧速度为零，确保与原始数据帧数一致 [帧数, 3]
        loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to(self.device), loaded_dict['root_rot_vel']), dim=0)
        ### dof_pos关节旋转角度，body_pos是身体部位的绝对位置
        # 提取自由度位置数据 [帧数, 156]，156=52个关节*3个自由度
        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+156].clone()
        # 计算自由度位置速度 [帧数, 156]
        loaded_dict['dof_pos_vel'] = self._compute_velocity(loaded_dict['dof_pos'], fps_data)

        data_length = loaded_dict['hoi_data'].shape[0]  # 获取数据长度（帧数）
        # 提取身体位置数据并重塑为 [帧数, 53, 3]，53个身体部位
        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 165: 165+53*3].clone().view(data_length, 53, 3)
        # 提取关键身体部位位置数据 [帧数, N*3]，N为关键身体部位数量
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(data_length, -1).clone()
        # 计算关键身体部位位置速度 [帧数, N*3]
        loaded_dict['key_body_pos_vel'] = self._compute_velocity(loaded_dict['key_body_pos'], fps_data)

        # 提取物体位置数据 [帧数, 3]
        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318+6:321+6].clone()
        # 计算物体位置速度 [帧数, 3]
        loaded_dict['obj_pos_vel'] = self._compute_velocity(loaded_dict['obj_pos'], fps_data)

        # 提取物体旋转数据 [帧数, 3]
        loaded_dict['obj_rot'] = -loaded_dict['hoi_data'][:, 321+6:324+6].clone()
        # 计算物体旋转速度 [帧数, 3]
        loaded_dict['obj_rot_vel'] = self._compute_velocity(loaded_dict['obj_rot'], fps_data)
        # 初始化速度处理，使用第二帧速度作为初始速度
        if self.init_vel:
            loaded_dict['obj_pos_vel'][0] = loaded_dict['obj_pos_vel'][1]  # 使用第二帧速度作为初始速度
        # 转换物体旋转向量为四元数 [帧数, 4]
        loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(-loaded_dict['hoi_data'][:, 327:330]).clone()

        # 提取接触数据 [帧数, 1]
        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330+6:331+6].clone())

        # 重新组合hoi_data，包含所需的所有数据
        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'],  # 根节点位置 [帧数, 3]
            loaded_dict['root_rot_3d'],  # 根节点3D旋转 [帧数, 3]
            loaded_dict['dof_pos'],  # 自由度位置 [帧数, 156]
            loaded_dict['dof_pos_vel'],  # 自由度位置速度 [帧数, 156]
            loaded_dict['obj_pos'],  # 物体位置 [帧数, 3]
            loaded_dict['obj_rot'],  # 物体旋转（四元数）[帧数, 4]
            loaded_dict['obj_pos_vel'],  # 物体位置速度 [帧数, 3]
            loaded_dict['key_body_pos'],  # 关键身体部位位置 [帧数, N*3]
            loaded_dict['contact']  # 接触数据 [帧数, 1]
        ), dim=-1)  # 按最后一个维度拼接
        
        return loaded_dict  # 返回处理后的序列数据

    # 计算速度的方法
    def _compute_velocity(self, positions, fps):
        """
        计算速度
        
        参数：
        positions (torch.Tensor): 位置张量 [帧数, 特征维度]
        fps (float): 帧率
        
        返回：
        torch.Tensor: 速度张量 [帧数, 特征维度]
        """
        # 计算相邻帧之间的位置差 [帧数-1, 特征维度]
        velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
        # 补全初始帧速度为零 [帧数, 特征维度]
        velocity = torch.cat((torch.zeros((1, positions.shape[-1])).to(self.device), velocity), dim=0)
        return velocity  # 返回速度数据

    # 平滑四元数序列的方法
    def smooth_quat_seq(self, quat_seq):
        """
        平滑四元数序列
        
        确保相邻四元数之间的连续性，通过反转四元数使得点积为正。
        
        参数：
        quat_seq (torch.Tensor): 四元数序列 [帧数, 4]
        
        返回：
        torch.Tensor: 平滑后的四元数序列 [帧数, 4]
        """
        n = quat_seq.size(0)  # 获取序列长度

        # 遍历四元数序列，确保相邻四元数之间的点积为正
        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])  # 计算相邻四元数的点积
            if dot_product < 0:  # 如果点积为负，反转当前四元数
                quat_seq[i] *=-1

        return quat_seq  # 返回平滑后的四元数序列

    # 计算运动权重的方法
    def _compute_motion_weights(self, motion_class):
        """
        计算运动权重
        
        根据运动类别的分布计算每个运动的采样权重，用于不平衡数据集的采样。
        
        参数：
        motion_class (np.array): 运动类别数组
        """
        # 获取唯一类别和计数
        unique_classes, counts = np.unique(motion_class, return_counts=True)
        # 创建类别到索引的映射字典
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        # 计算类别权重，反比于出现次数
        class_weights = 1 / counts
        # 特殊处理类别1，增加其采样概率（技能选择）
        if 1 in class_to_index:
            class_weights[class_to_index[1]]*=2
        # 将每个运动的类别转换为索引
        indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        # 计算每个运动的权重
        self._motion_weights = class_weights[indexed_classes]

    # 采样运动序列的方法
    def sample_motions(self, n):
        """
        采样运动序列
        
        根据运动权重采样n个运动ID。
        
        参数：
        n (int): 采样数量
        
        返回：
        torch.Tensor: 采样的运动ID张量 [n]
        """
        # 根据权重采样n个运动ID，允许重复
        motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=n, replacement=True)
        return motion_ids  # 返回采样的运动ID

    # 采样时间的方法
    def sample_time(self, motion_ids, truncate_time=None):
        """
        采样时间
        
        为每个运动采样起始帧。
        
        参数：
        motion_ids (torch.Tensor): 运动ID张量
        truncate_time (int, optional): 截断时间
        
        返回：
        torch.Tensor: 采样的起始帧张量
        """
        # 获取运动长度
        lengths = self.motion_lengths[motion_ids].cpu().numpy()

        start = 2  # 采样起始帧，避开初始不稳定帧
        end = lengths - 2  # 采样结束帧，避开末尾不稳定帧

        assert np.all(end > start)  # 确保所有运动长度都足够

        # 随机采样起始帧，+1因为np.random.randint的上限是开区间
        motion_times = np.random.randint(start, end + 1)

        # 转换为张量
        motion_times = torch.tensor(motion_times, device=self.device, dtype=torch.int)

        # 如果指定了截断时间，则调整采样时间
        if truncate_time is not None:
            assert truncate_time >= 0  # 确保截断时间非负
            # 确保有足够的剩余帧数
            motion_times = torch.min(motion_times, self.motion_lengths[motion_ids] - truncate_time)

        # 如果是播放数据集模式，则从第0帧开始
        if self.play_dataset:
            motion_times = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        return motion_times  # 返回采样的时间

    # 获取初始状态的方法
    def get_initial_state(self, env_ids, motion_ids, start_frames):
        """
        获取初始状态
        
        为每个环境获取初始状态，包括位置、旋转、速度等。
        
        参数：
        env_ids (list): 环境ID列表
        motion_ids (torch.Tensor): 运动ID张量
        start_frames (torch.Tensor): 起始帧张量
        
        返回：
        tuple: 包含初始状态的元组
        """
        assert len(motion_ids) == len(env_ids)  # 确保motion_ids和env_ids长度一致
        # 计算有效长度，即从起始帧到运动结束的帧数
        valid_lengths = self.motion_lengths[motion_ids] - start_frames
        # 设置每个环境的episode长度，取有效长度和最大episode长度的最小值
        self.envid2episode_lengths[env_ids] = torch.where(valid_lengths < self.max_episode_length,
                                    valid_lengths, self.max_episode_length)

        hoi_data_list = []  # 存储hoi数据
        root_pos_list = []  # 存储根节点位置
        root_rot_list = []  # 存储根节点旋转
        root_vel_list = []  # 存储根节点速度
        root_ang_vel_list = []  # 存储根节点角速度
        dof_pos_list = []  # 存储自由度位置
        dof_vel_list = []  # 存储自由度速度
        obj_pos_list = []  # 存储物体位置
        obj_pos_vel_list = []  # 存储物体位置速度
        obj_rot_list = []  # 存储物体旋转
        obj_rot_vel_list = []  # 存储物体旋转速度

        # 遍历所有环境
        for i, env_id in enumerate(env_ids):
            motion_id = motion_ids[i].item()  # 获取运动ID
            start_frame = start_frames[i].item()  # 获取起始帧

            self.envid2motid[env_id] = motion_id  # 记录环境ID到运动ID的映射
            episode_length = self.envid2episode_lengths[env_id].item()  # 获取episode长度

            # 根据数据文本标签选择不同的初始状态生成方法
            if self.hoi_data_dict[motion_id]['hoi_data_text'] == '000':
                # 特殊情况：物体位置随机初始化
                state = self._get_special_case_initial_state(motion_id, start_frame, episode_length)
            else:
                # 一般情况：使用数据中的初始状态
                state = self._get_general_case_initial_state(motion_id, start_frame, episode_length)

            # 更新奖励权重
            for k in self.reward_weights_default:
                self.reward_weights[k][env_id] =  torch.tensor(state['reward_weights'][k], dtype=torch.float32, device=self.device)
            # 收集状态数据
            hoi_data_list.append(state["hoi_data"])
            root_pos_list.append(state['init_root_pos'])
            root_rot_list.append(state['init_root_rot'])
            root_vel_list.append(state['init_root_pos_vel'])
            root_ang_vel_list.append(state['init_root_rot_vel'])
            dof_pos_list.append(state['init_dof_pos'])
            dof_vel_list.append(state['init_dof_pos_vel'])
            obj_pos_list.append(state["init_obj_pos"])
            obj_pos_vel_list.append(state["init_obj_pos_vel"])
            obj_rot_list.append(state["init_obj_rot"])
            obj_rot_vel_list.append(state["init_obj_rot_vel"])

        # 将列表转换为张量，按环境维度堆叠
        hoi_data = torch.stack(hoi_data_list, dim=0)
        root_pos = torch.stack(root_pos_list, dim=0)
        root_rot = torch.stack(root_rot_list, dim=0)
        root_vel = torch.stack(root_vel_list, dim=0)
        root_ang_vel = torch.stack(root_ang_vel_list, dim=0)
        dof_pos = torch.stack(dof_pos_list, dim=0)
        dof_vel = torch.stack(dof_vel_list, dim=0)
        obj_pos = torch.stack(obj_pos_list, dim=0)
        obj_pos_vel = torch.stack(obj_pos_vel_list, dim=0)
        obj_rot = torch.stack(obj_rot_list, dim=0)
        obj_rot_vel = torch.stack(obj_rot_vel_list, dim=0)

        # 返回初始状态元组
        return hoi_data, \
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, \
                obj_pos, obj_pos_vel, obj_rot, obj_rot_vel
                
    # 获取特殊情况初始状态的方法
    def _get_special_case_initial_state(self, motion_id, start_frame, episode_length):
        """
        获取特殊情况初始状态
        
        物体位置随机初始化的情况。
        
        参数：
        motion_id (int): 运动ID
        start_frame (int): 起始帧
        episode_length (int): episode长度
        
        返回：
        dict: 初始状态字典
        """
        # 对hoi数据进行填充，确保长度为max_episode_length
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)  # (left, right, top, bottom)
        )

        return {
            "reward_weights": self._get_special_case_reward_weights(),  # 获取特殊情况奖励权重
            "hoi_data": hoi_data,  # hoi数据
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],  # 初始根节点位置
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],  # 初始根节点旋转
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],  # 初始根节点位置速度
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],  # 初始根节点旋转速度
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],  # 初始自由度位置
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],  # 初始自由度位置速度
            "init_obj_pos": (torch.rand(3, device=self.device) * 10 - 5),  # 随机初始化物体位置 [-5, 5]
            "init_obj_pos_vel": torch.rand(3, device=self.device) * 5,  # 随机初始化物体位置速度 [0, 5]
            "init_obj_rot": torch.rand(4, device=self.device),  # 随机初始化物体旋转
            "init_obj_rot_vel": torch.rand(3, device=self.device) * 0.1  # 随机初始化物体旋转速度 [-0.1, 0.1]
        }

    # 获取一般情况初始状态的方法
    def _get_general_case_initial_state(self, motion_id, start_frame, episode_length):
        """
        获取一般情况初始状态
        
        使用数据中的初始状态。
        
        参数：
        motion_id (int): 运动ID
        start_frame (int): 起始帧
        episode_length (int): episode长度
        
        返回：
        dict: 初始状态字典
        """
        # 对hoi数据进行填充，确保长度为max_episode_length
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)  # (left, right, top, bottom)
        )

        return {
            "reward_weights": self._get_general_case_reward_weights(),  # 获取一般情况奖励权重
            "hoi_data": hoi_data,  # hoi数据
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],  # 初始根节点位置
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],  # 初始根节点旋转
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],  # 初始根节点位置速度
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],  # 初始根节点旋转速度
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],  # 初始自由度位置
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],  # 初始自由度位置速度
            "init_obj_pos": self.hoi_data_dict[motion_id]['obj_pos'][start_frame, :],  # 初始物体位置
            "init_obj_pos_vel": self.hoi_data_dict[motion_id]['obj_pos_vel'][start_frame, :],  # 初始物体位置速度
            "init_obj_rot": self.hoi_data_dict[motion_id]['obj_rot'][start_frame, :],  # 初始物体旋转
            "init_obj_rot_vel": self.hoi_data_dict[motion_id]['obj_rot_vel'][start_frame, :]  # 初始物体旋转速度
        }

    # 获取特殊情况奖励权重的方法
    def _get_special_case_reward_weights(self):
        """
        获取特殊情况奖励权重
        
        物体位置随机初始化的情况。
        
        返回：
        dict: 奖励权重字典
        """
        reward_weights = self.reward_weights_default  # 获取默认奖励权重
        return {
            "p": reward_weights["p"],  # 位置奖励权重
            "r": reward_weights["r"],  # 旋转奖励权重
            "op": 0.,  # 物体位置奖励权重设为0
            "ig": reward_weights["ig"] * 0.,  # 惯性奖励权重设为0（作者原本就这么写的）
            "cg1": 0.,  # 接触奖励权重1设为0
            "cg2": 0.,  # 接触奖励权重2设为0
            "pv": reward_weights["pv"],  # 位置速度奖励权重
            "rv": reward_weights["rv"],  # 旋转速度奖励权重
            "or": 0.,  # 物体旋转奖励权重
            "opv": 0.,  # 物体位置速度奖励权重
            "orv": 0.,  # 物体旋转速度奖励权重
        }

    # 获取一般情况奖励权重的方法
    def _get_general_case_reward_weights(self):
        """
        获取一般情况奖励权重
        
        使用数据中的初始状态的情况。
        
        返回：
        dict: 奖励权重字典
        """
        reward_weights = self.reward_weights_default  # 获取默认奖励权重
        return {
            "p": reward_weights["p"],  # 位置奖励权重
            "r": reward_weights["r"],  # 旋转奖励权重
            "op": 0.,  # 物体位置奖励权重
            "ig": reward_weights["ig"],  # 惯性奖励权重
            "cg1": 0.,  # 接触奖励权重1
            "cg2": 0.,  # 接触奖励权重2
            "pv": reward_weights["pv"],  # 位置速度奖励权重
            "rv": reward_weights["rv"],  # 旋转速度奖励权重
            "or": 0.,  # 物体旋转奖励权重
            "opv": 0.,  # 物体位置速度奖励权重
            "orv": 0.,  # 物体旋转速度奖励权重
        }