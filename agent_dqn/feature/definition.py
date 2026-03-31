#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_dqn.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


def reward_process(end_dist, history_dist, got_treasure=False, got_buff=False, treasure_dist=1.0, has_treasure_vision=0.0):
    """
    更新后的奖励函数，加入了对宝箱和Buff的奖励激励
    """
    # 基础步数惩罚：催促尽快完成
    step_reward = -0.001

    # 终点导向：距离终点越近越好
    end_reward = -0.02 * end_dist

    # 探索奖励：离开自己走过的地方
    dist_reward = min(0.001, 0.05 * history_dist)

    # =============== 新增的丰富奖励机制 ===================
    # 1. 吃到宝箱给予高额奖励（相当于小目标完成）
    treasure_collected_reward = 1.0 if got_treasure else 0.0

    # 2. 视野内有宝箱，鼓励向宝箱靠近
    treasure_approach_reward = 0.0
    if has_treasure_vision > 0:
        # treasure_dist 是 [0, 1] 归一化的距离。距离越近（越小），奖励越大
        treasure_approach_reward = 0.01 * (1.0 - treasure_dist)

    # 3. 吃到加速Buff给予中等奖励
    buff_collected_reward = 0.5 if got_buff else 0.0

    # 汇总：把所有的奖励加在一起
    total_reward = (
        step_reward
        + end_reward
        + dist_reward
        + treasure_collected_reward
        + treasure_approach_reward
        + buff_collected_reward
    )

    return [total_reward]


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = Config.DIM_OF_OBSERVATION
    legal_data_size = Config.DIM_OF_ACTION_DIRECTION
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[2 * obs_data_size : 2 * obs_data_size + legal_data_size],
        _obs_legal=s_data[2 * obs_data_size + legal_data_size : 2 * obs_data_size + 2 * legal_data_size],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
