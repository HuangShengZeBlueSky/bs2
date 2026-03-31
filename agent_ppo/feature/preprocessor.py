#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_ppo.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()

        # 新增追踪变量：为了计算奖励，需要记录上一帧的状态
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.current_treasure_count = 0
        self.current_buff_count = 0

    def _get_pos_feature(self, found, cur_pos, target_pos):
        # 如果没找到目标，返回全0特征
        if not found or target_pos is None:
            return np.zeros(6, dtype=np.float32)

        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
            dtype=np.float32
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, extra_info = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        # 获取得分信息（用于判断是否吃到宝箱/Buff发奖励）
        if "score_info" in obs:
            self.current_treasure_count = obs["score_info"].get("treasure_collected_count", 0)
            self.current_buff_count = obs["score_info"].get("buff_count", 0)

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # 英雄自身状态特征：是否加速，加速剩余时间，技能CD
        speed_up = hero.get("speed_up", 0)
        buff_time = norm(hero.get("buff_remain_time", 0), 2000) # 假设最长2000

        talent_status = 0 # 0是CD中，1是可用
        talent_cd = 0
        if "talent" in hero:
            talent_status = hero["talent"].get("status", 0)
            talent_cd = norm(hero["talent"].get("cooldown", 0), 2000) # 假设CD最长2000

        self.feature_hero_status = np.array([speed_up, buff_time, talent_status, talent_cd], dtype=np.float32)

        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # 终点位置、最近宝箱、最近Buff
        closest_treasure_dist = float('inf')
        closest_treasure_pos = None
        closest_buff_dist = float('inf')
        closest_buff_pos = None

        self.is_end_pos_found = False

        for organ in obs["frame_state"]["organs"]:
            # sub_type: 1代表宝箱, 2代表加速buff, 3代表起点, 4代表终点
            organ_pos = (organ["pos"]["x"], organ["pos"]["z"])

            # 终点逻辑 (保持不变)
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1: # 在视野内
                    self.end_pos = organ_pos
                    self.is_end_pos_found = True
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or getattr(self, "end_pos_dir", None) != end_pos_dir
                    or getattr(self, "end_pos_dis", None) != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )
                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

            # 找最近的可见宝箱 (status 1 表示可获取)
            elif organ["sub_type"] == 1 and organ["status"] == 1:
                dist = np.linalg.norm(np.array(self.cur_pos) - np.array(organ_pos))
                if dist < closest_treasure_dist:
                    closest_treasure_dist = dist
                    closest_treasure_pos = organ_pos

            # 找最近的可见加速Buff (status 1 表示可获取)
            elif organ["sub_type"] == 2 and organ["status"] == 1:
                dist = np.linalg.norm(np.array(self.cur_pos) - np.array(organ_pos))
                if dist < closest_buff_dist:
                    closest_buff_dist = dist
                    closest_buff_pos = organ_pos

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)

        # 计算各种物体的特征
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        # 新增特征计算
        has_treasure = 1 if closest_treasure_pos else 0
        self.feature_treasure_pos = self._get_pos_feature(has_treasure, self.cur_pos, closest_treasure_pos)

        has_buff = 1 if closest_buff_pos else 0
        self.feature_buff_pos = self._get_pos_feature(has_buff, self.cur_pos, closest_buff_pos)

        self.move_usable = True
        self.last_action = last_action

    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # 合法动作
        legal_action = self.get_legal_action()

        # 拼接所有特征 (原来的 + 新增的英雄状态、宝箱位置、Buff位置)
        feature = np.concatenate([
            self.cur_pos_norm,          # 2
            self.feature_end_pos,       # 6
            self.feature_history_pos,   # 6
            self.feature_hero_status,   # 4  (新!) 英雄自身状态/技能CD
            self.feature_treasure_pos,  # 6  (新!) 最近宝箱
            self.feature_buff_pos,      # 6  (新!) 最近Buff
            legal_action                # 8
        ])

        # 计算奖励参数：是否刚吃到了宝箱或Buff
        got_treasure = self.current_treasure_count > self.last_treasure_count
        got_buff = self.current_buff_count > self.last_buff_count

        # 更新记录
        self.last_treasure_count = self.current_treasure_count
        self.last_buff_count = self.current_buff_count

        # 调用修改后的 reward_process
        reward = reward_process(
            end_dist=self.feature_end_pos[-1],
            history_dist=self.feature_history_pos[-1],
            got_treasure=got_treasure,
            got_buff=got_buff,
            treasure_dist=self.feature_treasure_pos[-1] if self.feature_treasure_pos[0] else 1.0, # 如果没有看到宝箱，距离算最远(1.0)
            has_treasure_vision=self.feature_treasure_pos[0]
        )

        return (
            feature,
            legal_action,
            reward,
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action
