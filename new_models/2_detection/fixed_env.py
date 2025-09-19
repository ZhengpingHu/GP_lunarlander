#!/usr/bin/env python3
"""
fixed_env.py
Defines a custom Gymnasium environment for the LunarLander where the lander
can be reset to a specific, fixed initial angle and position.
(定义了一个自定义的Gymnasium环境，用于月球着陆器，
其中着陆器可以在重置时设置为特定的初始角度和位置。)
"""

import numpy as np
import math
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import Box2D

from gymnasium.envs.registration import register

# --- 从原始环境中导入常量 ---
from gymnasium.envs.box2d.lunar_lander import (
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    LEG_AWAY,
    LEG_DOWN,
    LEG_W,
    LEG_H,
    LEG_SPRING_TORQUE,
    LANDER_POLY,
)

class ContactDetector(Box2D.b2ContactListener):
    """
    Handles contact events to detect game-over conditions or leg contact.
    (处理接触事件以检测游戏结束条件或腿部接触。)
    """
    def __init__(self, env):
        Box2D.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        lander_contact = self.env.lander in bodies
        leg_contact = any(leg in bodies for leg in self.env.legs)

        if lander_contact and not leg_contact:
            self.env.game_over = True

        for i in range(2):
            if self.env.legs[i] in bodies:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class FixedLander(LunarLander):
    """
    A LunarLander environment that allows setting a fixed initial angle
    and position upon reset via the `options` dictionary.
    (一个月球着陆器环境，允许通过`options`字典在重置时设置固定的初始角度和位置。)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # --- 地形生成 (未改变) ---
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2 : CHUNKS // 2 + 3] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=Box2D.b2EdgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # --- 设置着陆器初始状态 ---
        # 默认值
        initial_x = VIEWPORT_W / SCALE / 2
        initial_y = self.helipad_y + (H * 0.3)
        theta = 0.0  # 弧度

        # **修改**: 如果提供了`options`字典，则覆盖默认值
        if options is not None:
            angle_deg = options.get("angle", 0.0)
            theta = math.radians(angle_deg)
            initial_x = options.get("x", initial_x)
            initial_y = options.get("y", initial_y)

        # --- 创建着陆器主体 ---
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=theta,
            fixtures=Box2D.b2FixtureDef(
                shape=Box2D.b2PolygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # --- 创建腿部 ---
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x, initial_y),
                angle=theta,
                fixtures=Box2D.b2FixtureDef(
                    shape=Box2D.b2PolygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)

            # **修正**: 修正关节定义以匹配原始环境，确保物理行为正确
            rjd = Box2D.b2RevoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.4
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.4

            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

# --- 向Gymnasium注册自定义环境 ---
register(
    id="FixedLander-v0",
    entry_point="fixed_env:FixedLander",
    max_episode_steps=1000,
)
