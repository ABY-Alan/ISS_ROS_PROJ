#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity, GetModelList

_SDF_SPHERE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.08</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def spawn_goal_point(x: float, y: float, z: float = 0.2, name: str = "goal_point") -> Tuple[float, float, float]:
    """
    Spawn a simple static sphere marker in Gazebo at (x, y, z) and return (x, y, z).
    Requires Gazebo running with gazebo_ros service /spawn_entity available.
    """
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=None)
            
    node = Node(f"spawn_goal_point_node_{name}") # 节点名增加 name 以防冲突

    try:
        cli = node.create_client(SpawnEntity, "/spawn_entity")
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Service /spawn_entity not available. Is Gazebo running?")

        req = SpawnEntity.Request()
        req.name = name
        req.xml = _SDF_SPHERE.format(model_name=name)
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)
        req.initial_pose.orientation.w = 1.0

        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)

        if fut.result() is None:
            raise RuntimeError("SpawnEntity call did not return (timeout/hang).")

        # return (float(x), float(y), float(z))
        return (float(x), float(y))

    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()


def delete_all_goals(keyword: str = "goal"):
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()
    node = Node("delete_all_goals_node")

    # 获取模型列表
    get_cli = node.create_client(GetModelList, "/get_model_list")
    if not get_cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("/get_model_list service not available")

    req = GetModelList.Request()
    fut = get_cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut)

    if fut.result() is None:
        raise RuntimeError("Failed to get model list")

    model_names = fut.result().model_names

    # 删除包含 keyword 的模型
    delete_cli = node.create_client(DeleteEntity, "/delete_entity")
    if not delete_cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("/delete_entity service not available")

    for name in model_names:
        if keyword in name:
            del_req = DeleteEntity.Request()
            del_req.name = name
            del_fut = delete_cli.call_async(del_req)
            rclpy.spin_until_future_complete(node, del_fut)

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()