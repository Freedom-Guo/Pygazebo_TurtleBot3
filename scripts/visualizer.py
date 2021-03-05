import rospy
import os
import copy

import numpy as np

from tf import transformations
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker, MarkerArray


class RvizVisulizer(object):
    def __init__(self, num_agents, agent_size):
        self.num_agents = num_agents
        self.agent_size = agent_size

        self.agent_markers = MarkerArray()
        self.agent_markers_id = 0
        self.agent_markers_pub = rospy.Publisher(
            'agent_markers', MarkerArray, queue_size=10
            )

        self.path_markers = MarkerArray()
        self.path_markers_id = 0
        self.path_markers_pub = rospy.Publisher(
            'path_markers', MarkerArray, queue_size=10
            )

        self.start_markers = MarkerArray()
        self.start_markers_id = 0
        self.start_markers_pub = rospy.Publisher(
            'start_markers', MarkerArray, queue_size=10
            )

        self.goal_markers = MarkerArray()
        self.goal_markers_id = 0
        self.goal_markers_pub = rospy.Publisher(
            'goal_markers', MarkerArray, queue_size=10
            )

    def reset(self, starts, goals):
        if len(self.path_markers.markers) != 0:
            for m in self.path_markers.markers:
                m.action = Marker.DELETE
        self.path_markers_pub.publish(self.path_markers)

        self.agent_markers = MarkerArray()
        self.agent_markers_id = 0
        self.path_markers = MarkerArray()
        self.path_markers_id = 0
        self.goal_markers = MarkerArray()
        self.goal_markers_id = 0
        self.start_markers = MarkerArray()
        self.start_markers_id = 0

        self.start_time = rospy.get_time()

        self._show_start(starts)
        self._show_goal(goals)

    def display(self, poses, prev_poses, acts):
        self._show_agent(poses, acts)
        self._show_path(
            poses, prev_poses,
            rospy.get_time() - self.start_time
            )
        
    def _show_agent(self, poses, acts):
        self.agent_markers = MarkerArray()
        self.agent_markers_id = 0

        for i, (pose, act) in enumerate(zip(poses, acts)):
            self._add_marker(
                Marker.MESH_RESOURCE, 
                [1., 0.12, 0., 1.],
                [self.agent_size, self.agent_size, self.agent_size],
                pose=pose,
                ns='agent')

            self._add_marker(
                Marker.TEXT_VIEW_FACING, 
                [0., 0., 0., .8],
                [0., 0., 1.5 * self.agent_size],
                pose=pose,
                ns='agent',
                text=str(i))

            abs_act = np.linalg.norm(act)
            q = transformations.quaternion_from_euler(
                    act[0] / abs_act, 
                    act[1] / abs_act, 
                    act[2] / abs_act)
            pose = pose.tolist()
            q = q.tolist()
            pose += q
            self._add_marker(
                Marker.ARROW, 
                [0.25, 0.74, 0.15, 0.7],
                [abs_act / 5.0, 0.02, 0.02],
                pose=pose,
                ns='agent')
            
        self.agent_markers_pub.publish(self.agent_markers)

    def _show_path(self, poses, prev_poses, t):
        for prev_pose, pose in zip(prev_poses, poses):
            self._add_marker(
                Marker.LINE_STRIP, 
                [0.8, 0.0, 0.0, 0.5], 
                [0.3, 0.1, 0.1],
                points=[prev_pose, pose],
                ns='path')

        self.path_markers_pub.publish(self.path_markers)

    def _show_start(self, starts):
        for i, start in enumerate(starts):
            self._add_marker(
                Marker.SPHERE, 
                [0., 1., 0., .8], 
                [.3, .3, .3],
                start,
                ns='start')

            self._add_marker(
                Marker.TEXT_VIEW_FACING, 
                [0., 0.0, 1., 1.], 
                [.0, .0, .50],
                start,
                ns='start',
                text=str(i))

        self.start_markers_pub.publish(self.start_markers)

    def _show_goal(self, goals):
        for i, goal in enumerate(goals):
            self._add_marker(
                Marker.SPHERE,
                [1., 1., 0., .8], 
                [.3, .3, .3],
                goal,
                ns='goal')

            self._add_marker(
                Marker.TEXT_VIEW_FACING, 
                [0., 0.5, 1.0, 1.], 
                [.0, .0, .5],
                goal,
                ns='goal',
                text=str(i))
                
        self.goal_markers_pub.publish(self.goal_markers)

    def _add_marker(self,
                    markerType,
                    color,
                    scale,
                    pose=None,
                    ns=None,
                    text=None,
                    points=None):

        if pose is not None:
            pose = self._to_pose(pose)

        marker = Marker()
        marker.header.frame_id = 'ground_truth'
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.type = markerType
        marker.action = Marker.ADD
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        if pose:
            marker.pose = pose
        if text:
            marker.text = text
            marker.pose.position.z += 0.1
            if ns == "goal":
                marker.pose.position.y += 0.2
            if ns == "start":
                marker.pose.position.y += 0.2

        if points:
            marker.points = []
            for p in points:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = p[2]
                marker.points.append(pt)

        if ns == 'agent':
            path = os.path.abspath(
                "../models/quadrotor/meshes/quadrotor_base.dae"
                )
            marker.mesh_resource = "file://" + path
            marker.id = self.agent_markers_id
            self.agent_markers.markers.append(marker)
            self.agent_markers_id += 1

        if ns == 'path':
            marker.id = self.path_markers_id
            self.path_markers.markers.append(marker)
            self.path_markers_id += 1

        if ns == 'start':
            marker.id = self.start_markers_id
            self.start_markers.markers.append(marker)
            self.start_markers_id += 1

        if ns == 'goal':
            marker.id = self.goal_markers_id
            self.goal_markers.markers.append(marker)
            self.goal_markers_id += 1

    def _to_pose(self, data):
        pose = Pose()

        if len(data) == 3:
            pose.position.x = data[0]
            pose.position.y = data[1]
            pose.position.z = data[2]
            return pose
        elif len(data) == 7:
            pose.position.x = data[0]
            pose.position.y = data[1]
            pose.position.z = data[2]
            pose.orientation.x = data[3]
            pose.orientation.y = data[4]
            pose.orientation.z = data[5]
            pose.orientation.w = data[6]
            return pose
        else:
            rospy.logerr("Invalid pose data.")
            raise RuntimeError