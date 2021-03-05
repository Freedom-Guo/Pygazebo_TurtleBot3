import argparse
import os

import numpy as np
import math
from lxml import etree
from lxml.etree import Element
import random


class Creator(object):
    def __init__(self, 
                 robot_num,
                 env_size, 
                 env_type="circle", 
                 file_name="../world/basic.world", 
                 modelname="quadrotor"):

        parser = etree.XMLParser(remove_blank_text=True)
        self.tree = etree.parse(
            os.path.abspath(file_name),
            parser
            )
        self.robot_num = robot_num
        self.env_size = env_size
        self.env_type = env_type
        self.modelname = modelname
        self.physics = etree.parse(
            os.path.abspath("../world/physics.world"),
            parser
            )
        self.spherical_coordinates = etree.parse(
            os.path.abspath("../world/spherical_coordinates.world"),
            parser
            )

    def run(self):
        root = self.tree.getroot()
        if self.env_type == "circle":
            for i in range(self.robot_num):
                angle = i * 2 * np.pi / self.robot_num
                sx = self.env_size * np.cos(angle)
                sy = self.env_size * np.sin(angle)
                sz = 10
                include = Element("include")
                name = Element("name")
                name.text = "Robot_%03d" % (i)
                uri = Element("uri")
                uri.text = "model://"+self.modelname
                pose = Element("pose")
                pose.text = str(sx)+" "+str(sy)+" "+str(sz)+" 0 0.000143 -1.57317"
                include.append(name)
                include.append(uri)
                include.append(pose)
                root[0].append(include)
                include_target = Element("include")
                name_target = Element("name")
                name_target.text = "Goal_%03d" % (i)
                uri_target = Element("uri")
                uri_target.text = "model://Goal__New"
                pose_target = Element("pose")
                pose_target.text = str(sx*2)+" "+str(sy*2)+" "+str(sz)+" 0 0.000143 -1.57317"
                include_target.append(name_target)
                include_target.append(uri_target)
                include_target.append(pose_target)
                root[0].append(include_target)
        elif self.env_type=="random":
            for i in range(self.robot_num):
                sx = 10*random.random()
                sy = 10*random.random()
                sz = 10*random.random()
                include = Element("include")
                name = Element("name")
                name.text = "Robot_%03d" % (i)
                uri = Element("uri")
                uri.text = "model://"+self.modelname
                pose = Element("pose")
                pose.text = str(sx)+" "+str(sy)+" "+str(sz)+" 0 0.000143 -1.57317"
                include.append(name)
                include.append(uri)
                include.append(pose)
                root[0].append(include)
                include_target = Element("include")
                name_target = Element("name")
                name_target.text = "Goal_%03d" % (i)
                uri_target = Element("uri")
                uri_target.text = "model://Goal__New"
                pose_target = Element("pose")
                pose_target.text = str(sx*2)+" "+str(sy*2)+" "+str(sz)+" 0 0.000143 -1.57317"
                include_target.append(name_target)
                include_target.append(uri_target)
                include_target.append(pose_target)
                root[0].append(include_target)
        elif self.env_type=="ball":
            parts = int(math.sqrt(self.robot_num))+1
            for i in range(self.robot_num):
                stheta = (i%parts)/float(parts) * np.pi
                sfy = int(i/parts)/float(parts) * np.pi*2
                sx = self.env_size * np.sin(stheta) * np.cos(sfy)
                sy = self.env_size * np.sin(stheta) * np.sin(sfy)
                sz = self.env_size * np.cos(stheta) + self.env_size
                include = Element("include")
                name = Element("name")
                name.text = "Robot_%03d" % (i)
                uri = Element("uri")
                uri.text = "model://"+self.modelname
                pose = Element("pose")
                pose.text = str(sx)+" "+str(sy)+" "+str(sz)+" 0 0.000143 -1.57317"
                include.append(name)
                include.append(uri)
                include.append(pose)
                root[0].append(include)
                include_target = Element("include")
                name_target = Element("name")
                name_target.text = "Goal_%03d" % (i)
                uri_target = Element("uri")
                uri_target.text = "model://Goal__New"
                pose_target = Element("pose")
                pose_target.text = str(sx*2)+" "+str(sy*2)+" "+str(sz)+" 0 0.000143 -1.57317"
                include_target.append(name_target)
                include_target.append(uri_target)
                include_target.append(pose_target)
                root[0].append(include_target)
                # include_target = Element("model", name = "goal"+str(i))
                # visual_target = Element("visual", name = "visual")
                # geometry = Element("geometry")
                # sphere = Element("sphere")
                # radius = Element("radius")
                # radius.text = "0.5"
                # sphere.append(radius)
                # geometry.append(sphere)
                # material = Element("material")
                # script = Element("script")
                # material_name = Element("name")
                # material_name.text = "Gazebo/Grey"
                # uri_material = Element("uri")
                # uri_material.text = "file://media/materials/scripts/gazebo.material"
                # script.append(material_name)
                # script.append(uri_material)
                # material.append(script)
                # visual_target.append(geometry)
                # visual_target.append(material)
                # pose_target = Element("pose")
                # pose_target.text = str(sx*2)+" "+str(sy*2)+" "+str(sz)+" 0 0.000143 -1.57317"
                # include_target.append(visual_target)
                # include_target.append(pose_target)
                # root[0].append(include_target)
        # plugin = Element("plugin", name="ros_interface_plugin", filename="librotors_gazebo_ros_interface_plugin.so")
        physics = self.physics.getroot()
        spherical_coordinates = self.spherical_coordinates.getroot()
        # root[0].append(plugin)
        root[0].append(spherical_coordinates)
        root[0].append(physics)
        self.tree.write(
            os.path.abspath("../world/multi_robots.world"),
            pretty_print=True, 
            xml_declaration=True, 
            encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multi-robot launch file for Gazebo")
    parser.add_argument(
        "--launch_file", default="../world/basic.world", type=str
    )
    parser.add_argument(
        "--robot_num", default=5, type=int
    )
    parser.add_argument(
        "--env_size", default=10.0, type=float
    )

    parser.add_argument(
        "--env_type", default="circle", type=str   # ball circle random
    )

    parser.add_argument(
        "--modelname", default="quadrotor", type=str   # ball circle random
    )
    args = parser.parse_args()

    creator = Creator(args.robot_num,
                      args.env_size,
                      args.env_type,
                      args.launch_file,
                      args.modelname)
    creator.run()
    
