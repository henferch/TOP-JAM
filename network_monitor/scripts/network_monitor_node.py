#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BSD 3-Clause License

Copyright (c) 2023, Hendry Ferreira Chame

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Hendry Ferreira Chame <hendry.ferreira-chame@loria.fr>

Publication:
    
"TOP-JAM: A bio-inspired topology-based model of joint attention for human-robot interaction"
Hendry Ferreira Chame, Aurélie Clodic and Rachid Alami
   
This work has been partially funded by the Agence Nationale de la Recherche 
grant ANITI ANR-19-PI3A-0004.
The work was conducted at :
- Team Robotics and InteractionS (RIS), LAAS-CNRS. 
  Université de Toulouse, CNRS, Toulouse, France.
- Team NeuroRhythms, LORIA-CNRS. 
  Campus Scientifique, 615 Rue du Jardin-Botanique, 
  54506 Vandœuvre-lès-Nancy, France.
  
"""

import rospy
from network_server.msg import NetworkServerMsg
from data_server.srv import DataServerSrv
from network_server.srv import NetworkServerSrv
from common_tools.MsgConverter import NetworkServerMsg as NeConverter
from common_tools.MsgConverter import NetworkServerSrv as NeSrvConverter
from network_monitor.GUI import Main

neState = None
metaData = None
neConverter = None
gui = None


def callback(msg_):
    global metaData, neState, neConverter, gui
    neState = neConverter.fromROSMsg(metaData, msg_) 
    gui.setState(neState)
    

def launchGUI(metadata):
    return      Main.getInstance(metaData)
    
def getMetaData():
    rospy.wait_for_service('get_network_server_meta_data')        
    try:
        neSrv = rospy.ServiceProxy('get_network_server_meta_data', NetworkServerSrv)
        resp = neSrv("")
        srvConv = NeSrvConverter()
        metaData = srvConv.fromROSMsg(resp)        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
   
    rospy.wait_for_service('get_data_server_meta_data')
    
    try:
        dsSrv = rospy.ServiceProxy('get_data_server_meta_data', DataServerSrv)
        resp = dsSrv("")
        metaData["propertiesId"] = resp.propertiesId
        metaData["objectsId"] = resp.objectsId
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)        
    return metaData


def network_monitor_node():
    global metaData, neConverter, gui
    rospy.init_node('network_monitor_node', anonymous=True)
    gui = Main.getInstance()
    neConverter = NeConverter()        
    metaData = getMetaData()    
    
        
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.Subscriber("/ANITI_LORIA/network_server_state", NetworkServerMsg, callback)
    
    gui.initialize(metaData)

if __name__ == '__main__':
        
    network_monitor_node()
    
        
    
