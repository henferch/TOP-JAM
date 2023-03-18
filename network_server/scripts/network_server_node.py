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
from std_msgs.msg import Header
from data_server.msg import DataServerMsg
from data_server.srv import DataServerSrv
from network_server.msg import NetworkServerMsg
from network_server.agent.AgentServer import AgentServer
from common_tools.Log import Log
from common_tools.Utils import Utils
from common_tools.MsgConverter import DataServerMsg as DsConverter
from common_tools.MsgConverter import NetworkServerMsg as NeConverter
import copy 

ut = None
log = None
_as = None
dsState = None
neState = None
dsConverter = None
neConverter = None
pubNe = None
dsMetaData = None
neMetaData = None

def callback(msg_):
    global _as, dsMetaData, dsConverter, neConverter, nePub, ut, log
    dsState = dsConverter.fromROSMsg(dsMetaData, msg_) 
    
    t1 = ut.getCurrentTimeMS()    
    _as.step(dsState)
    t2 = ut.getCurrentTimeMS()    
    log.info("Processing time in ms : {} (FrameId : {})".format(t2-t1, msg_.header.frame_id))
    neState = _as.getState()
    
    neMsg = NetworkServerMsg()
    neMsg.header = copy.deepcopy(msg_.header)
    neConverter.toROSMsg(neMetaData, neState, neMsg)    
    nePub.publish(neMsg)   
    

    
def network_server_Node():
    global _as, dsMetaData, neMetaData, dsConverter, neConverter, nePub, ut, log

    ut = Utils.getInstance() 
    log = Log.getInstance()  
    dsConverter = DsConverter()
    neConverter = NeConverter()        
    
    rospy.wait_for_service('get_data_server_meta_data')
    try:
        dsSrv = rospy.ServiceProxy('get_data_server_meta_data', DataServerSrv)
        resp = dsSrv("")
        dsMetaData =  {"obsPeriodInMs" : resp.obsPeriodInMs, "propertiesId" : resp.propertiesId, "objectsId" : resp.objectsId, "agentsId" : resp.agentsId}
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        
        
    _as = AgentServer.getInstance()
    neMetaData = _as.getMetaData()
    
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('network_server_node', anonymous=True)
    rospy.Subscriber("/ANITI_LORIA/data_server_Ob", DataServerMsg, callback)
    
    nePub = rospy.Publisher('/ANITI_LORIA/network_server_state', NetworkServerMsg, queue_size=10)
    
    rospy.loginfo("network_server_node running ...")


    freq = int(1000.0/(dsMetaData["obsPeriodInMs"]*1.0))
    rate = rospy.Rate(freq) # something around 50hz

    while not rospy.is_shutdown():
        rate.sleep()
    

if __name__ == '__main__':
    network_server_Node()
    
