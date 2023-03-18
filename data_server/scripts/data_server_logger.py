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
import rospkg
import numpy as np
from data_server.msg import DataServerMsg
from data_server.srv import DataServerSrv
from common_tools.Log import Log
from common_tools.Utils import Utils
from common_tools.MsgConverter import DataServerMsg as DsConverter

ut = None
log = None
dsState = None
dsConverter = None
dsMetaData = None
agentHd = []
agentHp = []
objectPo = []
frameId = []

def callback(msg_):
    global dsMetaData, dsConverter, agentHd, agentHp, frameId, objectPo, ut, log
    
    dsState = dsConverter.fromROSMsg(dsMetaData, msg_) 
    
    l_agentHd = []
    l_agentHp = []
    l_objectPo = []
    
    for aId in dsMetaData['agentsId']:
        aData = dsState[aId]    
        l_agentHd.append(aData['hd']['d'])
        l_agentHp.append(aData['hp']['d'])
    agentHd.append(np.hstack(l_agentHd))
    agentHp.append(np.hstack(l_agentHp))
    
    for oId in dsMetaData['objectsId']:
        l_objectPo.append(dsState[oId]['op']['d'])
    
    objectPo.append(np.hstack(l_objectPo))
    
    frameId. append(msg_.header.frame_id)
        
    print ("received message ID): {}".format(msg_.header.frame_id))


    
def data_server_logger():
    global dsMetaData, dsConverter, agentHd, agentHp, frameId, objectPo, ut, log

    ut = Utils.getInstance() 
    log = Log.getInstance()  #    
    dsConverter = DsConverter()
    
    rospy.wait_for_service('get_data_server_meta_data')
    try:
        dsSrv = rospy.ServiceProxy('get_data_server_meta_data', DataServerSrv)
        resp = dsSrv("")
        dsMetaData =  {"propertiesId" : resp.propertiesId, "objectsId" : resp.objectsId, "agentsId" : resp.agentsId}
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
            
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('data_server_logger', anonymous=True)
    rospy.Subscriber("/ANITI_LORIA/data_server_Ob", DataServerMsg, callback)
        
    rospy.loginfo("data_server_logger running ...")


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    rospack = rospkg.RosPack()
    packDir = rospack.get_path('data_server') + "/log/"
    
    np.save(packDir+"agentHd.npy", np.vstack(agentHd))
    np.save(packDir+"agentHp.npy", np.vstack(agentHp))
    np.save(packDir+"objectPo.npy", np.vstack(objectPo))
    np.save(packDir+"frameId.npy", np.vstack(frameId))    
    
    print("A total of {} messages recorded".format(len(frameId)))

if __name__ == '__main__':
    data_server_logger()
    
