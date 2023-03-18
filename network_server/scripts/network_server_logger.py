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
from network_server.msg import NetworkServerMsg
from network_server.srv import NetworkServerSrv
from common_tools.Log import Log
from common_tools.Utils import Utils
from common_tools.MsgConverter import NetworkServerMsg as NeConverter
from common_tools.MsgConverter import NetworkServerSrv as NeSrvConverter

ut = None
log = None
neState = None
metaData = None
neConverter = None
jsR1H1 = []
jsR1H2 = []
jsH1R1 = []
jsH1H2 = []
jsH2R1 = []
jsH2H1 = []
frameId = []

def callback(msg_):
    
    global metaData, neConverter, jsR1H1, jsR1H2, jsH1R1, jsH1H2, jsH2R1, jsH2H1, frameId    
    
    neState = neConverter.fromROSMsg(metaData, msg_)     
    JS = neState['J']    
    R1_JS = JS['R1']['JF']    
    jsR1H1.append(R1_JS['H1'])
    jsR1H2.append(R1_JS['H2'])
    
    H1_JS = JS['H1']['JF']
    jsH1R1.append(H1_JS['R1'])
    jsH1H2.append(H1_JS['H2'])
    
    H2_JS = JS['H2']['JF']
    jsH2R1.append(H2_JS['R1'])
    jsH2H1.append(H2_JS['H1'])

    frameId. append(msg_.header.frame_id)
    
    print ("received message ID): {}".format(msg_.header.frame_id))

    
def getMetaData():
    rospy.wait_for_service('get_network_server_meta_data')        
    metaData = None
    try:
        neSrv = rospy.ServiceProxy('get_network_server_meta_data', NetworkServerSrv)
        resp = neSrv("")
        srvConv = NeSrvConverter()
        metaData = srvConv.fromROSMsg(resp)        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
    return metaData
    
def network_server_Node():    
    
    global metaData, neConverter, jsR1H1, jsR1H2, jsH1R1, jsH1H2, jsH2R1, jsH2H1, frameId    
    
    neConverter = NeConverter()        
    metaData = getMetaData()    
    
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('network_server_logger', anonymous=True)    
    rospy.Subscriber("/ANITI_LORIA/network_server_state", NetworkServerMsg, callback)    
                
    rospy.loginfo("network_server_logger running ...")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    rospack = rospkg.RosPack()
    packDir = rospack.get_path('network_server') + "/log/"
    
    np.save(packDir+"jsR1H1.npy", np.vstack(jsR1H1))
    np.save(packDir+"jsR1H2.npy", np.vstack(jsR1H2))
    np.save(packDir+"jsH1R1.npy", np.vstack(jsH1R1))
    np.save(packDir+"jsH1H2.npy", np.vstack(jsH1H2))
    np.save(packDir+"jsH2R1.npy", np.vstack(jsH2R1))
    np.save(packDir+"jsH2H1.npy", np.vstack(jsH2H1))
    np.save(packDir+"frameId.npy", np.vstack(frameId))

    print("A total of {} messages recorded".format(len(frameId)))

if __name__ == '__main__':
    network_server_Node()
    
