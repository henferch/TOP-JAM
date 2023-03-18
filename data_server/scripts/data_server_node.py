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
from data_server.DataServer import DataServer
from common_tools.MsgConverter import DataServerMsg as DsConverter

def data_server_Node():
    
    pubGT = rospy.Publisher('/ANITI_LORIA/data_server_GT', DataServerMsg, queue_size=10)
    pubOb = rospy.Publisher('/ANITI_LORIA/data_server_Ob', DataServerMsg, queue_size=10)
        
    ds = DataServer.getInstance()
    dsMetaData = ds.getMetaData()
    converter = DsConverter()
    
    rospy.init_node('data_server_node', anonymous=True)
    
    freq = int(1000.0/(dsMetaData["obsPeriodInMs"]*1.0))
    rate = rospy.Rate(freq) # 20hz

    rospy.loginfo("data_server_node running ...")
    
    frameId = 0
    while not rospy.is_shutdown():
    
        ds.step()
        stGT, stOb  = ds.getState()
        
        frameIdStr = "{}".format(frameId)
        t = rospy.Time.now()
        
        hGT = Header() 
        hGT.frame_id = frameIdStr
        hGT.stamp = t
        msgGT = DataServerMsg()
        msgGT.header = hGT
    
        converter.toROSMsg(dsMetaData, stGT, msgGT)    

        hOb = Header()
        hOb.frame_id = frameIdStr
        hOb.stamp = t
        msgOb = DataServerMsg()
        msgOb.header = hOb
        
        converter.toROSMsg(dsMetaData, stOb, msgOb)
                
        pubGT.publish(msgGT)
        pubOb.publish(msgOb)
        
        frameId += 1
        rate.sleep()
        

if __name__ == '__main__':
    try:
        data_server_Node()
    except rospy.ROSInterruptException:
        pass
