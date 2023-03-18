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

from common_tools.Utils import Utils
from common_tools.PlaneTopology import PlaneTopology
from common_tools.ActivationFunction import *
from network_server.network.AttractorNetwork import AttractorNetwork 
from network_server.network.AssociativeNetwork import AssociativeNetwork
from network_server.agent.Agent import Agent
from data_server.DataServer import DataServer
import rospkg
import numpy as np
from collections import deque
import copy
import pprint


class AgentServer:
            
    __instance = None
    
    @staticmethod 
    def getInstance():
        
       """ Static access method. """
       if AgentServer.__instance == None:
          AgentServer()
       return AgentServer.__instance
   
    
    def __init__(self):
        
        """ Virtually private constructor. """
        if AgentServer.__instance != None:
           raise Exception("This class is a singleton class, use 'getInstance()' instead!")
        else:                                
            self._agents = None  
            self._buffer = None            
            self._bufferSize = None
            self._obsPeriodInMs = None
            self._believe = None
            AgentServer.__instance = self
            self.initialize()
        
    def initialize(self):
                
        ds = DataServer.getInstance()
    
        self._bufferSize = ds.getBufferSize()        
        self._obsPeriodInMs = ds.getObsPeriodInMs()
        self._agentsId = ds.getAgentsId()        
        self._propertiesId = ds.getPropertiesId()        
        self._objectsId = ds.getOjectsId()
        self._objectsProperties = ds.getOjectsProperties()
        self._observationsId = ds.getObservationsId()
        
        self._ut = Utils.getInstance() 
        
        rospack = rospkg.RosPack()
        packDir = rospack.get_path('network_server')
    
    
        param = self._ut.getParameters(packDir + '/parameters/' + self.__class__.__name__ + '.json')
        if param is None:
            raise Exception("please provide a valid '{}.json' file".format(self.__class__.__name__))        
        
        p = {"bufferSize": self._bufferSize,\
             "obsPeriodInMs": self._obsPeriodInMs, 
             "objectsId" : self._objectsId, 
             "objectsProperties":  self._objectsProperties,            
             "propertiesId" : self._propertiesId, 
             "agentsId" : self._agentsId, 
             "observationsId":self._observationsId}
        
        param.update(p) 
        param.update({'ut': self._ut})                          

        self._NProperties = len(self._propertiesId)
        self._objectPropertyAssocProperties = param["objectPropertyAssoc"]
        self._objectsPropertiesVector = {}        
        self._objectsTopologyNet = {}
        self._objectsPropertiesNet = {}
                
        
        #obsParam = param['observationsId']
        self._agentObsId = self._observationsId["agent"]
        self._propertyObsId = self._observationsId["property"]
        self._objectObsId = self._observationsId["object"]                
        
        self._agents = {}
        self._buffer = deque(maxlen=self._bufferSize)
        
        #initializing topologies
        topParam = param['topologyAttractor']
        topParam['dt'] = self._obsPeriodInMs
         
        self._topology = PlaneTopology({'data' : np.vstack(topParam['ref']), 'offset' : topParam['offset'], 'res' : topParam['res']})
        self._topologyDimensions = self._topology.getGridDimensions()
        print("self._topologyDimensions: ", self._topologyDimensions)
        
        ref = self._topology.getGrid()
        nRef = len(ref)
        nProp = len(self._propertiesId)
        topParam['ref'] = ref
        
        opInpPar = self._objectPropertyAssocProperties['inp']
        for ip in opInpPar:
            if ip['n'] == "topology":
                ip['n'] = nRef
                ip['id'] = 'op'
            elif ip['n'] == 'properties':
                ip['n'] = nProp
                ip['id'] = 'pp'
            
        for oId in self._objectsId:            
            objProps = self._objectsProperties[oId] 
            objPropVec = np.zeros((self._NProperties,), dtype=np.float32)
            i = 0
            for pId in self._propertiesId:
                if pId in objProps :
                    objPropVec[i] = 1.0
                i += 1

            self._objectsPropertiesVector[oId] = objPropVec

            topParam.update({'ownerId':"AgentServer", 'id':oId})
            self._objectsTopologyNet[oId] = AttractorNetwork(topParam)
            p = copy.copy(self._objectPropertyAssocProperties)
            p.update({'ownerId':"AgentServer", 'id':oId})
            self._objectsPropertiesNet[oId] = AssociativeNetwork(p)
            
        # create agents 
        param.update({'topology':self._topology, 'sigTop': topParam['sig']})
        metaJSIds = None
        for aId in self._agentsId:   
            a = Agent(aId, param)
            self._agents[aId] = a
            if metaJSIds is None:
                metaJSIds = a.getJointScaleIds()
            
        
        self._metaData = {}
        for aId in self._agentsId :
            self._metaData[aId] = self._agents[aId].getStateDim()
        self._metaData["topologyDimensions"] = self.getTopologyDimensions()            
        
        self._metaData["agentsId"] = self._agentsId
        self._metaData["jointScaleIds"] = metaJSIds                                    
        
    def getMetaData(self):
        return self._metaData
    
    def getTopologyDimensions(self):
        return self._topologyDimensions
    
    def getAgentsId(self):
        return copy.copy(self._agentsId)
    
        
    def step(self, state_):
        
        stP = state_ # Primery observations State
        
        # step object topologies
        obsObjTop = {}
        obsObjTopProp = {}
        for oId, net in self._objectsTopologyNet.items():
            obs = state_[oId]['op']
            op = self._topology.project(obs['d'])
            if not op is None:
                op = op[0:2]
            objTop = net.step([op], [obs['p']])
            obsObjTop[oId] = objTop
            obsObjTopProp[oId] = self._objectsPropertiesNet[oId].step(self._objectsPropertiesVector[oId], objTop)
        
        stP.update({'objTop':obsObjTop, 'objTopProp': obsObjTopProp})
        
        stI = {} # individual features state           
        stS = {} # social features state                
        stG = {} # group features state                
        stJ = {} # joint features state                
        
        for aId, a in self._agents.items():                              
            stI[aId] = a.stepIndividualFeatures(copy.deepcopy(stP))
            
        for aId, a in self._agents.items():                              
            stS[aId] = a.stepSocialFeatures(copy.deepcopy(stP))
        
        stIc = copy.copy(stI)
        stSc = copy.copy(stS)
        
        # clonning states  from features I and S
        for (k1, v1), (k2, v2) in zip(stSc.items(), stIc.items()):    
            for k, v in v2.items():
                v1[k] = v
                
        for aId, a in self._agents.items():                                          
            stG[aId] = a.stepGroupFeatures(copy.deepcopy(stSc))

        stGc = copy.copy(stG)                    
        for (k1, v1), (k2, v2) in zip(stGc.items(), stSc.items()):    
            for k, v in v2.items():
                v1[k] = v
        
        for aId, a in self._agents.items():                                          
            stJ[aId] = a.stepJointFeatures(copy.deepcopy(stGc))
                    
        self._buffer.append({'I': stI, 'S': stS, 'G': stG, 'J': stJ})
    
    def getState(self):
        return copy.deepcopy(self._buffer[-1])
    
    def getObjPropKeys(self):
        return self._objectsId, self._propertiesId
        
        
    def getData(self, agentI_, agentJ_, featureType_, featureId_):                
                        
        if featureType_ == 'I':
            return copy.copy(self._buffer[agentI_]['I'][featureId_])
        
        elif featureType_ == 'S':
            return copy.copy(self._buffer[agentI_]['S'][agentJ_][featureId_])
        
        elif featureType_ == 'G':
            return copy.copy(self._buffer[agentI_]['G'][agentJ_][featureId_])
        
        elif featureType_ == 'J':
            return copy.copy(self._buffer[agentI_]['J'][agentJ_][featureId_])        
        else:
            raise Exception("Unknown feature type '{}'".format(featureType_))
                