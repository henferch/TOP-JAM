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
from common_tools.Log import Log
from data_server.SimulatorProvider import SimulatorProvider 
from data_server.MocapProvider import MocapProvider
from collections import deque
import rospkg

class DataServer():
        
    __instance = None
    
    @staticmethod 
    def getInstance():
       """ Static access method. """
       if DataServer.__instance == None:
          DataServer()
       return DataServer.__instance
    
    def __init__(self):
        """ Virtually private constructor. """
        if DataServer.__instance != None:
           raise Exception("This class is a singleton class, use 'getInstance()' instead!")
        else:          
                        
            super(DataServer, self).__init__()
            self._ut = Utils.getInstance()            
            self._log = Log.getInstance()            
            self._agentServer = None
            self._obsBuffer = None            
            self._bufferSize = None
            self._lastTime = None                       
            self._obsPeriodInMs = None
            self._agentsId = None
            self._propertiesId = None
            self._objectsId = None            
            self._meta = None            
            self._state = None
            self._run = False
            self._provider = None
            self._ref = None
            DataServer.__instance = self
            self.initialize()
        
    def initialize(self):
        rospack = rospkg.RosPack()
        packDir = rospack.get_path('data_server') 
        self._step = 0
        self._lastTime = self._ut.getCurrentTimeMS()                
        param = self._ut.getParameters(packDir + '/parameters/' +self.__class__.__name__ + '.json')        
        if param is None:
            raise Exception("please provide a valid 'parameters/DataServer.json' file")
        self._bufferSize = param['bufferSize']        
        self._obsBuffer = deque(maxlen=self._bufferSize)   
        self._obsPeriodInMs = param['obsPeriodInMs']       
        self._agentsId = param['agentsId']                    
        self._propertiesId  = param['propertiesId']          
        self._objectsId  = param['objectsId']               
        self._meta = {"obsPeriodInMs": self._obsPeriodInMs, "agentsId" : self._agentsId ,  "propertiesId" : self._propertiesId, "objectsId" : self._objectsId}
        self._observationsId = param["observationsId"]
        self._objectsProperties = param["objectsProperties"]
        param.update({'packDir':packDir})                
        self._provider = eval("{}(param)".format(param["providerClass"]))       
        self._state, self._stateObs = self._provider.step()            
        self._run = True
                
    def getMetaData(self): 
        return self._meta
    
    def isSimulation(self):
        if self._provider is None:
            raise Exception("The class instance has not been initialized!")
        return type(self._provider) == SimulatorProvider 
        
    def step(self):
        self._state, self._stateObs = self._provider.step()
        self._obsBuffer.append(self._stateObs)
        self._lastTime = self._ut.getCurrentTimeMS()
    
    def getBufferSize(self):
        return self._bufferSize
    
    def getObsPeriodInMs(self):
        return self._obsPeriodInMs
    
    def getAgentsId(self):
        return self._agentsId
    
    def getObservationsId(self):
         return self._observationsId 
    
    def getPropertiesId(self):
        return self._propertiesId
    
    def getOjectsId(self):
        return self._objectsId 
    
    def getOjectsProperties(self):
        return self._objectsProperties

    def getState(self):
        return self._state, self._stateObs
        
    def getObsBuffer(self):
        self._obsBuffer
                   
            
        
