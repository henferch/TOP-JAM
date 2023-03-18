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

from network_server.network.BaseNetwork import BaseNetwork
from network_server.network.layer.AttractorLayer import AttractorLayer
from common_tools.ActivationFunction import *
import numpy as np
import copy
import pprint

class AttractorNetwork(BaseNetwork):
    
    def __init__(self, p_ = None):        
        BaseNetwork.__init__(self, p_)
        self._ref = np.array(p_['ref'])
        self._N = len(self._ref)                        
        self._y = np.zeros((self._N,),dtype=np.float32) 
        inp = p_['inp']
        self._layerId = "attractor"
        if inp == None or len(inp) < 1 :
            raise Exception("please provide the description of at least 1 inputs in the json file for the network [{}]'".format(self.__class__.__name__, self._id))
                
        p = copy.copy(p_)
        p.update({'n':self._N, 'netId':self._id, "layerId":self._layerId})
        self._attractor = AttractorLayer(p)
        
    def stepFromFiringRate(self, input_, inputGain_, bel_=None, belGain_=None):
        self._y = self._attractor.stepFromFiringRate(input_, inputGain_, bel_, belGain_)        
        return copy.copy(self._y)
    
    def step(self, input_, inputGain_, bel_=None, belGain_=None):        
        self._y = self._attractor.step(input_, inputGain_, bel_, belGain_)        
        return copy.copy(self._y)
    
    def getState(self):        
        ya, ga, ba, ha, ua = self._attractor.getState()
        return {'id': self._id, 'y':self._y, self._layerId: {'y':ya, 'g':ga, 'b':ba, 'h':ha, 'u':ua}}        
    
            
        
        
            
        
            
        
        
