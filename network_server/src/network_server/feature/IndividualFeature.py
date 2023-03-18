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

import numpy as np
from network_server.feature.BaseFeature import BaseFeature
from network_server.network.AttractorNetwork import AttractorNetwork
import pprint


"""
    IA: Subject i attends to a given topology region
"""
class IA(BaseFeature):
    def __init__(self, id_, p_):
        BaseFeature.__init__(self, id_, p_)
        self._top = p_['top']        
        p_.update({'ownerId':self._agentId, 'id':self._id})
        self._net = AttractorNetwork(p_)     
    def step(self, data_, bel_=None):                
        dA = data_[self._agentId]
        dA_hd = dA['hd']
        dA_hp = dA['hp']        
        ld = self._top.intersect(dA_hp['d'], dA_hd['d'])       
        
        if not ld is None:
            ld = ld[0:2]    
        ldProb = np.min([dA_hd['p'][0], dA_hp['p'][0]])        
        self._state = self._net.step([ld], [ldProb])
        return self._state

"""
    IO: Subject i attends to object o   
"""
class IO(BaseFeature):
    def __init__(self, id_, p_):
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = AttractorNetwork(p_)   
    def step(self, inp_, bel_=None):         
        obsTop = inp_['objTop'][self._otherId]
        IA_agent = inp_[self._agentId]['IA']
        dot = self.saturate(self._ut.dotProduct(obsTop, IA_agent))
        self._state = self._net.step([dot], [1.0])                
        return self._state


"""
    IP: Subject i attends to property p   
"""
class IP(BaseFeature):
    def __init__(self, id_, p_):                        
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = AttractorNetwork(p_)   
    def step(self, inp_, bel_=None):                 
        x = inp_['pv']
        IA_agent = inp_[self._agentId]['IA']
        maxDot = self._minRef
        for oId, W in inp_['objTopProp'].items():
            ip = np.dot(x, W)
            if ip.sum() < 1.0e-10:
                continue
            dot = self.saturate(self._ut.dotProduct(ip,IA_agent))
            if dot > maxDot:
                maxDot = dot
        self._state = self._net.step([self.saturate(maxDot)], [1.0])        
        return self._state
        
        
"""
    IS: Subjects i speaks
"""
class IS(IO):
    def __init__(self, id_, p_):
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = AttractorNetwork(p_)   
    def step(self, inp_, bel_=None):        
        sp = inp_[self._agentId]['sp']
        v = sp['d']
        self._state = self._net.step([self.saturate(v)], sp['p'])
        return self._state
        
    