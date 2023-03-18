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

from network_server.feature.BaseFeature import BaseFeature
from network_server.network.AttractorNetwork import AttractorNetwork
from network_server.operator.LogicalOperator import And
from network_server.operator.LogicalOperator import Or
from common_tools.ActivationFunction import *

"""
    GA: Subjects i, j share topology attention
"""
class GA(BaseFeature):
    def __init__(self, id_, p_):
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = AttractorNetwork(p_)
    def step(self, inp_, bel_=None):
        IA_agent = inp_[self._agentId]['IA']
        IA_other = inp_[self._otherId]['IA']        
        dot = self.saturate(self._ut.dotProduct(IA_agent, IA_other))
        self._state = self._net.step([dot],[1.0])
        return self._state

"""
    GB: Subjects i, j are looking at each other body
"""
class GB(BaseFeature):
    def __init__(self, id_, p_):        
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = And(p_)   
        self._afn = eval("{}(p_)".format(p_['afn']))
    def step(self, inp_, bel_=None):
        SB_agent = inp_[self._agentId]['SB'][self._otherId]
        SB_other = inp_[self._otherId]['SB'][self._agentId]
        self._state = self._afn.compute(self._net.step(SB_agent, SB_other))
        return self._state                        
    
"""
    GF: Subjects i, j are looking at each other face
"""
class GF(GB):
    def __init__(self, id_, p_):        
        GB.__init__(self, id_, p_)        
    def step(self, inp_, bel_=None):
        SF_agent = inp_[self._agentId]['SF'][self._otherId]
        SF_other = inp_[self._otherId]['SF'][self._agentId]
        self._state = self._afn.compute(self._net.step(SF_agent, SF_other))
        return self._state                        
        
"""
    GO: Subjects i, j are oriented toward each other
"""
class GO(GB):
    def __init__(self, id_, p_):        
        GB.__init__(self, id_, p_)        
    def step(self, inp_, bel_=None):
        SO_agent = inp_[self._agentId]['SO'][self._otherId]
        SO_other = inp_[self._otherId]['SO'][self._agentId]
        self._state = self._afn.compute(self._net.step(SO_agent, SO_other))
        return self._state                        
        
"""
    GC: Subjects i, j are are proximal to each other 
"""
class GC(GB):
    def __init__(self, id_, p_):        
        GB.__init__(self, id_, p_)        
        self._net = Or(p_)
    def step(self, inp_, bel_=None):
        SC_agent = inp_[self._agentId]['SC'][self._otherId]
        SC_other = inp_[self._otherId]['SC'][self._agentId]
        self._state = self._afn.compute(self._net.step(SC_agent, SC_other))
        return self._state                        

"""
    GT: Subjects i, j are in physical contact

"""
class GT(GC):
    def __init__(self, id_, p_):        
        GC.__init__(self, id_, p_)
    def step(self, inp_, bel_=None):
        ST_agent = inp_[self._agentId]['ST'][self._otherId]
        ST_other = inp_[self._otherId]['ST'][self._agentId]
        self._state = self._afn.compute(self._net.step(ST_agent, ST_other))

        return self._state                        

"""
    GS: Subjects i, j talk
"""
class GS(GC):
    def __init__(self, id_, p_):        
        GC.__init__(self, id_, p_)   
    def step(self, inp_, bel_=None):
        IS_agent = inp_[self._agentId]['IS']
        IS_other = inp_[self._otherId]['IS']
        self._state = self._afn.compute(self._net.step(IS_agent, IS_other))
        return self._state                        

