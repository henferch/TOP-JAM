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
from common_tools.ActivationFunction import *
from network_server.feature.BaseFeature import BaseFeature
from network_server.network.AttractorNetwork import AttractorNetwork
from network_server.operator.LogicalOperator import And
from network_server.operator.LogicalOperator import Or

class JointScale(BaseFeature):

    def __init__(self, id_, p_):
        self._id = id_
        self._agentId = p_['agentId']
        self._otherId = p_['otherId']
        self._typeId = p_["typeId"]
        p_['n'] = len(self._typeId)
        BaseFeature.__init__(self, id_, p_)
        self._firingRates = []
        
        # getting attractor net parameters         
        pA = {}        
        for pId in p_.keys():
            if not pId in ["inp","typeId"] :
                pA[pId] = p_[pId]            
                                
        pA.update({'ownerId':self._agentId, 'id':self._id, "inp": [{"id":"js", "n": len(self._typeId), "W": "gauss"}]})
        self._net = AttractorNetwork(pA)    
                
        # getting neuron unit fr parameters 
        inpList = p_['inp']
        
        for t in self._typeId:
            inps = []
            for inpT in inpList:
                if inpT['uId'] == t:
                    inps.append(inpT)
            pt = {'id': t, 'ut': p_['ut'], 'inp': inps, 'n': 1, "agentId": self._agentId, 'otherId':self._otherId}
            fr = eval("{}({}, pt)".format(t,t))
            self._firingRates.append(fr)
    def step(self, inp_, bel_=None):
        fr = []
        for neuron in self._firingRates:
            fr.append(neuron.step(self._state, inp_, bel_))
        self._state = self._net.stepFromFiringRate([np.hstack(fr)], [1.0])
        return self._state 
        

class BaseUnit():
    def __init__(self, id_, p_):
        self._id = id_
        self._ut = p_['ut']        
        self._opAnd = And(p_)
        self._opOr = Or(p_)
        self._inp = p_['inp']
        self._agentId = p_["agentId"]
        self._otherId = p_["otherId"]
    def step(self, st_, inp_, bel_=None):
        raise Exception("Unimplemented method")  
    def getMask(self, ids_):
        masks = []
        masksIds = []        
        for i in self._inp:                
            iId = i['id']
            if iId in ids_:
                masksIds.append(iId)
                m = self._ut.multiGaussian(np.array(i['ref'], dtype= np.float32), i['mu'], np.array(i['sig'], dtype= np.float32))
                m /= m.sum()
                masks.append(m)
        return masksIds, masks

"""
    Ind: Individual attention
"""
class Ind(BaseUnit):
    def __init__(self, id_, p_):
        BaseUnit.__init__(self, id_, p_)        
        
        self._masksId , self._masks = self.getMask(['GA','SB','GT'])        
    def step(self, st_, inp_, bel_=None):        
        andOps = []
        for inpId, mask in zip(self._masksId, self._masks):
            andOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask)) 
        self._state = np.array(self._opAnd.step(*tuple(andOps)), dtype=np.float32)
        return self._state

"""
    Mon: Monitoring attention
"""
class Mon(BaseUnit):
    def __init__(self, id_, p_):
        BaseUnit.__init__(self, id_, p_)        
        self._andMasksId , self._andMasks = self.getMask(['GB','GF'])
        self._orMasksId , self._orMasks = self.getMask(['GA','SB'])        
    def step(self, st_, inp_, bel_=None):
        orOps = []
        for inpId, mask in zip(self._orMasksId, self._orMasks):
            orOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        andOps = [self._opOr.step(*tuple(orOps))]
        for inpId, mask in zip(self._andMasksId, self._andMasks):
            andOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        self._state = np.array(self._opAnd.step(*tuple(andOps)), dtype=np.float32)
        return self._state
        
"""
    Com: Common Attention
"""
class Com(BaseUnit):
    def __init__(self, id_, p_):
        BaseUnit.__init__(self, id_, p_)
        self._andMasksId , self._andMasks = self.getMask(['GA','GS','GB','SB'])
        self._orMasksId , self._orMasks = self.getMask(['GO','GC'])        
    def step(self, st_, inp_, bel_=None):
        orOps = []
        for inpId, mask in zip(self._orMasksId, self._orMasks):
            orOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        andOps = [self._opOr.step(*tuple(orOps))]
        for inpId, mask in zip(self._andMasksId, self._andMasks):
            andOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        self._state = np.array(self._opAnd.step(*tuple(andOps)), dtype=np.float32)
        return self._state

"""
    Mu: Mutual Attention   
"""
class Mut(BaseUnit):
    def __init__(self, id_, p_):
        BaseUnit.__init__(self, id_, p_)                
        self._andMasksId , self._andMasks = self.getMask(['GA','GS'])
        self._orMasksId , self._orMasks = self.getMask(['GF','GT'])
    def step(self, st_, inp_, bel_=None):
        orOps = []
        for inpId, mask in zip(self._orMasksId, self._orMasks):
            orOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        andOps = [self._opOr.step(*tuple(orOps))]
        for inpId, mask in zip(self._andMasksId, self._andMasks):
            andOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        self._state = np.array(self._opAnd.step(*tuple(andOps)), dtype=np.float32)
        return self._state

"""
    Sha: Shared Attention
"""
class Sha(BaseUnit):
    def __init__(self, id_, p_):
        BaseUnit.__init__(self, id_, p_)
        self._andMasksId , self._andMasks = self.getMask(['GA','GS'])
        self._orMasksId , self._orMasks = self.getMask(['GF','GT'])
    def step(self, st_, inp_, bel_=None):
        orOps = []
        for inpId, mask in zip(self._orMasksId, self._orMasks):
            orOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        andOps = [self._opOr.step(*tuple(orOps))]
        for inpId, mask in zip(self._andMasksId, self._andMasks):
            andOps.append(self._ut.dotProduct(inp_[self._agentId][inpId][self._otherId], mask))
        self._state = np.array(self._opAnd.step(*tuple(andOps)), dtype=np.float32)
        return self._state
