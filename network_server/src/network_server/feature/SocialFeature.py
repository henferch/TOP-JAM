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

"""
SF: Subject i looks at subject’s j face
"""
class SF(BaseFeature):
    def __init__(self, id_, p_):
        BaseFeature.__init__(self, id_, p_)        
        p_.update({'ownerId':self._agentId, 'id':self._id})                
        self._net = AttractorNetwork(p_)   
    def step(self, inp_, bel_=None):
        a = inp_[self._agentId]
        o = inp_[self._otherId]
        a_hp = a['hp']
        a_hd = a['hd']
        o_hp = o['hp']                
        p = np.min([a_hp['p'][0], a_hd['p'][0], o_hp['p'][0]])        
        dot = self.saturate(self._ut.dotProduct(np.array(o_hp['d']) - np.array(a_hp['d']), np.array(a_hd['d'])))
        self._state = self._net.step([dot], [p])                
        return self._state               
        
"""
SR: Subject i looks at subject’s j right hand         
"""
class SR(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)                      
    def step(self, inp_, bel_=None):
        a = inp_[self._agentId]
        o = inp_[self._otherId]
        a_hp = a['hp']
        a_hd = a['hd']
        o_rp = o['rp']
        p = np.min([a_hp['p'][0], a_hd['p'][0], o_rp['p'][0]])                        
        dot = self.saturate(self._ut.dotProduct(np.array(o_rp['d']-np.array(a_hp['d'])), np.array(a_hd['d'])))
        self._state = self._net.step([dot], [p])        
        return self._state

"""
SL: Subject i looks at subject’s j left hand
"""
class SL(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)                      
    def step(self, inp_, bel_=None):
        a = inp_[self._agentId]
        o = inp_[self._otherId]
        a_hp = a['hp']
        a_hd = a['hd']
        o_lp = o['lp']
        p = np.min([a_hp['p'][0], a_hd['p'][0], o_lp['p'][0]])                
        dot = self.saturate(self._ut.dotProduct(np.array(o_lp['d']-np.array(a_hp['d'])), np.array(a_hd['d'])))
        self._state = self._net.step([dot], [p])
        return self._state

"""
SB: Subject i looks at subject’s j body 
"""
class SB(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)                      
    def step(self, inp_, bel_=None):                    
        a = inp_[self._agentId]
        o = inp_[self._otherId]
        a_hp = a['hp'] 
        a_hp_d = np.array(a_hp['d']) 
        a_hd = a['hd']         
        a_hd_d = np.array(a_hd['d'])         
        o_hp = o['hp']
        o_tp = o['tp']
        o_lp = o['lp']
        o_rp = o['rp']
        pos = [np.array(o_tp['d'])]        
        p = np.min([a_hp['p'][0], a_hd['p'][0], o_hp['p'][0], o_tp['p'][0], o_lp['p'][0], o_rp['p'][0]])
        maxDot = 1.0e-20
      
        for d in pos:        
            dot = self.saturate(self._ut.dotProduct(d-a_hp_d,a_hd_d))        
            if dot > maxDot:
                maxDot = dot            
        self._state = self._net.step([maxDot], [p])        
        return self._state

"""
SC: Subject i is close to subject j       
"""
class SC(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)                      
        self._maxDist = p_['maxDist']
    def step(self, inp_, bel_=None):                    
        a = inp_[self._agentId]
        o = inp_[self._otherId]                    
        a_tp = a['tp'] 
        a_rp = a['rp'] 
        a_lp = a['lp'] 
        a_hp = a['hp']
        
        o_tp = o['tp'] 
        o_rp = o['rp'] 
        o_lp = o['lp'] 
        o_hp = o['hp']
        
        pa = [np.array(a_tp['d']), np.array(a_rp['d']), np.array(a_lp['d']), np.array(a_hp['d'])]
        po = [np.array(o_tp['d']), np.array(o_rp['d']), np.array(o_lp['d']), np.array(o_hp['d'])]            
        p = np.min([a_tp['p'][0], a_rp['p'][0], a_lp['p'][0], a_hp['p'][0], o_tp['p'][0], o_rp['p'][0], o_lp['p'][0], o_hp['p'][0]])
        mindist = self._maxDist
        for a in pa:
            for o in po:  
                norm = np.linalg.norm(a-o)
                if norm < mindist :
                    mindist = norm
        self._state = self._net.step([1.0 - mindist/self._maxDist], [p])        
        return self._state

"""
SO: Subject i body is oriented toward subject j body     
"""
class SO(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)    
    def step(self, inp_, bel_=None):                      
        a = inp_[self._agentId]
        o = inp_[self._otherId]                    
        a_tp = a['tp'] 
        a_to = a['to'] 
        o_tp = o['tp'] 
        p = np.min([a_tp['p'][0], a_to['p'][0], o_tp['p'][0]])    
        aTorOr = self._ut.get3DOrientationfromYawPitch(a_to['d'], 0.0)                       
        dot = self.saturate(self._ut.dotProduct(np.array(o_tp['d'])-np.array(a_tp['d']), aTorOr))
        self._state = self._net.step([dot], [p])
        return self._state

"""
ST: Subject i touches subject j
"""
class ST(SF):
    def __init__(self, id_, p_):
        SF.__init__(self, id_, p_)                        
        self._maxDist = p_['maxDist']
    def step(self, inp_, bel_=None):
        a = inp_[self._agentId]
        o = inp_[self._otherId]                    
        a_rp = a['rp'] 
        a_lp = a['lp'] 
        
        o_tp = o['tp'] 
        o_rp = o['rp'] 
        o_lp = o['lp'] 
        o_hp = o['hp']

        pa = [np.array(a_rp['d']), np.array(a_lp['d'])]
        po = [np.array(o_tp['d']), np.array(o_rp['d']), np.array(o_lp['d']), np.array(o_hp['d'])]            
        p = np.min([a_rp['p'][0], a_lp['p'][0], o_tp['p'][0], o_rp['p'][0], o_lp['p'][0], o_hp['p'][0]])
        mindist = self._maxDist
        for a in pa:
            for o in po:
                norm = np.linalg.norm(a-o)
                if norm < mindist :
                    mindist = norm
        self._state = self._net.step([1.0 - mindist/self._maxDist], [p])        
        return self._state
