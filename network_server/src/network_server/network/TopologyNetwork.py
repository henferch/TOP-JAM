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

from common_tools.ActivationFunction import ActivationNormalize
from network_server.network.layer.AssociativeLayer import AssociativeLayer
from network_server.network.AttractorNetwork import AttractorNetwork
import numpy as np

class TopologyNetwork(AttractorNetwork):
    def __init__(self, ownerId_, id_, ref_, sigma_, inh_, dt_, tau_, afn_, inputs_):        
        AttractorNetwork.__init__(self, ownerId_, id_, ref_, sigma_, inh_, dt_, tau_, afn_, inputs_)     
        self._inputAsLayer = []        
        self._inputLr = []                
        for i in inputs_:            
            iid = i['id']
            n = i['n']                        
            lr = i['lr']                
            learn = i['learn']
            self._inputLr.append(lr)            
            al = None
            if learn:
                al = AssociativeLayer(self._ownerId, self._id, "(topology,{})".format(iid), (self._N, n), lr, ActivationNormalize())
            self._inputAsLayer.append(al)            
                    
    def getAssociation(self, id_=None):        
        Wa = []
        if not id_ is None:            
            for i in id_:
                al = self._inputAsLayer[self._inputId[i]]
                Wali = None
                if not al is None:
                    Wali = al.getW() 
                Wa.append(Wali)
        else:
            for aL in self._inputAsLayer:
                Wa.append(aL.getW())                
        return Wa
        
    def step(self, input_, inputGain_, bel_=None, belGain_=None):        
        his = []        

        for lFF, i in zip(self._inputFFLayer, input_):
            his.append(lFF.step(np.array(i), 1.0))         

        h = self._attractor.step(his, inputGain_, bel_, belGain_)                
        for lAs in self._inputAsLayer:                        
            if not (lAs is None):
                lAs.step(h, input_[lAs.getId()])
    
    def getState(self):
        iFFState = []        
        iAsState = []        
        for lFF, lAs in zip(self._inputAtLayer, self._inputAsLayer):
            hFF, uFF = lFF.getState()
            hAs = None
            uAs = None
            if not lAs is None:
                hAs, uAs = lAs.getState()
            iFFState.append({'id':lFF.getId(), 'h': hFF, 'u':uFF})    
            iAsState.append({'id':lAs.getId(), 'h': hAs, 'u':uAs})    
                    
        ya, ga, ba, ha, ua = self._attractor.getState()
        return {'id': self._id, 'y':self._y,\
                'attractor': {'id':self._attId, 'y':ya, 'g':ga, 'b':ba, 'h':ha, 'u':ua},
                'input': iFFState, 'association': iAsState}        
    

        
        
            
        
            
        
        