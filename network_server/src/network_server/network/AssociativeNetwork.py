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
import copy
from network_server.network.BaseNetwork import BaseNetwork
from network_server.network.layer.AssociativeLayer import AssociativeLayer

class AssociativeNetwork(BaseNetwork):
    #def __init__(self, ownerId_, id_, lr_, afn_, inputs_):   
    def __init__(self, p_ = None):   
        BaseNetwork.__init__(self, p_)
        inp = p_['inp']
        if not len(inp) == 2:
            raise Exception('AssociativeNetwork requires a list of inputs !')
        self._lr = p_['lr']
        
        i = inp[0]
        j = inp[1]
        self._id_i = i['id'].lower()
        self._id_j = j['id'].lower()
        si = i['n']
        sj = j['n']
        
        self._size = (si,sj)
        self._N = si*sj
        self._y = np.zeros((self._N,), dtype=np.float32)        
        
        self._assIdij = "({},{})".format(i['id'],j['id'])
        p_['netId'] = p_['id']
        p_['layerId'] = self._assIdij
        p_['size'] = self._size
        self._associativeLayer = AssociativeLayer(p_)
                    
    def getAssociation(self):        
        return copy.copy(self._associativeLayer.getW())
        
        
    def step(self, i_, j_, bel_=None, belGain_=None):                
        self._y = self._associativeLayer.step(i_, j_, bel_, belGain_)        
        return copy.copy(self._y) 
        
    def getState(self):
        yi, gi, bi, hi, ui = self._inputLayer_i.getState()    
        yj, gj, bj, hj, uj = self._inputLayer_j.getState()    
        ya, ga, ba, ha, ua = self._associativeLayer.getState()
        return {'id': self._id, 'y': self._y,\
                'association':{'id': self._assIdij, 'y':ya, 'g':ga, 'b':ba, 'h':ha, 'u':ua},
                'input': [{'id': self._id_i, 'yi': yi, 'g':gi, 'bi': bi, 'h':hi, 'u':ui},
                          {'id': self._id_j, 'yj': yj, 'g':gj, 'bj': bj, 'h':hj, 'u':uj}]}        
        
