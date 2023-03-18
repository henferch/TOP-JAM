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
import copy
import numpy as np
from network_server.network.layer.BaseLayer import BaseLayer

class AssociativeLayer(BaseLayer):
    #def __init__(self, ownerId_, netId_, id_, size_, lr_, afn_):
    def __init__(self, p_ = None):                        
        BaseLayer.__init__(self, p_)
        self._size = p_['size']
        self._ni = self._size[0]
        self._nj = self._size[1]
        self._N = self._ni * self._nj
        self._lr = p_['lr']
        self._W = np.zeros(self._size, dtype=np.float32)
        self._h = np.zeros(self._size, dtype=np.float32)
        self._b = np.zeros(self._size, dtype=np.float32)
        self._y = np.zeros(self._size, dtype=np.float32)
        
    def step(self, i_, j_, bel_=None, belGain_=None):
        for i in range(self._ni):
            ii = i_[i]
            for j in range(self._nj):
                self._W[i,j] = (1.0-self._lr)*self._W[i,j] + self._lr*ii*j_[j]                    
        self._h = self._afn.compute(self._W)            
        self._b = bel_
        self._bG = belGain_
        self._y = self._ut.computeStateBelief(self._h, self._b, self._bG)                        
        return copy.copy(self._y)
    
