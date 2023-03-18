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
from network_server.network.layer.BaseLayer import BaseLayer

class AttractorLayer(BaseLayer):
    def __init__(self, p_= None) :    
        BaseLayer.__init__(self, p_)  

        self._ref = np.array(p_['ref'])         # attractor reference values
        self._N = self._ref.shape[0]            # attractor unit number
        self._sigma = np.array(p_['sig'])              
        self._inputN = len(p_['inp'])           # input layers number
        self._inh = p_['inh']                   # inhibition factor
        self._dt = p_['dt']                     # time step in ms
        self._tau = min(p_['tau'],self._dt)     # time constant         
        self._TAU = (1.0/self._dt)*self._tau 
        self._one_Min_TAU = 1.0 - self._TAU
        
        self._W = np.zeros((self._N,self._N),dtype=np.float32) # recurrent weights        
                
        
        # precomputed const 
        self._invSigma, self._den = self._ut.getMultiGaussianConst(self._sigma)
                
        # setting recurrent weights analytically
        for i in range(self._N):            
            mgpc = self._ut.multiGaussianPrecomp(self._ref, self._ref[i], self._invSigma, self._den)
            self._W[i,:] = mgpc/mgpc.sum()        
        
        self._W_inh = self._W + self._inh

        self._Winput =  []# input mapping to attractor
        
        for i in p_['inp']:            
            if i['W'] == "gauss":
                self._Winput.append(None)            
            elif i['W'] == "eye":
                self._Winput.append("e")
            else:
                self._Winput.append(np.array(i['W']))
                
        
        self._f = np.zeros((self._N,),dtype=np.float32) 
        self._u = np.zeros((self._N,),dtype=np.float32) 
        self._h = np.zeros((self._N,),dtype=np.float32)
        self._b = np.zeros((self._N,),dtype=np.float32) 
        self._y = np.zeros((self._N,),dtype=np.float32) 
                
    def stepFromFiringRate(self, fr_, inputGain_, bel_=None, belGain_=None):
        self._f = np.dot(self._W_inh, self._h)
        
        for fr, g in zip(fr_, inputGain_):
            for i in range(self._N):
                self._f[i] += g*fr[i]
        self._u = self._TAU*self._h + self._one_Min_TAU*self._f
        self._h = self._afn.compute(self._u)    
        self._b = bel_
        self._bG = belGain_
        return copy.copy(self._h)
        
    def step(self, input_, inputGain_, bel_=None, belGain_=None):
        self._f = np.dot(self._W_inh, self._h)
        
        for i, g, Wi in zip(input_, inputGain_, self._Winput):
            if not i is None:
                m = None
                if Wi is None : 
                    m = self._ut.multiGaussianPrecomp(self._ref, i, self._invSigma, self._den)
                elif Wi == "e":
                    m = i
                else:
                    m = np.dot(i, Wi)                    
                self._f += g*m
                
        self._u = self._TAU*self._h + self._one_Min_TAU*self._f
        self._h = self._afn.compute(self._u)    
        
        self._b = bel_
        self._bG = belGain_
        return copy.copy(self._h)
                                 
