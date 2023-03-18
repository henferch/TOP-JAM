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
from enum import Enum
import random

class Noise:
    
    class Type(Enum):
        Null = 1
        Gaussian = 2
        Uniform = 3
        Binomial = 4
        
    def __init__(self, type_, params_):
        self._type = type_                
        self._lims = np.array(params_['l']).reshape((2,-1))        
        self._dim = self._lims.shape[1]        
        self._mu = 0.0
        self._sigma = 1.0
        self._prob = 0.0
        if 'mu' in params_:
            self._mu = params_['mu']
        if 'sigma' in params_:
            self._sigma = params_['sigma']
        if 'p' in params_:
            self._prob = params_['p']
            
    def add(self, obs_):                
        if not type(obs_) is list:
            obs_ = [obs_]       
        val = None 
        if self._type == Noise.Type.Null:
            val = np.array(obs_) 
        elif self._type == Noise.Type.Gaussian:            
            val = np.array(obs_) + np.random.normal(self._mu, self._sigma, (len(obs_),))            
            # restrict noise to space limits
            for s in range(self._dim):          
                val[s] = max(self._lims[0,s], min(val[s],self._lims[1,s]))             
        elif self._type == Noise.Type.Uniform:            
            val = np.array(obs_) 
            if random.random() < self._prob:
                val = np.random.uniform(self._lims[0,:], self._lims[1,:], (len(obs_),))                 
            for s in range(self._dim):          
                val[s] = max(self._lims[0,s], min(val[s],self._lims[1,s]))
        elif self._type == Noise.Type.Binomial:
            val = np.random.binomial((len(obs_),), self._prob)            
        
        return val.tolist()