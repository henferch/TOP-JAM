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

from common_tools.Utils import Utils
from common_tools.ActivationFunction import *
from common_tools.Log import Log
import numpy as np
import copy 

class BaseLayer:
    def __init__(self, p_ = None):
        self._ut = Utils.getInstance()
        self._log = Log.getInstance()
        self._ownerId = p_['ownerId']
        self._netId = p_['id']
        self._id = p_['layerId']
        self._afn = None
        if isinstance(p_['afn'], str):
            self._afn = eval(p_['afn'])(p_)
        else:
            self._afn = p_['afn']
        self._W = None
        self._u = None        
        self._h = None
        self._b = None
        self._bG = None
        self._y = None
        self._N = None
        self._log.debug("Construction: {} '{}.{}' created".format(self.__class__.__name__, self._netId, self._id))
        
    def getId(self):
        return "{}.{}".format(self._netId,self._id)
        
    def getN(self):
        return self._N
    
    def setW(self, W_):
        if not self._W.shape == W_.shape:
            raise Exception("The weight matrix size should be {}!".format(self._W.shape))
        self._W = W_
    
    def getW(self, W_):        
        copy.copy(self._W)
    
    def getState(self):
        return copy.copy(self._y), self._bG, copy.copy(self._b), copy.copy(self._h), copy.copy(self._u)
    
    def step(self):
        raise Exception("Unimplemented method")        
    
            
    def getDesctiption(self):
        des = {'class': self.__class__.__name__, 'id': self._id, 'n': self._N, 'afn': self._afn.__class__.__name__}
        return des

    def __del__(self):
        self._ut.unregisterLayerId(self._netId, self._id)
        self._log.debug("Destruction: {} '{}.{}' destroyed".format(self.__class__.__name__, self._netId, self._id))
        