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

class BaseNetwork:
    
    def __init__(self, p_=None):
        self._ut = Utils.getInstance()
        self._log = Log.getInstance()
        self._id = p_['id']
        self._ownerId = p_['ownerId']
        self._inputLayer = None
        self._afn = None
        self._N = None
        
        if isinstance(p_['afn'], str):
            self._afn = eval(p_['afn'])(p_)
        else:
            self._afn = p_['afn']
    
        self._log.debug("Construction: {} '{}.{}'".format(self.__class__.__name__, self._ownerId, self._id))
        
    def getOutputDim(self):
        return self._N
    
    def getDescription(self):
        des = {'class': self.__class__.__name__, 'id': self._id}
        iput = []
        if not self._inputLayer is None:
            for i in self._inputLayer:
                iput.append(i.getDesctiption())    
        des['input'] = iput
        des['afn'] = self._afn.__class__.__name__
        return des     
    
    def __del__(self):
        self._ut.unregisterNetworkId(self._id)
        self._log.debug("Destruction: {} '{}.{}'".format(self.__class__.__name__, self._ownerId, self._id))
        
    def render(self, clear_=False):
        raise Exception("Unimplemented method")