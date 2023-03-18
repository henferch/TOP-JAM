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
from common_tools.Utils import Utils
from common_tools.Log import Log

class ActivationLinear:
    def __init__(self, p_ = None):
        self._log = Log.getInstance()
        self._ut = Utils.getInstance()
        self._eps = self._ut.getEpsilon()
        self._log.debug("Construction: {} instance created".format(self.__class__.__name__))
    def compute(self, state_):
        return state_    
    def getMean(self, state_):
        return np.mean(state_)
    def __del__(self):
        self._log.debug("Destruction: {} instance distroyed".format(self.__class__.__name__))
        
class ActivationSigmoid(ActivationLinear):    
    def __init__(self, p_=None):
        ActivationLinear.__init__(self, p_)
        self._c = p_['c']
        self._l = p_['l']
    def compute(self, x_):
        return self._ut.sigmoid(x_, self._c, self._l) + self._eps

class ActivationSoftmax(ActivationLinear):    
    def __init__(self, p_=None):
        ActivationLinear.__init__(self, p_)
        self._smg = p_['smg']
    def compute(self, x_):
        return self._ut.softmax(x_*self._smg) + self._eps

class ActivationSaturation(ActivationLinear):
    def __init__(self, p_ = None):
        ActivationLinear.__init__(self, p_)  
        sat = p_['sat']
        if not len(sat)  == 2:
            raise Exception("A min and max saluration should be provided to [{}] activation function ".format(self.__class__.__name__))
        self._satMin,  self._satMax = sat  
    def compute(self, x_):
        return np.clip(x_, self._satMin, self._satMax) + self._eps


class ActivationNormalize(ActivationLinear):
    def __init__(self, p_ = None):
        ActivationLinear.__init__(self, p_)  
    def compute(self, x_):        
        return x_/(x_.sum()+self._eps)
    
class ActivationNormMax(ActivationLinear):
    def __init__(self, p_ = None):
        ActivationLinear.__init__(self, p_)  
    def compute(self, x_):
        return x_/(x_.max()+self._eps)

class ActivationGaussian(ActivationLinear):
    def __init__(self, p_ = None):
        ActivationLinear.__init__(self, p_)
        self._ref = p_['ref']
        self._sigma = p_['sigma']
        # precomputed const 
        self._invSigma = None
        self._den = None
        k = self._sigma.shape[0]*1.0
        if k > 1.0:
            self._invSigma = np.linalg.inv(self._sigma)
            self._den = np.sqrt((2.0*np.pi)**k + np.abs(np.linalg.det(self._sigma)))
        else:
            self._invSigma = (1.0/(self._sigma+self._eps)).reshape((1,1))
            self._den = self._sigma[0]
    def compute(self, x_):
        return self._ut.multiGaussianPrecomp(self._ref, x_, self._invSigma, self._den) + self._eps
            
        