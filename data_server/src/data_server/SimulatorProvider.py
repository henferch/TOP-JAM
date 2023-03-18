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

from data_server.Provider import Provider
from common_tools.Noise import Noise
import numpy as np
import copy
import pprint

class SimulatorProvider(Provider):
    
    def __init__(self, param_):
        Provider.__init__(self)
        self._simInstance = None
        self._stTransition = None
        self._state = None
        self._stateObs = None
        self._tinit = None
        self._cti = 0
        self._ctr = {}
        self._ctiMax = 0
        self._nextTrTime = 0 
        self._noise = {}
        self._transitions = []
        self._obsPeriodInMs = param_['obsPeriodInMs']
        params = self._ut.getParameters(param_['packDir'] + '/parameters/' +self.__class__.__name__ + '.json')
        if not params is None:
            noiseClass = params['noise']
            noiseProfile = params['noiseProfile']
            for k, p in params['observationNoise'].items():
                nP = {}
                for ki, pi in noiseProfile[p].items():
                    nPP = {}
                    for kii, pii in pi.items():
                        nc = noiseClass[pii['noise']]
                        nPP[kii] = eval("{}({},pii['param'])".format(nc['class'], nc['type']))    
                    nP[ki] = nPP
                self._noise[k] = nP

            self._state = params["state"]
            self._stateObs = copy.copy(self._state)
            self._stTransition = params["transition"]
            self._ctiMax = len(self._stTransition) - 1
            if self._ctiMax > -1:
                self._ctr = self._stTransition[self._cti]
                self._nextTrTime = self._ctr['ti'] 
        
        self._lastTime = self._ut.getCurrentTimeMS()        
    
    def addTransition(self, tr_):
        print("New transition : \n")
        pprint.pprint(tr_)
        ti = tr_['ti']
        te = tr_['te']
        ns = int((te - ti)/self._obsPeriodInMs)
        dt = 1.0/float(ns*1.0)
        tr_['c'] = 1
        tr_['n'] = ns        
        
        for t in tr_['d']:
            cst = self._state[t['id']]
            for k, v in t['st'].items():
                cst_v = cst[k]
                v['dd'] = (np.array(v['d']) - cst_v['d'])*dt
                v['dp'] = (np.array(v['p']) - cst_v['p'])*dt
        self._transitions.append(tr_)
        
    def _doTransitions(self):
        transitions = []
        for tr in self._transitions:
            for t in tr['d']:
                cst = self._state[t['id']]
                for k, v in t['st'].items():
                    cst_v = cst[k]
                    cst_v['d'] = (np.array(cst_v['d']) + v['dd']).tolist()
                    cst_v['p'] = (np.array(cst_v['p']) + v['dp']).tolist()
                c = tr['c'] + 1 
                if c <= tr['n']:
                    tr['c'] = c
                    transitions.append(tr)
        self._transitions = transitions
                
    def _getState(self, t_):
        for t in t_['d']:
            newState = self._state[t['id']]
            for k, v in t['st'].items(): 
                newState[k] = v       
    
    def _addNoise(self):
        newObs = {}        
        for (sk, sv), (nk, nv) in zip(self._state.items(), self._noise.items()):
            newObsi = {}
            for (ski, svi), (nki, nvi) in zip(sv.items(), nv.items()):
                newObsii = {}
                for (skii, svii), (nkii, nvii) in zip(svi.items(), nvi.items()):
                    newObsii[skii] = nvii.add(svii)    
                newObsi[ski] = newObsii
            newObs[sk] = newObsi
        self._stateObs = newObs
        
    def step(self):
        if self._tinit is None:
            self._tinit = self._ut.getCurrentTimeMS() 
        t = self._ut.getCurrentTimeMS() - self._tinit 
        if t >= self._nextTrTime and self._cti <= self._ctiMax: 
            self.addTransition(self._ctr)
            self._cti += 1
            if self._cti <=  self._ctiMax :
                self._ctr = self._stTransition[self._cti]
                self._nextTrTime = self._ctr['ti'] 
        self._doTransitions()
        self._addNoise()
        
        return copy.copy(self._state), copy.copy(self._stateObs)
        
    def release(self):
        # nothing particular to be done
        return None
    
    def read(self):
        self._data, self._lastTime = self._simInstance.getState()
        