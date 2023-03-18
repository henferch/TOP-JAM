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
from common_tools.Utils import Utils

class MocapProvider(Provider):
    
    def __init__(self,param_):
        Provider.__init__(self)
        self._utils = Utils.getInstance() 
        self._stateObs = None
        self._data = None
        raise Exception("Unimplemented method")
    
    """ Method for building the state JSON, this information should be updated 
    with data from the motion capture system"""    
    def stateFactory(self):
        state = {"pSHt": {"pp": {"d": [1.0],            "p": [1.0]}},\
                 "pSHs": {"pp": {"d": [1.0],            "p": [1.0]}},
                 "pSHc": {"pp": {"d": [1.0],            "p": [1.0]}},
                 "pCOr": {"pp": {"d": [1.0],            "p": [1.0]}},
                 "pCOg": {"pp": {"d": [1.0],            "p": [1.0]}},
                 "pCOb": {"pp": {"d": [1.0],            "p": [1.0]}},
                 "o0": {"op": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}},
                 "o1": {"op": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}},
                 "o2": {"op": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}},
                 "o3": {"op": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}},
                 "o4": {"op": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}},
                 "A1":  {"tp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "to": {"d": [0.0],              "p": [1.0]}, 
                        "hp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "hd": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "rp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "ro": {"d": [0.0, 0.0],         "p": [1.0]},
                        "lp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "lo": {"d": [0.0, 0.0],         "p": [1.0]},
                        "sp": {"d": [0.0],              "p": [1.0]}},
                 "A2":  {"tp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "to": {"d": [0.0],              "p": [1.0]}, 
                        "hp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "hd": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "rp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "ro": {"d": [0.0, 0.0],         "p": [1.0]},
                        "lp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "lo": {"d": [0.0, 0.0],         "p": [1.0]},
                        "sp": {"d": [0.0],              "p": [1.0]}},
                 "A3":  {"tp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "to": {"d": [0.0],              "p": [1.0]}, 
                        "hp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "hd": {"d": [0.0, 0.0, 0.0],    "p": [1.0]}, 
                        "rp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "ro": {"d": [0.0, 0.0],         "p": [1.0]},
                        "lp": {"d": [0.0, 0.0, 0.0],    "p": [1.0]},
                        "lo": {"d": [0.0, 0.0],         "p": [1.0]},
                        "sp": {"d": [0.0],              "p": [1.0]}}}
        return state
        
    def step(self):
        raise Exception("Unimplemented method")
        

