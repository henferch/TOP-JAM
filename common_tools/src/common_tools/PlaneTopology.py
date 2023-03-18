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

from skspatial.objects import Plane, Points, Line
import numpy as np
import copy

class PlaneTopology:
    def __init__(self, params_):
        self._data = params_['data']
        offsetProportion = params_['offset'] # space offser proportion
        self._res = params_['res']    
        minTemp = np.min(self._data, 0)        
        maxTemp = np.max(self._data, 0)                
        ranTemp = np.array(maxTemp) - np.array(minTemp)
        offset = ranTemp*offsetProportion
        self._mindat = minTemp - offset/2.0
        self._maxdat = maxTemp + offset/2.0
        self._range = ranTemp + offset
        (rows, cols) = self._data.shape
        self._gridDimensions = np.round(self._range/self._res).astype(int)
        self._center = None
        self._plane = None
        self._epsilon=1e-6
        
        # computing the plane's center
        pList = []
        self._center = np.zeros((3,), dtype=np.float32)
        for r in range(rows):
            self._center += self._data[r,:]    
            pList.append(self._data[r,:].tolist())
        self._center /= float(rows*1.0)
        self._points = Points(pList)

        # computing the plane's best fit
        try :
            self._plane = Plane.best_fit(self._points)
        except ValueError as ve:
            print(ve)
        
    def getGridDimensions(self):
        return self._gridDimensions
    
    def getLims(self):
        return copy.copy(self._mindat), copy.copy(self._maxdat)
     
    def getCenter(self):
        return copy.copy(self._center)
    
    def getGrid(self, res_=None):                
        res = res_
        if res is None:
            res = self._res
        else :
            self._gridDimensions = np.round(self._range/res).astype(int)
        # compute grid centroids for the topology 
        c = res*0.5
        grid = []        
        for j in range(self._gridDimensions[0]):            
            for i in range(self._gridDimensions[1]):    
                grid.append([self._mindat[0] + res*i + c, self._mindat[1] + res*j + c])                 
        return grid
     
    def checkLimits(self,p_): 
        if p_[0] < self._mindat[0] or p_[0] > self._maxdat[0] or p_[1] < self._mindat[1] or p_[1] > self._maxdat[1]:
            return None
        return p_

    def project(self, p_):
        projPoint = None        
        try :
            projPoint = self._plane.project_point(p_)
        except ValueError as ve:
            print(ve)
        return self.checkLimits(projPoint)
                
    def intersect(self, p_, ray_):
        intPoint = None
        line = Line(p_, ray_)
        try :
            intPoint = self._plane.intersect_line(line)
        except ValueError as ve:
            print(ve)
        return self.checkLimits(intPoint)

    