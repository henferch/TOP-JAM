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

from network_server.feature.IndividualFeature import *
from network_server.feature.SocialFeature import *
from network_server.feature.GroupFeature import *
from network_server.feature.JointFeature import JointScale
import numpy as np
import pprint

class Agent:
    def __init__(self, id_, p_):
        
        self._id =  id_        
        self._ut =  p_['ut']        
        self._objectsId = p_['objectsId']
        self._topology =  p_['topology']
        self._refTopology = self._topology.getGrid()
        self._NTopology = len(self._refTopology)
        self._sigTopology = p_['sigTop']
        self._NObjects = len(self._objectsId)                
        self._propertiesId = p_['propertiesId']        
        self._NProperties = len(self._propertiesId)                        
        self._propertiesVector = {}        
        i = 0
        for pId in self._propertiesId:
            propVec = np.zeros((self._NProperties,), dtype=np.float32)
            propVec[i] = 1.0
            self._propertiesVector[pId] = propVec
            i += 1
                    
        self._agentsId = p_['agentsId']
        self._nAgents = len(self._agentsId)      
        self._othersId = []
        for a in self._agentsId :
            if a == self._id :
                continue
            self._othersId.append(a)
        self._individualFeatures = {}        
        self._socialFeatures = {}        
        self._groupFeatures = {}        
        self._jointFeatures = None
        
        self._dt = p_['obsPeriodInMs']
        
        # instantiating features by class name        
        self._indFeatIds, self._indFeatList = self._createFeaturesFromParams(p_['individualFeature'])
        self._socFeatIds, self._socFeatList = self._createFeaturesFromParams(p_['socialFeature'])
        self._groFeatIds, self._groFeatList = self._createFeaturesFromParams(p_['groupFeature'])
        self._joiFeatIds, self._joiFeatList = self._createFeaturesFromParams(p_['jointScale'])
        self._jointScaleIds = p_['jointScale'][0]['JF']['param']['typeId']        
        
    def getJointScaleIds(self):
        return self._jointScaleIds
    
    def buildIdDimLists(self, featIds_, fetList):
        idList =  []
        featDim = []
        
        for fId, f in zip(featIds_, fetList):
            idList += ["*",fId]
            fD = f['d']
            for a in self._othersId:
                featDim.append(fD[a].getDim())
                idList.append(a)
        return idList, featDim

        
    def getStateDim(self):
        stateDim = {}
        
        indFeatDim = []
        indIdList =  []
        for fId, f in zip(self._indFeatIds, self._indFeatList):
            indIdList += ["*",fId]
            c = f['c']
            if c == "1":
                indFeatDim.append(f['d'].getDim())
            elif c == "p" : 
                fD = f['d']
                for p in self._propertiesId:
                    indIdList += [p]
                    indFeatDim.append(fD[p].getDim())
            elif c == "o" :
                fD = f['d']
                for o in self._objectsId:
                    indIdList += [o]
                    indFeatDim.append(fD[o].getDim())

                    
        socIdList, socFeatDim = self.buildIdDimLists(self._socFeatIds, self._socFeatList)
        groIdList, groFeatDim = self.buildIdDimLists(self._groFeatIds, self._groFeatList)
        joiIdList, joiFeatDim = self.buildIdDimLists(self._joiFeatIds, self._joiFeatList)
                          
        stateDim["individualFeatureIds"] = indIdList
        stateDim["individualFeatureDim"] = indFeatDim

        stateDim["socialFeatureIds"] = socIdList
        stateDim["socialFeatureDim"] = socFeatDim

        stateDim["groupFeatureIds"] = groIdList
        stateDim["groupFeatureDim"] = groFeatDim

        stateDim["jointFeatureIds"] = joiIdList
        stateDim["jointFeatureDim"] = joiFeatDim
                        
        return stateDim
           
    
    def _createFeaturesFromParams(self, params_):
        
        featIds = []
        featList = []
        
        for fparams in params_ :        
            for k, p in fparams.items():
                cardinality = p['cardinality'][0]
                featSet = {"c" : cardinality}                            
                extra = {'agentId': self._id, 'otherId': None, 'ut': self._ut, 'dt': self._dt, 'top': self._topology}                        
                par = p['param'] 
                par.update(extra)
                               
                par['n'] = self._getN(par)
                for inp in par['inp']:
                    inp['n'] = self._getN(inp)
                
                if cardinality == "1" :                      
                    featSet.update({'d': eval(k)(k,par) })                
                elif cardinality == "o" : # objects 
                    dSet = {}                                        
                    for oId in self._objectsId:                    
                        par['otherId'] = oId
                        dSet[oId] = eval(k)("{}_{}".format(k,oId),par) 
                    featSet.update({'d': dSet })
                elif cardinality == "p" :  # properties
                    dSet = {}
                    for pId in self._propertiesId:                    
                        par['otherId'] = pId                    
                        dSet[pId] = eval(k)("{}_{}".format(k,pId),par) 
                    featSet.update({'d': dSet })
                elif cardinality == "a" : # agents 
                    dSet = {}
                    for aId in self._agentsId:                    
                        if aId == self._id:
                            continue
                        par['otherId'] = aId         
                        if k == "JF":
                            dSet[aId] = JointScale(k, par)
                        else:
                            dSet[aId] = eval(k)("{}_{}".format(k,aId),par) 
                    featSet.update({'d': dSet })             
                else:
                    raise Exception("unknown cardinality [{}] for individualFeature [{}]".format(cardinality, k))  
                        
            featIds.append(k)
            featList.append(featSet)
            
            
        return featIds,  featList 
        
    def _getN(self, par_):
        n = par_['n']
        if n == 'topology':
            n = self._NTopology
            par_['ref'] = self._refTopology
            par_['sig'] = self._sigTopology
        elif n == 'objects':    
            n = self._NObjects
        elif n == 'properties':
            n = self._NProperties
        return n
    
    def stepIndividualFeatures(self, obsState_):        
        stI = {}
        st = obsState_
        for k, f in zip(self._indFeatIds, self._indFeatList) : 
            c = f['c']
            d = f['d']
            st_f = None
            if c == "1":
                st_f = d.step(st)
            elif c == "o": # objects 
                st_f = {}
                for oId, f in d.items():                    
                    st_f[oId] = f.step(st)                     
            elif c == "p": # properties
                st_f = {}                
                for pId, f in d.items(): 
                    st['pv'] = self._propertiesVector[pId]
                    st_f[pId] = f.step(st)                     
            st[self._id][k] = st_f
            
            stI[k] = st_f
        return stI
    
    def stepSocialFeatures(self, obsState_):      
        stS = {}
        st = obsState_
        for k, f in zip(self._socFeatIds, self._socFeatList) :  
            c = f['c']
            d = f['d']
            if c == "1":
                st_f = d.step(st)
            elif c == "a": # agents
                st_f = {}
                for aId, f in d.items():                    
                    st_f[aId] = f.step(st)                                 
            st[self._id][k] = st_f
            stS[k] = st_f
        return stS
                  
        
    def stepGroupFeatures(self, obsState_):            
        stG = {}
        st = obsState_
        for k, f in zip(self._groFeatIds, self._groFeatList) :  
            st_f = {}
            d = f['d']
            for aId, f in d.items():                    
                st_f[aId] = f.step(st)
            st[self._id][k] = st_f
            stG[k] = st_f
        return stG
                                    
    def stepJointFeatures(self, obsState_):            
        stJ = {}        
        st = obsState_
        for k, f in zip(self._joiFeatIds, self._joiFeatList) :  
            st_f = {}
            d = f['d']
            for aId, f in d.items():  
                st_f[aId] = f.step(st)                                             
            stJ[k] = st_f        
        return stJ
                   
