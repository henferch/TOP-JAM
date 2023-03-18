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

class DataServerMsg():
    
    def toROSMsg(self, meta_, state_, msg):
        
        properties = []
        objects = []
        agents = []
        
        # properties 
        for m in meta_["propertiesId"]:
            mD = state_[m]["pp"]
            properties += [mD["d"][0], mD["p"][0]]
        
        # objects 
        for m in meta_["objectsId"]:
            mD = state_[m]["op"]
            objects += mD["d"][0:3] + [mD["p"][0]]
        
        # agents
        for m in meta_["agentsId"]:
            mD = state_[m]
            mF = mD["tp"]; agents += mF["d"][0:3] + [mF["p"][0]]
            mF = mD["to"]; agents += [mF["d"][0],   mF["p"][0]]
            mF = mD["hp"]; agents += mF["d"][0:3] + [mF["p"][0]]
            mF = mD["hd"]; agents += mF["d"][0:3] + [mF["p"][0]]
            mF = mD["rp"]; agents += mF["d"][0:3] + [mF["p"][0]]
            mF = mD["lp"]; agents += mF["d"][0:3] + [mF["p"][0]]
            mF = mD["sp"]; agents += [mF["d"][0],   mF["p"][0]]
                        
        msg.properties_data = properties
        msg.objects_data = objects
        msg.agents_data = agents        
    
    def fromROSMsg(self, meta_, msg_):
        state = {}
        
        p = msg_.properties_data
        o = msg_.objects_data
        a = msg_.agents_data

        i = 0
        for m in meta_["propertiesId"] : 
            state[m] = {"pp" : {"d" : np.array(p[i], np.float32), "p" : np.array(p[i+1], np.float32)} }
            i+= 2
        
        i = 0        
        for m in meta_["objectsId"] : 
            j = i+3
            state[m] = {"op" : {"d" : np.array(o[i:j], np.float32), "p" : np.array(o[j], np.float32)} }
            i+= 4
        
        i = 0
        for m in meta_["agentsId"] : 
            mD = {}
            j = i+3 ; mD["tp"] = {"d": np.array(a[i:j], np.float32), "p": np.array([a[j]], np.float32)} ; i += 4
            j = i+1 ; mD["to"] = {"d": np.array([a[i]], np.float32), "p": np.array([a[j]], np.float32)} ; i += 2
            j = i+3 ; mD["hp"] = {"d": np.array(a[i:j], np.float32), "p": np.array([a[j]], np.float32)} ; i += 4
            j = i+3 ; mD["hd"] = {"d": np.array(a[i:j], np.float32), "p": np.array([a[j]], np.float32)} ; i += 4
            j = i+3 ; mD["rp"] = {"d": np.array(a[i:j], np.float32), "p": np.array([a[j]], np.float32)} ; i += 4
            j = i+3 ; mD["lp"] = {"d": np.array(a[i:j], np.float32), "p": np.array([a[j]], np.float32)} ; i += 4
            j = i+1 ; mD["sp"] = {"d": np.array([a[i]], np.float32), "p": np.array([a[j]], np.float32)} ; i += 2
            state[m] = mD
            
        return state

class NetworkServerSrv():
    def toROSMsg(self, state_, msg_):
        
        individual_feature_ids = [];    individual_feature_dim = [];   individual_feature_ids_length = -1
        social_feature_ids = [];        social_feature_dim = [];       social_feature_ids_length = -1
        group_feature_ids = [];         group_feature_dim = [];        group_feature_ids_length = -1
        joint_feature_ids = [];         joint_feature_dim = [];        joint_feature_ids_length = -1

        msg_.topology_dimensions = state_['topologyDimensions']
        msg_.agent_ids = state_['agentsId']
        msg_.joint_scale_ids = state_['jointScaleIds']
        
        for a in msg_.agent_ids:
            v = state_[a]         
            indFeatIds = v['individualFeatureIds'] 
            if individual_feature_ids_length < 0:
                individual_feature_ids_length = len(indFeatIds)
                individual_feature_dim += v['individualFeatureDim']
            individual_feature_ids += indFeatIds
            
            socFeatIds = v['socialFeatureIds']
            if social_feature_ids_length < 0:
                social_feature_ids_length = len(socFeatIds)
                social_feature_dim += v['socialFeatureDim']
            social_feature_ids += socFeatIds
            
            groFeatIds = v['groupFeatureIds']
            if group_feature_ids_length < 0:
                group_feature_ids_length = len(groFeatIds)
                group_feature_dim += v['groupFeatureDim']
            group_feature_ids += groFeatIds
            
            joiFeatIds = v['jointFeatureIds']
            if joint_feature_ids_length < 0:
                joint_feature_ids_length = len(joiFeatIds)
                joint_feature_dim += v['jointFeatureDim']
            joint_feature_ids += joiFeatIds
            
        msg_.individual_feature_ids = individual_feature_ids
        msg_.social_feature_ids = social_feature_ids
        msg_.group_feature_ids = group_feature_ids
        msg_.joint_feature_ids = joint_feature_ids
        
        msg_.individual_feature_ids_length = individual_feature_ids_length
        msg_.social_feature_ids_length = social_feature_ids_length
        msg_.group_feature_ids_length = group_feature_ids_length
        msg_.joint_feature_ids_length = joint_feature_ids_length
        
        msg_.individual_feature_dim = individual_feature_dim
        msg_.social_feature_dim = social_feature_dim
        msg_.group_feature_dim = group_feature_dim
        msg_.joint_feature_dim = joint_feature_dim
        

    def fromROSMsg(self, msg_):
        state = {}
        
        iIndFeat = 0
        iSocFeat = 0
        iGroFeat = 0
        iJoiFeat = 0
        
        for aId in msg_.agent_ids :
            stA = {}
            jIndFeat = iIndFeat + msg_.individual_feature_ids_length
            stA['individualFeatureIds'] = msg_.individual_feature_ids[iIndFeat:jIndFeat]
            iIndFeat = jIndFeat
            stA['individualFeatureDim'] = msg_.individual_feature_dim
            
            jSocFeat = iSocFeat + msg_.social_feature_ids_length
            stA['socialFeatureIds'] = msg_.social_feature_ids[iSocFeat:jSocFeat]
            iSocFeat = jSocFeat
            stA['socialFeatureDim'] = msg_.social_feature_dim

            jGroFeat = iGroFeat + msg_.group_feature_ids_length
            stA['groupFeatureIds'] = msg_.group_feature_ids[iGroFeat:jGroFeat]
            iGroFeat = jGroFeat
            stA['groupFeatureDim'] = msg_.group_feature_dim
            
            jJoiFeat = iJoiFeat + msg_.joint_feature_ids_length
            stA['jointFeatureIds'] = msg_.joint_feature_ids[iJoiFeat:jJoiFeat]
            iJoiFeat = jJoiFeat
            stA['jointFeatureDim'] = msg_.joint_feature_dim
            
            state[aId] = stA 
        
        state['topologyDimensions'] = msg_.topology_dimensions
        state['agentsId'] = msg_.agent_ids
        state['jointScaleIds'] = msg_.joint_scale_ids
        
        return state 
                            
class NetworkServerMsg():
    
    def _buildStateIdList(self, metaList_):
        n = len(metaList_)
        i = 0
        fIds = []
        subIds = []
        subList = []
        while i < n:
            c = metaList_[i]
            if c == "*" :
                i += 1
                fIds.append(metaList_[i])
                if i > 1 :
                    subIds.append(subList) 
                    subList = []                

            else :
                subList.append(c)
            i += 1
        subIds.append(subList)
        
        return fIds, subIds
    
    def _fromStMapToList(self, idList, subIdList, st_):
        
        d = []

        for i in range(len(idList)):
            fIds_i = idList[i]
            fSubIds_i = subIdList[i]
            
            st_i = st_[fIds_i]
            if len(fSubIds_i) == 0:
                d += st_i.tolist()
            else:
                for f in fSubIds_i:
                    d += st_i[f].tolist()
        return d

    
    def _fromListToStMap(self, i_, data_, dims_, listIds_):

        fIds, subIds = self._buildStateIdList(listIds_)
        
        st = {}
        iData = i_;
        jData = iData 
        fi = 0            
        for fid, sfid in zip(fIds, subIds):
            if len(sfid) == 0:
                jData = iData + dims_[fi];
                st[fid] = np.array(data_[iData:jData], dtype=np.float32)            
                fi += 1
                iData = jData
            else:
                stAf = {}
                for ssfid in sfid:
                    jData = iData + dims_[fi];
                    fi += 1
                    stAf[ssfid] = np.array(data_[iData:jData], dtype=np.float32)               
                    iData = jData
                st[fid] = stAf                        
        
        return st, iData
    
      
        
    def toROSMsg(self, meta_, state_, msg):
           
        aD = []
        stI = state_["I"]
        stS = state_["S"]
        stG = state_["G"]
        stJ = state_["J"]
        
        agentsId = meta_["agentsId"]
        
        for aId in agentsId:
            
            aMeta = meta_[aId]
            
            indFIds, indFSubIds = self._buildStateIdList(aMeta["individualFeatureIds"])
            aD += self._fromStMapToList(indFIds, indFSubIds, stI[aId])
            
            socFIds, socFSubIds = self._buildStateIdList(aMeta["socialFeatureIds"])
            aD += self._fromStMapToList(socFIds, socFSubIds, stS[aId])
            
            groFIds, groFSubIds = self._buildStateIdList(aMeta["groupFeatureIds"])            
            aD += self._fromStMapToList(groFIds, groFSubIds, stG[aId])
            
            joiFIds, joiFSubIds = self._buildStateIdList(aMeta["jointFeatureIds"])        
            aD += self._fromStMapToList(joiFIds, joiFSubIds, stJ[aId])
        
        msg.agents_data = aD 
                    
        return msg

    def fromROSMsg(self, meta_, msg_):
                    
        aD = msg_.agents_data
        
        stI = {}
        stS = {}
        stG = {}
        stJ = {}
        
        agentsId = meta_["agentsId"]
        
        iD = 0
        for a in agentsId:
            mA = meta_[a]
            indFeaIds = mA['individualFeatureIds']             
            indFeaDim = mA['individualFeatureDim']
            socFeaIds = mA['socialFeatureIds']
            socFeaDim = mA['socialFeatureDim']
            groFeaIds = mA['groupFeatureIds']
            groFeaDim = mA['groupFeatureDim']
            joiFeaIds = mA['jointFeatureIds']
            joiFeaDim = mA['jointFeatureDim']
            
            stIa, iD = self._fromListToStMap(iD, aD, indFeaDim, indFeaIds)
            stSa, iD = self._fromListToStMap(iD, aD, socFeaDim, socFeaIds)
            stGa, iD = self._fromListToStMap(iD, aD, groFeaDim, groFeaIds)
            stJa, iD = self._fromListToStMap(iD, aD, joiFeaDim, joiFeaIds)
            
            stI[a] = stIa
            stS[a] = stSa
            stG[a] = stGa
            stJ[a] = stJa

        
        state = {'I': stI, 'S': stS, 'G': stG, 'J': stJ}
        
        return state