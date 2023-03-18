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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rospkg
class Main():
    __instance = None
    
    @staticmethod 
    def getInstance():
       """ Static access method. """
       if Main.__instance == None:
          Main()
       return Main.__instance
   
    def __init__(self):
         """ Virtually private constructor. """
         if Main.__instance != None:
            raise Exception("This class is a singleton class, use 'getInstance()' instead!")
         else:                                             
             Main.__instance = self
     
    def closeGUI(self, event_):        
        
        print('Model GUI timer stopped')
        self._timer.stop()
        plt.close()        
        print('Model GUI closed')
        self._isRunning = False
            
    def isRunning(self):
        return self._isRunning
    
    def initialize(self, meta_):        

        self._fontsize = 14
        self._smallfontsize = 8
        IAn = meta_["topologyDimensions"]
        objIds = meta_["objectsId"]
        propIds = meta_["propertiesId"]
        
        agentsId = meta_["agentsId"]
        
        self._jointScaleIds = meta_["jointScaleIds"]
        self._ut = Utils.getInstance()
        self._IASize = (IAn[0],IAn[1])
        self._objIds = objIds
        self._propIds = propIds
        
        # Preparing the GUI
        self._fig = None
        self._axs = None
        self._fW = 15.0 # figure width
        self._fH = 10.0 # figure height
        
        self._facecolor = "#202729"
        #self._facecolor = "#000000"

        self._fig = plt.figure(figsize=(self._fW, self._fH), facecolor=self._facecolor)
        self._fig.tight_layout()
        gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[1.0, 2, 2, 2], width_ratios=[0.4, 2])
        
        self._nAtors = 3
        self._actorIndex = 0
        self._actorLabels = {"A1" :  agentsId[0], "A2" :  agentsId[1], "A3" : agentsId[2] }
        self._actorA1Label = self._actorLabels['A{}'.format(self._actorIndex+1)]
        self._actorA2Label = self._actorLabels['A{}'.format((self._actorIndex+1) % self._nAtors + 1)]
        self._actorA3Label = self._actorLabels['A{}'.format((self._actorIndex+2) % self._nAtors + 1)]
        
        # set the spacing between subplots
        plt.subplots_adjust(left=0.2,\
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.01,
                            hspace=0.4)

        
        self._headerCanvas = self._fig.add_subplot(gs[0, :])
        self._attractorCanvas = self._fig.add_subplot(gs[1, 0])
        self._IO_IP_ISCanvas = self._fig.add_subplot(gs[1, 1])
        self._SF_SG_JF_1_Canvas = self._fig.add_subplot(gs[2, :])        
        self._SF_SG_JF_2_Canvas = self._fig.add_subplot(gs[3, :])        
        
        self._sectionTitleColor ='#dddddd'
        self._sectionTitleFontweigth = 'bold'
        self._sectionTitleFontsize = 'small'
        self._markerSelectedEdgecolor = '#ffffff'
        self._markerNormalEdgecolor = '#5a5a5a'
        
        # Header view
        rospack = rospkg.RosPack()
        packDir = rospack.get_path('network_monitor') 
        self._img = Image.open(packDir + "/ressources/logo.png")
        #self._headerCanvas.set_facecolor("#32515a")               
        self._headerCanvas.set_xticks([])
        self._headerCanvas.set_yticks([])
        #self._headerCanvas.set_xlim(0,1800)
        self._headerCanvas.set_xlim(0,1900)
        self._headerCanvas.imshow(self._img, origin="upper")
        
        self._attractorCanvas.set_facecolor(self._facecolor)                                
        self._attractorCanvas.spines['top'].set_visible(False)
        self._attractorCanvas.spines['left'].set_visible(False)
        self._attractorCanvas.spines['bottom'].set_visible(False)
        self._attractorCanvas.spines['right'].set_visible(False)
        self._attractorCanvas.set_xticks([])
        self._attractorCanvas.set_yticks([])
        self._attractorCanvas.set_anchor('SE')
        self._attractorCanvas.margins(tight=True)

        self._guiObjects = {}        
        
        # Ia        
        plotScalefactor = 0.05 # important to determine the plot range !
        matrix = np.random.rand(self._IASize[0],self._IASize[1])*plotScalefactor
        oH = self._attractorCanvas.matshow(matrix)
        self._attractorCanvas.spines['top'].set_visible(False)
        self._attractorCanvas.spines['left'].set_visible(False)
        self._attractorCanvas.spines['bottom'].set_visible(False)
        self._attractorCanvas.spines['right'].set_visible(False)
        self._attractorCanvas.set_xticks([])
        self._attractorCanvas.set_yticks([])
        self._attractorCanvas.margins(tight=True)
        
        
        self._guiObjects = {"A1": {"IA" : oH}, "A2" : {}, "A3" : {}, "L" : [], "T": []}
        
        x = y = 0.0
        self._attractorCanvas.text(x+13, y+2, r'I$^{\rm a}$', fontsize=self._fontsize, c="#ffffff", weight="bold")                
        oH = self._attractorCanvas.text(x-10, y+5, "Actor {}".format(self._actorLabels['A1']), fontsize=self._fontsize, c="#ffffff", weight="bold")        
        self._guiObjects['L'].append(oH) 
                
                
        self._IO_IP_ISCanvas.set_facecolor(self._facecolor)                        
        self._IO_IP_ISCanvas.spines['top'].set_visible(False)
        self._IO_IP_ISCanvas.spines['left'].set_visible(False)
        self._IO_IP_ISCanvas.spines['bottom'].set_visible(False)
        self._IO_IP_ISCanvas.spines['right'].set_visible(False)
        self._IO_IP_ISCanvas.set_xticks([])
        self._IO_IP_ISCanvas.set_yticks([])
        self._IO_IP_ISCanvas.set_anchor('W')
        self._IO_IP_ISCanvas.margins(tight=True)                
                                  
        
        neuronColor = "#ffffaa"
        neuronEdgeColor = "#ffffff"
        self._neuronMarkerSize = 16 
        self._neuronMarkerEdgeWidth = 1
        hSkipt = 0.5
        vSkipt = 0.5
        nNeurons = 5
        nObjetcs = len(self._objIds)
        nProperties = len(self._propIds)
        

        self._IO_IP_ISCanvas.set_xlim([-0.5, hSkipt * (nObjetcs + nProperties + 1) + 0.5])
        self._IO_IP_ISCanvas.set_ylim([-0.5, vSkipt * nNeurons + 0.5])
        x = y = 0.0
        
        # IO Neurons 
        self._IO_IP_ISCanvas.text(x+(hSkipt*nObjetcs*0.35)-0.1, y+(vSkipt * nNeurons), r'I$^{\rm o}$', fontsize=self._fontsize, c="#ffffff", weight="bold")
        hSkipt = 0.425
        gui_o = {}
        for oId in self._objIds  :
            y = 0.0
            oH_list = []
            for n in range(nNeurons) :             
                oH, = self._IO_IP_ISCanvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas                 
                oH_list.append(oH)
                y += vSkipt
            gui_o[oId] = oH_list
            x += hSkipt

        self._guiObjects['A1']["IO"] = gui_o
        
        neuronColor = "#ffaaff"        
        y = 0.0
        self._IO_IP_ISCanvas.text(x+(hSkipt*nProperties*0.45)-0.1, y+(vSkipt * nNeurons), r'I$^{\rm p}$', fontsize=self._fontsize, c="#ffffff", weight="bold")
                                  
        # IP Neurons
        gui_p = {}
        for pId in self._propIds :
            y = 0.0
            oH_list = []
            for n in range(nNeurons) :             
                oH, = self._IO_IP_ISCanvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
                oH_list.append(oH)                
                y += vSkipt
            gui_p[pId] = oH_list
            x += hSkipt
        self._guiObjects['A1']["IP"] = gui_p
        
        neuronColor = "#aaffff"        
        y = 0.0
        self._IO_IP_ISCanvas.text(x-0.1, y+(vSkipt * nNeurons), r'I$^{\rm s}$', fontsize=self._fontsize, c="#ffffff", weight="bold")
                                  
        # IS Neurons
        oH_list = []
        for n in range(nNeurons) :             
            oH, = self._IO_IP_ISCanvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
            oH_list.append(oH)                
            y += vSkipt        
        
        hSkipt = 0.5
        self._guiObjects['A1']["IS"] = oH_list

        self._SF_SG_JF_1_Canvas.set_facecolor(self._facecolor)                        
        self._SF_SG_JF_1_Canvas.spines['top'].set_visible(False)
        self._SF_SG_JF_1_Canvas.spines['left'].set_visible(False)
        self._SF_SG_JF_1_Canvas.spines['bottom'].set_visible(False)
        self._SF_SG_JF_1_Canvas.spines['right'].set_visible(False)
        self._SF_SG_JF_1_Canvas.set_xticks([])
        self._SF_SG_JF_1_Canvas.set_yticks([])
        self._SF_SG_JF_1_Canvas.margins(tight=True)
                                                    
        idS = ["SF", "SB", "SO", "SC", "ST"] 
        idSLabels = {"SF": r'S$^{\rm f}$',\
                     "SL": r'S$^{\rm l}$', 
                     "SR": r'S$^{\rm r}$', 
                     "SB": r'S$^{\rm b}$', 
                     "SO": r'S$^{\rm o}$', 
                     "SC": r'S$^{\rm c}$', 
                     "ST": r'S$^{\rm t}$'}
        
        idG = ["GA", "GS", "GO", "GT", "GC", "GB", "GF"]         
        idGLabels = {"GA": r'G$^{\rm a}$',\
                     "GB": r'G$^{\rm b}$', 
                     "GF": r'G$^{\rm f}$', 
                     "GO": r'G$^{\rm o}$', 
                     "GC": r'G$^{\rm c}$', 
                     "GT": r'G$^{\rm t}$', 
                     "GS": r'G$^{\rm s}$'}
        
        idJ = ["$Ind$", "$Mon$", "$Com$", "$Mut$", "$Sha$"]
        
        self._SF_SG_JF_1_Canvas.set_xlim([-0.5, hSkipt * (len(idS) + len(idG) + 2) + 0.5])
        self._SF_SG_JF_1_Canvas.set_ylim([-0.5, vSkipt * nNeurons + 0.5])        
        
        x = 0.0
        y = 0.0
        
        xSJ = x+6.4
        ySJ = y+(vSkipt * nNeurons *0.5)+1.3
        gui_a2_S = {}
        oH = self._SF_SG_JF_1_Canvas.text(x-0.9, y+(vSkipt * nNeurons *0.5), "{}⇋{}".format(self._actorLabels['A1'], self._actorLabels['A2']), fontsize=self._fontsize, c="#ffffff", weight="bold")
        self._guiObjects['L'].append(oH) 
        oH = self._SF_SG_JF_1_Canvas.text(xSJ, ySJ, "Scane of jointness", fontsize=self._smallfontsize, c="#ffffff", weight="bold")        
        self._guiObjects['T'].append(oH) 
        neuronColor = "#aaffaa"
        for i in idS :
            y = 0.0
            oH_list = []
            self._SF_SG_JF_1_Canvas.text(x-0.1, y+(vSkipt * nNeurons), idSLabels[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            for n in range(nNeurons) :             
                oH, = self._SF_SG_JF_1_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
                oH_list.append(oH)          
                y += vSkipt
            gui_a2_S[i] = oH_list
            x += hSkipt
        self._guiObjects['A2']["SF"] = gui_a2_S

        x += hSkipt*0.5
        y = 0.0
        neuronColor = "#fffaaa"
        gui_a2_G = {}
        for i in idG :
            y = 0.0
            oH_list = []
            self._SF_SG_JF_1_Canvas.text(x-0.1, y+(vSkipt * nNeurons), idGLabels[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            for n in range(nNeurons) :             
                oH, = self._SF_SG_JF_1_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
                oH_list.append(oH)          
                y += vSkipt
            gui_a2_G[i] = oH_list
            x += hSkipt
        self._guiObjects['A2']["GF"] = gui_a2_G

        x += hSkipt*0.5
        y = 0.0        
        oH_list_2 = []
        nidJ = len(idJ)
        for i in range(nidJ) :             
            self._SF_SG_JF_1_Canvas.text(x+0.2, y-0.1, idJ[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            oH, = self._SF_SG_JF_1_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
            oH_list_2.append(oH)          
            y += vSkipt
        self._guiObjects['A2']["JF"] = oH_list_2
        
        x += hSkipt
        
        self._SF_SG_JF_2_Canvas.set_facecolor(self._facecolor)                        
        self._SF_SG_JF_2_Canvas.spines['top'].set_visible(False)
        self._SF_SG_JF_2_Canvas.spines['left'].set_visible(False)
        self._SF_SG_JF_2_Canvas.spines['bottom'].set_visible(False)
        self._SF_SG_JF_2_Canvas.spines['right'].set_visible(False)
        self._SF_SG_JF_2_Canvas.set_xticks([])
        self._SF_SG_JF_2_Canvas.set_yticks([])
        self._SF_SG_JF_2_Canvas.margins(tight=True)
                
        self._SF_SG_JF_2_Canvas.set_xlim([-0.5, hSkipt * (len(idS) + len(idG) + 2) + 0.5])
        self._SF_SG_JF_2_Canvas.set_ylim([-0.5, vSkipt * nNeurons + 0.5])
        
        x = 0.0
        y = 0.0
        
        gui_a3_S = {}
        oH = self._SF_SG_JF_2_Canvas.text(x-0.9, y+(vSkipt * nNeurons *0.5), "{}⇋{}".format(self._actorLabels['A1'], self._actorLabels['A3']), fontsize=self._fontsize, c="#ffffff", weight="bold")
        self._guiObjects['L'].append(oH) 
        oH = self._SF_SG_JF_2_Canvas.text(xSJ, ySJ, "Scane of jointness", fontsize=self._smallfontsize, c="#ffffff", weight="bold")        
        self._guiObjects['T'].append(oH) 
        neuronColor = "#aaffaa"
        for i in idS :
            y = 0.0
            oH_list = []
            self._SF_SG_JF_2_Canvas.text(x-0.1, y+(vSkipt * nNeurons), idSLabels[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            for n in range(nNeurons) :             
                oH, = self._SF_SG_JF_2_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
                oH_list.append(oH)          
                y += vSkipt
            gui_a3_S[i] = oH_list
            x += hSkipt
        self._guiObjects['A3']["SF"] = gui_a3_S

        x += hSkipt*0.5
        y = 0.0
        neuronColor = "#fffaaa"
        gui_a3_G = {}
        for i in idG :
            y = 0.0
            oH_list = []
            self._SF_SG_JF_2_Canvas.text(x-0.1, y+(vSkipt * nNeurons), idGLabels[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            for n in range(nNeurons) :             
                oH, = self._SF_SG_JF_2_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
                oH_list.append(oH)          
                y += vSkipt
            gui_a3_G[i] = oH_list
            x += hSkipt
        self._guiObjects['A3']["GF"] = gui_a3_G

        x += hSkipt*0.5
        y = 0.0        
        oH_list_3 = []
        nidJ = len(idJ)
        for i in range(nidJ) :             
            self._SF_SG_JF_2_Canvas.text(x+0.2, y-0.1, idJ[i], fontsize=self._fontsize, c="#ffffff", weight="bold")
            oH, = self._SF_SG_JF_2_Canvas.plot(x, y, marker='o', markersize=self._neuronMarkerSize, color=neuronColor, alpha = 0.0, markeredgewidth=self._neuronMarkerEdgeWidth, markeredgecolor=neuronEdgeColor) # canvas 
            oH_list_3.append(oH)          
            y += vSkipt
        self._guiObjects['A3']["JF"] = oH_list_3
        
        x += hSkipt
        
        # Interface events and loop            
        self._period = 50   
        self._timer = None        
        self._step = 0
        
        self._attractorCanvas.figure.canvas.mpl_connect('key_press_event', self.keyPressed)
        self._attractorCanvas.figure.canvas.mpl_connect('close_event', self.closeGUI)

        self._isRunning = True
        self._state = None
        self._timer = self._fig.canvas.new_timer(interval=self._period, callbacks=[(self.render, [], {})])
        self._timer.start()        
        
        plt.show()
        self._fig.canvas.draw()
        
    
    def keyPressed(self, event):        
        if event.key == "tab":
            print('pressed', event.key)
            self._actorIndex = (self._actorIndex + 1) % self._nAtors                        
            self._actorA1Label = self._actorLabels['A{}'.format(self._actorIndex+1)]
            self._actorA2Label = self._actorLabels['A{}'.format((self._actorIndex+1) % self._nAtors + 1)]
            self._actorA3Label = self._actorLabels['A{}'.format((self._actorIndex+2) % self._nAtors + 1)]
        if event.key == "escape":
            print('pressed', event.key)
            self.closeGUI(None)
            
    def setState(self, state_):
        self._state = state_
        
            
    def getTimeSteps(self):
        return self._step

    def render(self):
                
        
        if self._state is None :
            return

        i = self._actorIndex
        k = 0
        for oH in self._guiObjects["L"]:
            if k > 0:
                oH.set(text="{}⇋{}".format(self._actorLabels['A{}'.format(self._actorIndex+1)], self._actorLabels['A{}'.format(i+1)]))
            else:
                oH.set(text="Actor {}".format(self._actorLabels['A{}'.format(self._actorIndex+1)]))
                k += 1
            i = (i + 1) % 3
        
        stI = self._state['I']
        stS = self._state['S']
        stG = self._state['G']
        stJ = self._state['J']
        
        # indivirual features
        dataI = stI[self._actorA1Label]    
        A1 = self._guiObjects["A1"]
        A1['IA'].set(data=dataI['IA'].reshape(self._IASize))
        for k, v in A1.items():
            if k == "IA":
                continue
            elif k == "IS":
                dataIS = dataI['IS']
                i = 0
                for oH in v:
                    oH.set(alpha=dataIS[i])
                    i += 1
            else: 
                dataX = None
                if k == "IO" : 
                    dataX = dataI['IO']
                elif k == "IP" : 
                    dataX = dataI['IP']
                for (k1, v1) in v.items():
                    dataIOo = dataX[k1]
                    i = 0
                    for oH in v1:
                        oH.set(alpha=dataIOo[i])
                        i += 1
                        
        # Social, Group and  Joint features
        dataS = stS[self._actorA1Label]
        dataG = stG[self._actorA1Label]
        dataJ = stJ[self._actorA1Label]

        for a, other in zip(["A2","A3"], [self._actorA2Label, self._actorA3Label]):
            
            A = self._guiObjects[a]
            
            for k, v in A.items():
                d = None
                if k == "JF":
                    d = dataJ
                    i = 0
                    do = d[k][other]

                    for oH in v:
                        oH.set(alpha=do[i])
                        i += 1
                else : 
                    if k == "SF":
                        d = dataS
                    elif k == "GF":
                        d = dataG
                        
                    for k1, v1 in v.items():
                        do = d[k1][other]
                        i = 0
                        for oH in v1:
                            oH.set(alpha=do[i])
                            i += 1
        
        self._fig.canvas.draw()  
        self._step += 1
        print("step: " , self._step)
        
