#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:10:34 2021

@author: hfchame
"""

from common_tools.Utils import Utils

class Provider:
    def __init__(self):
        self._lastTime = None
        self._ut = Utils.getInstance()            
    
    def initialize(self, param_):
        raise Exception("Unimplemented method")
    
    def relase(self):
        raise Exception("Unimplemented method")
    
    def step(self):
        raise Exception("Unimplemented method")
    
    def read(self):
        raise Exception("Unimplemented method")