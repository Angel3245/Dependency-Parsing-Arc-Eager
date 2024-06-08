# Copyright (C) 2024  Jose Ángel Pérez Garrido
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import deque
from copy import deepcopy
from .dependency_tree import DependencyTree

class State:
    """
    Represents a state of the Arc-eager algorithm

    Variables:
    - a (sigma) = Stack, which stores partially processed words (LIFO)
    - b (beta) = Buffer, which stores the words that still need to be read (FIFO)
    - arcs = Set of arcs already created
    """
    def __init__(self) -> None:
        self.a = deque()
        self.b = deque()
        self.arcs = DependencyTree()

    @property
    def sigma(self):
        return self.a
    
    @property
    def beta(self):
        return self.b
    
    def get_sigma(self,n=1):
        if(n == 1):
            return self.sigma[-1]
        
        return list(self.sigma)[-n:]
    
    def pop_sigma(self):
        return self.sigma.pop()
    
    def put_sigma(self, item):
        self.sigma.append(item)

    def get_beta(self,n=1):
        if(n == 1):
            return self.beta[0]
        
        return list(self.beta)[:n]
    
    def pop_beta(self):
        return self.beta.popleft()
    
    def put_beta(self, item):
        self.beta.append(item)

    def add_arc(self,head,deprel,node):
        self.arcs.add_dependency_relation(head,deprel,node)
    
    def has_head(self,id):
        for arc in self.arcs.get_tree():
            # arc = (P, D, T) // P = parent token (head), D = deprel tag, T = current token
            if(id == arc[2]):
                return True
            
        return False
    
    def is_head(self,id):
        for arc in self.arcs.get_tree():
            # arc = (P, D, T) // P = parent token (head), D = deprel tag, T = current token
            if(id == arc[0]):
                return True
            
        return False
    
    def is_sigma_empty(self):
        return len(self.sigma) == 0

    def is_beta_empty(self):
        return len(self.beta) == 0
    
    def __str__(self) -> str:
        return "Sigma: "+str([token["id"] for token in self.a])+" | Beta: "+str([token["id"] for token in self.b])+" | Arcs: "+str(self.arcs)