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

import copy

from .state import State
from .dependency_tree import DependencyTree
from .transition import Transition

class Arc_Eager_Oracle:    
    
    def __call__(self,reference_trees) -> None:
        toret_states = []
        toret_transitions = []

        for reference_tree in reference_trees:
            states, transitions = (self.generate_transitions(reference_tree))
    
            toret_states.append(states)
            toret_transitions.append(transitions)

        #states, transitions = (self.generate_transitions(reference_trees[10]))
            
        #toret_states.append(states)
        #toret_transitions.append(transitions)

        return toret_states, toret_transitions #List of lists of transitions and states that produce

    def generate_transitions(self, reference_tree):
        """
        Input: A dependency tree
        Output: A sequence of transitions that would generate the same dependency tree
                for the input sequence
        """
        states = []
        transitions = []

        s = self.initial_state(reference_tree.sentence)

        #print("Initial state:\n",s)

        while not self.final_state(s):
            # Copy and save state
            states.append(copy.deepcopy(s))

            if self.LEFT_ARC_is_valid(s) and self.LEFT_ARC_is_correct(reference_tree,s):
                #print("Applying LEFT_ARC")
                transitions.append(self.apply_left_arc(s))

            elif self.RIGHT_ARC_is_valid(s) and self.RIGHT_ARC_is_correct(reference_tree,s):
                #print("Applying RIGHT_ARC")
                transitions.append(self.apply_right_arc(s))

            elif self.REDUCE_is_valid(s) and self.REDUCE_is_correct(reference_tree,s):
                #print("Applying REDUCE")
                transitions.append(self.apply_reduce(s))

            else:
                #print("Applying SHIFT")
                transitions.append(self.apply_shift(s))

            #print("State:\n",s)
        
        #print(reference_tree)
        return states, transitions  #Transitions and states that produce
                                    #transitions[0] produces states[0], transitions[1] produces states[1] ...
            
    #
    #   State functions
    #
    def initial_state(self, sentence):
        """
            Initially, sigma only contains the ROOT node and beta
            contains the complete sequence used as input and no
            arcs have been created yet.
        """
        toret = State()

        for token in sentence:
            if(token["id"] == 0):
                toret.put_sigma(token)
            else:
                toret.put_beta(token)

        return toret

    def final_state(self, state:State):
        """
            Check if beta = []
        """
        return state.is_beta_empty()
    
    #
    #   Arc-eager algorithm actions
    #
    def LEFT_ARC_is_valid(self, state:State):
        """
            Check if LEFT_ARC meet the preconditions:
            Condition 1. The top of the stack cannot be ROOT (i=0)
            Condition 2. The top of the stack cannot have a head 
        """
        if(state.is_sigma_empty()):
            return False

        # Get word currently at the top of the stack (i)
        i = state.get_sigma()

        # Condition 1. The top of the stack cannot be ROOT (i=0)
        # Condition 2. The top of the stack cannot have a head 
        return not (i["id"] == 0 or state.has_head(i)) 
    
    def LEFT_ARC_is_correct(self, reference_tree:DependencyTree, state:State):
        """
            Check if LEFT_ARC contributes to obtaining the target tree,
            i.e the resulting arc is contained in the reference tree
        """
        # Get first word in the buffer (j) and the
        # word currently at the top of the stack (i)
        i=state.get_sigma()
        j=state.get_beta()

        # Create arc from the first word in the buffer to the
        # word currently at the top of the stack
        arc = (j["id"],i["deprel"],i["id"]) 

        return arc in reference_tree.get_tree()

    def apply_left_arc(self,state:State):
        # Get first word in the buffer (j) and the
        # word currently at the top of the stack (i)
        i=state.pop_sigma()
        j=state.get_beta()

        # Create arc from the first word in the buffer to the
        # word currently at the top of the stack
        state.add_arc(j["id"],i["deprel"],i["id"]) 

        return Transition("LEFT_ARC",i["deprel"]) 


    def RIGHT_ARC_is_valid(self, state:State):
        """
            Check if RIGHT_ARC meet the preconditions:
            Condition 1. The first word in the buffer cannot have a head 
        """
        # Get first word in the buffer (j)
        j = state.get_beta()

        # The first word in the buffer cannot have a head 
        return not state.has_head(j["id"]) 
    
    def RIGHT_ARC_is_correct(self, reference_tree:DependencyTree, state:State):
        """
            Check if RIGHT_ARC contributes to obtaining the target tree,
            i.e the resulting arc is contained in the reference tree
        """
        # Get first word in the buffer (j) and the
        # word currently at the top of the stack (i)
        i=state.get_sigma()
        j=state.get_beta()

        # Create arc from the first word in the buffer to the
        # word currently at the top of the stack
        arc = (i["id"],j["deprel"],j["id"])

        return arc in reference_tree.get_tree()

    def apply_right_arc(self,state:State):
        # Get first word in the buffer (j) and the
        # word currently at the top of the stack (i)
        i=state.get_sigma()
        j=state.pop_beta()

        # Create arc from the word at the top of the stack to
        # the first word in the buffer
        state.add_arc(i["id"],j["deprel"],j["id"]) 

        # Return word currently at the top of the stack into it
        # and move the first word from the buffer to the top
        # of the stack
        state.put_sigma(j)

        return Transition("RIGHT_ARC",j["deprel"]) 

    def REDUCE_is_valid(self, state:State):
        if(state.is_sigma_empty()):
            return False
        
        # Get first word in the stack (i)
        i = state.get_sigma()

        # The first word in the stack must have a head 
        return state.has_head(i["id"]) 
    
    def REDUCE_is_correct(self, reference_tree:DependencyTree, state:State):
        """
            Check if REDUCE contributes to obtaining the target tree,
            i.e there is no children contained in the reference tree
        """
        # Get arcs from reference tree not yet assigned
        remaining_arcs = reference_tree - state.arcs
        #print("Remaining arcs:",remaining_arcs)
        #print([arc[0] for arc in remaining_arcs])

        # Get the word currently at the top of the stack (i)
        i = state.get_sigma()
        #print(i["id"])

        # If the arc has not been assigned yet we cannot reduce
        return not i["id"] in [arc[0] for arc in remaining_arcs]

    def apply_reduce(self,state:State):
        # Remove the word from the top of the stack
        state.pop_sigma()

        return Transition("REDUCE","None")

    def apply_shift(self,state:State):
        # Move the first word from the buffer to the top of the stack
        state.put_sigma(state.pop_beta())

        return Transition("SHIFT","None")