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

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from src.dependency_tree import *

def preprocess_inputs(states, form_dict, upos_dict, sigma_size=2, beta_size=2):

    inputs = [] #List of states
    
    for sentence_states in states:
        for state in sentence_states:

            sigma_values = state.get_sigma(sigma_size)
            beta_values = state.get_beta(beta_size)

            # Create input feature from word form and part-of-speech
            # taking N tokens in sigma/beta
            # NOTE: Tokens parameters are converted into a label using the corresponding dictionary
            
            # W contains the word form from tokens
            W_sigma = [(form_dict[token["form"]] if token["form"] in form_dict.keys() else form_dict["[UNK]"]) for token in sigma_values] 
            W_beta = [(form_dict[token["form"]] if token["form"] in form_dict.keys() else form_dict["[UNK]"]) for token in beta_values]

            # Add padding (token None) if missing tokens
            W_sigma = pad_sequences([W_sigma],maxlen=sigma_size, value=form_dict["None"], padding="pre")[0]
            W_beta = pad_sequences([W_beta],maxlen=beta_size, value=form_dict["None"], padding="post")[0]

            # P contains the part-of-speech from tokens
            P_sigma = [(upos_dict[token["upos"]] if token["upos"] in upos_dict.keys() else upos_dict["[UNK]"]) for token in sigma_values]
            P_beta = [(upos_dict[token["upos"]] if token["upos"] in upos_dict.keys() else upos_dict["[UNK]"]) for token in beta_values]

            # Add padding if missing tokens
            P_sigma = pad_sequences([P_sigma],maxlen=sigma_size, value=upos_dict["None"], padding="pre")[0]
            P_beta = pad_sequences([P_beta],maxlen=beta_size, value=upos_dict["None"], padding="post")[0]

            # Create input feature
            state_input = list(W_sigma) + list(W_beta) + list(P_sigma) + list(P_beta)

            inputs.append(state_input)
       

    # Convert inputs list to a numpy array
    inputs = np.array(inputs)

    #print(inputs[10])
    return inputs

def preprocess_targets(transitions, deprel_dict, transition_dict):

    transition_targets = [] # List of transitions
    relation_targets = [] # List of relations
    
    for sentence_transitions in transitions:
        for transition in sentence_transitions:
            # Create target feature from action and relation
            # NOTE: Parameters are converted into a label using the corresponding dictionary
            if(transition.relation in deprel_dict.keys()):
                deprel_label = deprel_dict[transition.relation]     
            else:
                deprel_label = deprel_dict["[UNK]"]

            # Save targets
            transition_targets.append(transition_dict[transition.action])
            relation_targets.append(deprel_label)
       

    # Convert targets lists to a numpy array
    transition_targets = np.array(transition_targets)
    relation_targets = np.array(relation_targets)

    return transition_targets, relation_targets

def create_dependency_trees(sentences):
    toret = []

    for sentence in sentences:
        # Create a Dependency tree for each sentence
        tree = DependencyTree()
        tree.create_tree_from_sentence(sentence)

        # Get only projective sentences
        if(tree.is_projective()):
            toret.append(tree)
        #else:
        #    print("Non-projective sentence found:",sentence)
            
    return toret

def generate_dict(samples,initial_value=0):
    # Create an empty dictionary to store the unique strings and their labels
    toret = {}
    
    # Add an UNKNOWN token 
    toret["[UNK]"] = initial_value

    # Counter for labeling
    label_counter = initial_value+1

    # Iterate through the list
    for sentence in samples:
        for item in sentence:
            if item not in toret:
                # If the string is not in the dictionary, add it with a label
                toret[item] = label_counter
                label_counter += 1

    return toret