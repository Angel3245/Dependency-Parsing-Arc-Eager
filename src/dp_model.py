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

import keras
import tensorflow as tf
import numpy as np
from tqdm.keras import TqdmCallback
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.callbacks import EarlyStopping
from .oracle import Arc_Eager_Oracle
from .preprocess import preprocess_inputs
from .dependency_tree import DependencyTree
import copy

class DPModel(object):

    def __init__(self, form_dict, upos_dict, deprel_label_dict, transition_label_dict):
        self.model = None
        self.topology = None
        self.form_dict = form_dict
        self.upos_dict = upos_dict
        self.deprel_label_dict = deprel_label_dict
        self.transition_label_dict = transition_label_dict

    def build_model(self, topology):

        self.topology = topology

        # Create an Input layer (Each sigma and beta token feed its word_form and part-of-speech) # TODO: Check input number
        input_size = (topology["sigma_size"]+topology["beta_size"])*2
        
        inputs = tf.keras.Input(shape=(input_size,), dtype=tf.int32)

        # An Embedding layer
        x = Embedding(max(len(self.form_dict),len(self.upos_dict))+2, 2)(inputs)

        # A Flatten layer
        x = Flatten()(x)

        # Add Dense layers 
        for _ in range(topology["num_dense_layers"]):
            x = Dense(units=8, activation='relu')(x)
            # Add Dropout after each Dense layer
            x = Dropout(topology["dropout_rate"])(x)  

        # Dense layers for the computation of results
        x1 = Dense(units=8, activation='relu')(x)
        x2 = Dense(units=8, activation='relu')(x)

        # Output layers with the appropiate activation function for a multiclass classifier

        # Transition output (4 possible Arc-Eager transitions)
        out1 = Dense(units=len(self.transition_label_dict), activation='softmax', name="transition_output")(x1)
        # Relation output (size of the possible relations dictionary)
        out2 = Dense(units=len(self.deprel_label_dict), activation='softmax', name="relation_output")(x2)

        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=[out1,out2])

        print(self.model.summary())
        #keras.utils.plot_model(self.model, show_shapes=True)

    
    def train(self, train_sets, val_sets, hyperparameters):
        # Prepare training input in batches
        train_ds = tf.data.Dataset.from_tensor_slices((train_sets[0],train_sets[1]))
        train_ds = train_ds.batch(hyperparameters["batch_size"])

        #val_ds = tf.data.Dataset.from_tensor_slices((val_sets[0],val_sets[1]))
        #val_ds = val_ds.batch(hyperparameters["batch_size"])

        # Compile the model
        self.model.compile(loss=hyperparameters["loss"],
              optimizer=hyperparameters["optimizer"],
              metrics=hyperparameters["metrics"])

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train the model and show validation loss and validation accuracy at the end of each epoch
        history = self.model.fit(train_ds,
            epochs=hyperparameters["epochs"], validation_data=(val_sets[0],val_sets[1]),
            callbacks=[TqdmCallback(),es]
        )

        return history
        

    def predict(self, test_sets):
        # Prediction “vertically”
        
        # Create the Oracle in order to use some functions
        oracle = Arc_Eager_Oracle()

        # Get tags
        deprel_tags = list(self.deprel_label_dict.keys()) 
        transition_tags = list(self.transition_label_dict.keys())

        # Take a batch of input sentences.
        batch = {oracle.initial_state(sentence):idx for idx,sentence in enumerate(test_sets)} # List of initial states (sigma, beta, arc) of the sentences
        toret = [None] * len(batch)

        # While the batch is not empty
        while not batch == {}:
            # Extract the batch of features for the current set of states: [[W_sigma + W_beta + P_sigma + P_beta], [W_sigma + W_beta + P_sigma + P_beta], ...]
            inputs = preprocess_inputs([list(batch)], self.form_dict, self.upos_dict, self.topology["sigma_size"], self.topology["beta_size"])
        
            # Make predictions with the trained model at the batch level.
            transitions,relations = self.model.predict(inputs) # For each initial state a list of 4 prob (for transitions) and a list of how many probs as different relations (48) (for relations)
            
            for state, transition_probabilities, relation_probabilities in zip(list(batch),transitions,relations):  
                # Convert numpy array of transition probabilities to list
                transition_probabilities = list(transition_probabilities)

                # Sort the possible transitions descending by probability
                sort_transition_probabilites = transition_probabilities.copy()
                sort_transition_probabilites.sort(reverse=True)
                
                # Get the deprel with the highest probability
                idx_relation_tag = relation_probabilities.argmax()
                deprel = deprel_tags[idx_relation_tag]
                
                transition_aplied = False

                while(not transition_aplied):
                    # Get the transition with highest probability
                    idx_transition_tag = transition_probabilities.index(sort_transition_probabilites.pop(0))
                    transition = transition_tags[idx_transition_tag]
                    
                    # Verify that the predicted transitions meet the preconditions
                    if transition == "LEFT_ARC" and oracle.LEFT_ARC_is_valid(state):
                        i=state.pop_sigma()
                        j=state.get_beta()
                        state.add_arc(j["id"],deprel,i["id"]) 
                        #Transition(transition,deprel) 
                        transition_aplied = True

                    elif transition == "RIGHT_ARC" and oracle.RIGHT_ARC_is_valid(state):
                        i=state.get_sigma()
                        j=state.pop_beta()
                        state.add_arc(i["id"],deprel,j["id"]) 
                        state.put_sigma(j)
                        #Transition(transition,deprel) 
                        transition_aplied = True

                    elif transition == "REDUCE" and oracle.REDUCE_is_valid(state):
                        state.pop_sigma()
                        #Transition(transition,None) 
                        transition_aplied = True

                    elif transition == "SHIFT":
                        state.put_sigma(state.pop_beta())
                        #Transition(transition,None) 
                        transition_aplied = True

           
                # Remove those instances from the batch that have reached a final state and save the generated Dependency Tree
                if oracle.final_state(state):
                    toret[batch.pop(state)] = state.arcs

        return toret