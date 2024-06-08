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

from src.preprocess import *
from src.postprocess import *
from src.dependency_tree import *
from src.dp_model import *
from src.fileparser import *
from src.oracle import *

from pathlib import Path
from InquirerPy import inquirer
import os
import pickle
import subprocess
import matplotlib.pyplot as plt

def ask_training_config(datafolder):
    # Ask user to select between supported treebanks 
    language = inquirer.select(
        message = str(f"Select treebank language for training:"),
        choices = os.listdir(str(f"{datafolder}/Datasources")),
    ).execute()

    # Ask user network topology
    # Input sigma size features
    sigma_size = inquirer.number(
        message = "Select the sigma size for the input features:",
        default=2
    ).execute()

    # Input Beta size features
    beta_size = inquirer.number(
        message = "Select the beta size for the input features:",
        default=2
    ).execute()

    num_dense_layers = inquirer.number(
        message = "Select the number of dense layers:",
    ).execute()

    if int(num_dense_layers) > 0:
        dropout_rate_correct = False
        while not dropout_rate_correct:
            dropout_rate = inquirer.text(
                message = "Select the rate of dropout:",
            ).execute()
            try:
                dropout_rate = float(dropout_rate)
                dropout_rate_correct = True
            except ValueError:
                print("That's not a valid decimal number. Please try again.")
    else:
        dropout_rate = 0


    # Ask user hyperparameters for training
    # Optimizer
    optimizer = inquirer.select(
        message = "Select an optimizer:",
        choices = ["adam", "sgd", "rmsprop", "adadelta", "adagrad", "adamax", "nadam", "ftrl"],
    ).execute()

    # Number of epochs
    epochs = inquirer.number(
        message = "Select a number of epochs value:",
    ).execute()

    # Batch size
    batch_size = inquirer.number(
        message = "Select a batch size value:",
    ).execute()

    # Set training hyperparameters
    hyperparameters = {
        "loss" : "sparse_categorical_crossentropy",
        "optimizer" : optimizer,
        "metrics" : ["accuracy"],
        "epochs" : int(epochs),
        "batch_size" : int(batch_size)
    }

    # Set topology
    topology = {
        "sigma_size" : int(sigma_size),
        "beta_size" : int(beta_size),
        "num_dense_layers" : int(num_dense_layers),
        "dropout_rate" : int(dropout_rate)
    }

    return language, hyperparameters, topology

def ask_evaluation_config(datafolder):
    # Get trained models
    if not os.path.isdir(str(f"{datafolder}/Model_output")):
        raise Exception("No trained models found in ./Model_output.")
    
    model_names = os.listdir(str(f"{datafolder}/Model_output"))

    if len(model_names) == 0:
        raise Exception("No trained models found in ./Model_output.")
                  
    # Ask user to select between supported treebanks 
    language = inquirer.select(
        message=str(f"Select treebank language for evaluation:"),
        choices=os.listdir(str(f"{datafolder}/Datasources")),
    ).execute()

    # Ask user to select a previously trained model
    pos_model = inquirer.select(
        message="Select a model:",
        choices=model_names,
    ).execute()

    return language, pos_model


def train(datafolder):
    """
        Ask user for training configuration, train a POS model and save it as a pickle file.
    """
    language, hyperparameters, topology = ask_training_config(datafolder)

    print("Loading",language,"dataset...")

    # PREPROCESS INPUT SAMPLES
    parser = Conllu_parser()

    # Parse train file
    input_str = str(f"{datafolder}/Datasources/{language}/train.conllu")
    train_sentences = parser(input_str)

    # Parse validation file
    input_str = str(f"{datafolder}/Datasources/{language}/dev.conllu")
    val_sentences = parser(input_str)

    # Parse test file
    #input_str = str(f"{datafolder}/Datasources/{language}/test.conllu")
    #test_sentences = parser(input_str)


    # Generate dictionaries for labeling columns FORM, UPOS and DEPREL
    print("Generating dictionaries for labeling columns FORM, UPOS and DEPREL...")
    form_dict = generate_dict([(token["form"] for token in sentence) for sentence in train_sentences],2) #NOTE: labeling starts in 2
    form_dict["None"] = 1 # NOTE: Add special token None (padding if stack or buffer does not have enough elements)
    upos_dict = generate_dict([(token["upos"] for token in sentence) for sentence in train_sentences],2) #NOTE: labeling starts in 2
    upos_dict["None"] = 1 # NOTE: Add special token None (padding if stack or buffer does not have enough elements)
    deprel_dict = generate_dict([(token["deprel"] for token in sentence) for sentence in train_sentences])
    #deprel_dict = generate_dict([(token["deprel"] for token in sentence) for sentence in train_sentences],1) #NOTE: labeling starts in 1
    #deprel_dict["None"] = 0 # NOTE: Add special token None (transition does not have a relation, i.e. is REDUCE or SHIFT)
   
    
    # Generate dictionary for labeling transitions
    print("Generating dictionary for labeling transitions...")
    transition_dict = {
        "LEFT_ARC": 0,
        "RIGHT_ARC": 1,
        "REDUCE": 2,
        "SHIFT": 3,
    }
 
    #print("FORM Dictionary")
    #print(form_dict)
    #print("UPOS Dictionary")
    #print(upos_dict)
    #print("DEPREL Dictionary")
    #print(deprel_dict)

    # Create Dependency Parsing model architecture
    print("Creating Reference trees...")
    reference_train_trees = create_dependency_trees(train_sentences)
    reference_val_trees = create_dependency_trees(val_sentences)
    #reference_test_trees = create_dependency_trees(test_sentences)
   

    # Create Oracle
    print("Creating Arc-Eager Oracle...")
    oracle = Arc_Eager_Oracle()
   

    # Get transitions. Each pair (state, transition) is a training sample
    print("Preprocessing sentences with Arc-Eager Oracle...")
    train_states,train_transitions = oracle(reference_train_trees)
    val_states,val_transitions = oracle(reference_val_trees)
    #test_states,test_transitions = oracle(reference_test_trees)
   

    # Create samples
    print("Creating input and target samples...")
    x_train = preprocess_inputs(train_states, form_dict, upos_dict, topology["sigma_size"], topology["beta_size"])
    y_train = preprocess_targets(train_transitions, deprel_dict, transition_dict) #(transition_targets, relation_targets)
    
    x_val = preprocess_inputs(val_states, form_dict, upos_dict, topology["sigma_size"], topology["beta_size"])
    y_val = preprocess_targets(val_transitions, deprel_dict, transition_dict)
    #x_test = preprocess_inputs(test_states, form_dict, upos_dict, topology["sigma_size"], topology["beta_size"])
    #y_test = preprocess_targets(test_transitions, deprel_dict, transition_dict)
   
    #print(x_val)
    #print(y_val)
    

    print("Creating model...")
    model = DPModel(form_dict,upos_dict,deprel_dict, transition_dict) 
    model.build_model(topology)


    # Train
    print("TRAINING...")
    history = model.train((x_train,y_train),(x_val,y_val),hyperparameters)
   

    # Save model as a pickle file
    print("Saving model as a pickle file...")
    if not os.path.exists(str(f"{datafolder}/Model_output")):
        os.makedirs(str(f"{datafolder}/Model_output"))

    with open(str(f"{datafolder}/Model_output/{language}.pickle"), "wb") as data_file:
        pickle.dump(model,data_file)


    # Generate training plots
    print("Generating training plots...")
    # summarize history for accuracy
    plt.plot(history.history['relation_output_accuracy'])
    plt.plot(history.history['val_relation_output_accuracy'])
    plt.plot(history.history['transition_output_accuracy'])
    plt.plot(history.history['val_transition_output_accuracy'])
    plt.title('model accuracy for relation/transition outputs')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_relation', 'val_relation','train_transition', 'val_transition'], loc='upper left')
    #plt.show()

    # Save plot
    if not os.path.exists(str(f"{datafolder}/Plots/{language}")):
        os.makedirs(str(f"{datafolder}/Plots/{language}"))
    plt.savefig(str(f"{datafolder}/Plots/{language}/Plot_accuracy.png"))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['relation_output_loss'])
    plt.plot(history.history['val_relation_output_loss'])
    plt.plot(history.history['transition_output_loss'])
    plt.plot(history.history['val_transition_output_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_relation', 'val_relation','train_transition', 'val_transition'], loc='upper left')
    #plt.show()

    # Save plot
    plt.savefig(str(f"{datafolder}/Plots/{language}/Plot_loss.png"))
    plt.close()
    


def evaluate(datafolder):
    """
        Ask user for evaluation treebank and evaluate a POS model previously saved as a pickle file.
    """
    language_samples, pos_model = ask_evaluation_config(datafolder)
    
    # Load model
    print("Loading pickle model...")
    with open(str(f"{datafolder}/Model_output/{pos_model}"), "rb") as data_file:
        model = pickle.load(data_file)
    #model = keras.models.load_model(str(f"{datafolder}/Model_output/{pos_model}"))
    

    # Preprocess test samples
    print("Preprocessing test samples...")
    parser = Conllu_parser()
    input_str = str(f"{datafolder}/Datasources/{language_samples}/test.conllu")
    input_preprocessed_str = str(f"{datafolder}/Datasources/{language_samples}/test-prep.conllu")
    test_sentences = parser(input_str, input_preprocessed_str)


    # Get output Dependency tree
    print("Making prediction...")
    output_trees = model.predict(test_sentences)

    #print("\n\nSENTENCE:",test_sentences[100])
    #print("\nTREE:",output_trees[100])
    #print(output_trees)

    print("Applying heuristics...")
    output_trees_corrected = apply_heuristics(test_sentences,output_trees)
  

    print("Writing output CoNLL-U file...")
    writer = Conllu_writer()
    output_str = str(f"{datafolder}/Datasources/{language_samples}/test-results.conllu")
    writer(output_str,test_sentences,output_trees_corrected)


    print("Evaluating results...")
    eval=str(f"src/conll18_ud_eval.py")
    comand = ["python", eval, input_preprocessed_str, output_str, "-v"]
    result = subprocess.run(comand, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    

def main():
    datafolder=Path.cwd()

    print("Dependency parsing")

    # Ask user to select between the different functionalities supported by the program 
    functionality = inquirer.select(
        message="Select functionality:",
        choices=["Train a model", "Evaluate a model previously trained"],
    ).execute()

    function_map = {
        "Train a model" : train,
        "Evaluate a model previously trained" : evaluate
    }
    
    function_map[functionality](datafolder)


if __name__ == "__main__":
    main()