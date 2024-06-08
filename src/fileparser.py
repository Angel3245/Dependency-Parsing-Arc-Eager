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

from conllu import *
from .dependency_tree import DependencyTree

class Conllu_parser(object):

    def __call__(self,*args):
        with open(args[0], "r", encoding="utf-8") as file:
            sentences = parse(file.read())

        #print("Dataset loaded in memory.")
        #print("e.g. sentence number 10:",sentences[10])

        if len(args) == 1:
            for i in range(len(sentences)):
                # Ignore empty tokens and multiword units
                sentences[i]=sentences[i].filter(id=lambda x: not '-' in str(x) and not '.' in str(x)) 
                #sentences[i]=sentences[i].filter(id=lambda x: isinstance(x, int)) 

                # Add special ROOT item as ID 0
                sentences[i].append({"id": 0, "form": "ROOT", "lemma":"ROOT", "deprel":"None", "head":"None"})

        else:
            with open(args[1], "w", encoding="utf-8") as file:
                for i in range(len(sentences)):
                    # Ignore empty tokens and multiword units
                    sentences[i]=sentences[i].filter(id=lambda x: not '-' in str(x) and not '.' in str(x)) 
                    #sentences[i]=sentences[i].filter(id=lambda x: isinstance(x, int)) 

                    file.writelines([sentences[i].serialize()])
                    # Add special ROOT item as ID 0
                    sentences[i].append({"id": 0, "form": "ROOT", "lemma":"ROOT"})

        return sentences




class Conllu_writer(object):

    def __call__(self,dataset_output,sentences_output:list[TokenList], trees_output:list[DependencyTree]):
       
        for i in range(len(sentences_output)):
            sentences_output[i]=sentences_output[i].filter(id=lambda x: str(x)!='0') 
            for token in sentences_output[i]:
                for node in trees_output[i].get_tree():
                    if token["id"] == node[2]:
                        token["head"] = node[0]
                        token["deprel"] = node[1]

        with open(dataset_output, "w", encoding="utf-8") as file:
            file.writelines([sentence.serialize() for sentence in sentences_output])

