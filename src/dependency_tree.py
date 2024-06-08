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

def is_projective(arcs: list):
    """
    Determines if a dependency tree has crossing arcs or not.
    Parameters:
    arcs (list): A list of tuples of the form (headid, dependentid), coding
    the arcs of the sentence, e.g, [(0,3), (1,4), …]
    Returns:
    A boolean: True if the tree is projective, False otherwise
    """
    for (i,j) in arcs:
        for (k,l) in arcs:
            if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
                return False
    return True
            
class DependencyTree:

    def __init__(self) -> None:
        self.tree = [] #[(P, D, T), (P, D, T), (P, D, T), (P, D, T), ...]
        
    def create_tree_from_sentence(self, sentence):
        self.sentence = sentence

        for token in sentence:
            # Avoid tokens without dependency relations
            if not token["head"] == "_" and not token["head"] == "None":
                # Create arc and save relation
                self.add_dependency_relation(token["head"], token["deprel"], token["id"]) 

    def add_dependency_relation(self,head,deprel,token):
        # Add dependency relation (deprel) between current token (id) to its parent (head)
        # (P, D, T) // P = parent token (head), D = deprel tag, T = current token
        self.tree.append((head, deprel, token))

    def delete_dependency_relation(self,head,deprel,token):
        self.tree.remove((head, deprel, token))
    
    def get_tree(self):
        return self.tree

    def is_projective(self):
        return is_projective([(relation[0],relation[2]) for relation in self.tree])

    def __str__(self) -> str:
        return str([relation for relation in self.tree]) 

    def __sub__(self, tree):
        # Return dependency relations in the current tree that are not in the passed tree
        # Current tree - current tree ^ passed tree
        return [item for item in self.tree if item not in tree.get_tree()]
