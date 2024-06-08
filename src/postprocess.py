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

from .dependency_tree import *
from .utils import *
from collections import defaultdict

deprel_priority = [
    "nsubj", "obl", "conj", "cc", "ccomp", "advcl", "xcomp", "expl", "dep", "compound", "orphan", "nmod", "iobj", "cop",
    "advmod", "amod", "nummod", "det", "mark", "case", "aux", "aux:pass", "discourse", "vocative", "list", "punct", "obj",
    "nsubj:outer", "flat", "fixed", "goeswith", "dislocated", "reparandum", "flat:foreign", "_"
]

def apply_heuristics(test_sentences, output_trees):
    """
    Apply various heuristics to correct the dependency trees in output_trees based on test_sentences.
    """
    for ind_tree, tree in enumerate(output_trees):
        tree = fix_deprel_root_issues(tree)
        tree = fix_empty_root_tree(tree, test_sentences[ind_tree])

        if is_root_without_children(tree):
            tree = fix_root_without_children(tree, test_sentences[ind_tree])
        else:
            root_children, tree_without_root = get_root_children(tree)
            if len(root_children) > 1:
                tree = fix_multiple_root_children(tree_without_root, root_children)

        tokens_without_parent = find_tokens_without_parent(tree, test_sentences[ind_tree])
        if tokens_without_parent:
            tree = fix_tokens_without_parent(tree, tokens_without_parent)

        found_cycles = Graph(tree).findCycles()
        if found_cycles:
            tree = fix_cycles(tree, found_cycles)

        output_trees[ind_tree] = tree

    return output_trees

def fix_deprel_root_issues(tree: DependencyTree):
    """
    Correct nodes with P = T and wrong root dependencies in the given tree.
    """
    corrected_tree = DependencyTree()
    for node in tree.get_tree():
        if node[0] != node[2]:
            if node[0] != 0 and node[1] == "root":
                corrected_tree.add_dependency_relation(node[0], "[UNK]", node[2])
            elif not (node[0] == 0 and node[1] != "root"):
                corrected_tree.add_dependency_relation(node[0], node[1], node[2])
    return corrected_tree

def fix_empty_root_tree(tree: DependencyTree, sentence):
    """
    Correct empty trees by adding the root dependency if the sentence has at least two elements.
    """
    if not tree.get_tree() and len(sentence) >= 2:
        tree.add_dependency_relation(0, "root", 1)
    return tree

def is_root_without_children(tree: DependencyTree):
    """
    Check if the root token has any children.
    """
    return all(node[0] != 0 or node[1] != "root" for node in tree.get_tree())

def fix_root_without_children(tree: DependencyTree, sentence):
    """
    Apply heuristic to correct root token without children by assigning possible children.
    """
    tokens_without_parent = [
        token["id"] for token in sentence if token["id"] != 0 and 
        not any(node[2] == token["id"] for node in tree.get_tree()) and 
        any(node[0] == token["id"] for node in tree.get_tree())
    ]
    
    node_deprel_dict = defaultdict(list)
    for node in tree.get_tree():
        if node[0] in tokens_without_parent:
            node_deprel_dict[node[1]].append(node)

    for deprel in deprel_priority:
        if deprel in node_deprel_dict:
            for node in node_deprel_dict[deprel]:
                tree.add_dependency_relation(0, 'root', node[0])
                return tree

    tree.add_dependency_relation(0, 'root', tokens_without_parent[0])
    return tree

def get_root_children(tree: DependencyTree):
    """
    Check if the root token has more than one child and return those children along with the tree without root dependencies.
    """
    root_children = [node[2] for node in tree.get_tree() if node[0] == 0 and node[1] == "root"]
    tree_without_root = DependencyTree()

    for node in tree.get_tree():
        if not (node[0] == 0 and node[1] == "root"):
            tree_without_root.add_dependency_relation(node[0], node[1], node[2])

    return root_children, tree_without_root

def fix_multiple_root_children(tree: DependencyTree, root_children):
    """
    Apply heuristic to correct root token with more than one child by assigning appropriate parents.
    """
    tokens_without_parent = [
        token for token in root_children if 
        not any(node[2] == token for node in tree.get_tree()) and 
        any(node[0] == token for node in tree.get_tree())
    ]

    node_deprel_dict = defaultdict(list)
    for node in tree.get_tree():
        if node[0] in tokens_without_parent:
            node_deprel_dict[node[1]].append(node)

    for deprel in deprel_priority:
        if deprel in node_deprel_dict:
            for node in node_deprel_dict[deprel]:
                tree.add_dependency_relation(0, "root", node[0])
                return tree

    tree.add_dependency_relation(0, "root", root_children[0])
    return tree

def find_tokens_without_parent(tree: DependencyTree, sentence):
    """
    Identify tokens in the sentence that do not have parents in the tree.
    """
    return [
        token["id"] for token in sentence if token["id"] != 0 and 
        not any(node[2] == token["id"] for node in tree.get_tree())
    ]

def fix_tokens_without_parent(tree: DependencyTree, tokens_without_parent):
    """
    Apply heuristic to assign parents to tokens without parents in the tree.
    """
    root_child = next(node[2] for node in tree.get_tree() if node[0] == 0 and node[1] == "root")

    for token in tokens_without_parent:
        tree.add_dependency_relation(root_child, "[UNK]", token)
        for node in tree.get_tree():
            if node[0] == token and node[2] == root_child:
                tree.delete_dependency_relation(token, node[1], root_child)
                break

    return tree

def fix_cycles(tree: DependencyTree, cycles):
    """
    Apply heuristic to correct cycles found in the tree.
    """
    root_child = next(node[2] for node in tree.get_tree() if node[0] == 0 and node[1] == "root")

    for cycle in cycles:
        if root_child in cycle:
            for node in tree.get_tree():
                if node[0] != 0 and node[2] == root_child:
                    tree.delete_dependency_relation(node[0], node[1], node[2])
                    break
        else:
            for node in tree.get_tree():
                if node[0] == cycle[-2] and node[2] == cycle[-1]:
                    tree.delete_dependency_relation(node[0], node[1], node[2])
                    break

    return tree