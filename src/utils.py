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

from collections import defaultdict, Counter
from .dependency_tree import DependencyTree

class Graph:
    """
    Graph representation of a dependency tree to find cycles.
    """
    def __init__(self, tree: DependencyTree):
        self.graph = defaultdict(list)
        self.tree = tree.get_tree()
        for node in self.tree:
            self.graph[node[0]].append(node[2])

    def DFS(self, node, visited, path):
        """
        Depth-First Search (DFS) to find cycles in the graph.
        """
        visited[node] = True
        path.append(node)

        for neighbor in self.graph[node]:
            if visited[neighbor]:
                self.cycles.append(path[path.index(neighbor):] + [neighbor])
            else:
                self.DFS(neighbor, visited, path)

        path.pop()
        visited[node] = False

    def findCycles(self):
        """
        Find all cycles in the graph.
        """
        nodes = {node for edge in self.tree for node in (edge[0], edge[2])}
        visited = {node: False for node in nodes}
        self.cycles = []

        for node in list(self.graph):
            self.DFS(node, visited, [])

        return self.remove_duplicates(self.cycles)

    def lists_have_same_elements(self, list1, list2):
        """
        Check if two lists have the same elements.
        """
        return Counter(list1[:-1]) == Counter(list2[:-1])

    def remove_duplicates(self, lists):
        """
        Remove duplicate cycles from the list of cycles.
        """
        result = []

        for sublist in lists:
            if not any(self.lists_have_same_elements(sublist, existing) for existing in result):
                result.append(sublist)

        return result