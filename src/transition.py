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

class Transition:
    """
    Represents a transition of the Arc-eager algorithm

    Variables:
    - action = Arc-eager transition applyied (LEFT-ARC, RIGHT-ARC, REDUCE, SHIFT)
    - relation = Relation set after applying the transition
    """
    def __init__(self,action,relation) -> None:
        self.action = action
        self.relation = relation
    
    def __str__(self) -> str:
        return "Action: "+str(self.action)+" | Relation: "+str(self.relation)