import mesa
import random
import networkx as nx

print("DEBUG: agents.py module loaded")

class BaseAnt(mesa.Agent):
    """Base ant agent with common behavior for both versions."""
    
    def __init__(self, model):
        super().__init__(model)
        self.position = model.anthill
        self.carrying_food = False
        self.path = [model.anthill]
        self.target_node = None  # For returning behavior
    
    def step(self):
        if self.carrying_food:
            self._return_with_food()
        else:
            self._search_for_food()
    
    def _search_for_food(self):
        """Search for food following pheromone trails or randomly."""
        neighbors = list(self.model.graph.neighbors(self.position))
        if not neighbors:
            return
        
        # Check for non-zero weight edges (pheromone trails)
        weighted_neighbors = [
            (n, self.model.graph.edges[self.position, n]['weight'])
            for n in neighbors
            if self.model.graph.edges[self.position, n]['weight'] > 0
            and n not in self.path[-3:]  # Avoid recent nodes
        ]
        
        if weighted_neighbors:
            # Follow highest weight edge
            max_weight = max(w for _, w in weighted_neighbors)
            best = [n for n, w in weighted_neighbors if w == max_weight]
            next_node = random.choice(best)
        else:
            # Random search - avoid going back immediately
            available = [n for n in neighbors if n != self.path[-1]] if len(self.path) > 1 else neighbors
            next_node = random.choice(available if available else neighbors)
        
        self._move_to(next_node)
        
        # Check if found food
        if self.model.food_values[self.position] > 0:
            self.carrying_food = True
            self.model.food_values[self.position] -= 1
            self.model.total_food_collected += 1
            self.target_node = self._get_deposit_target()
    
    def _return_with_food(self):
        """Return food to target node, increasing edge weights."""
        if self.position == self.target_node:
            self._deposit_food()
            return
        
        # Navigate toward target using shortest path
        try:
            path_to_target = nx.shortest_path(
                self.model.graph, self.position, self.target_node
            )
            next_node = path_to_target[1] if len(path_to_target) > 1 else self.target_node
        except nx.NetworkXNoPath:
            next_node = random.choice(list(self.model.graph.neighbors(self.position)))
        
        # Increase pheromone on traversed edge
        edge = (min(self.position, next_node), max(self.position, next_node))
        if self.model.graph.has_edge(*edge):
            self.model.graph.edges[edge]['weight'] += 2
        
        self._move_to(next_node)
        
        # Check if reached target
        if self.position == self.target_node:
            self._deposit_food()
    
    def _deposit_food(self):
        """Deposit food and start searching again."""
        self.carrying_food = False
        self.target_node = None
        self.path = [self.position]
        self.model.food_at_deposits[self.position] = self.model.food_at_deposits.get(self.position, 0) + 1
    
    def _move_to(self, node):
        """Move to a node and update path."""
        self.position = node
        self.path.append(node)
        if len(self.path) > 20:
            self.path = self.path[-20:]  # Keep path manageable
    
    def _get_deposit_target(self):
        """Override in subclasses to define where to deposit food."""
        raise NotImplementedError


class AntV1(BaseAnt):
    """Version 1: Returns food directly to the anthill."""
    
    
    def _get_deposit_target(self):
        return self.model.anthill


class AntV2(BaseAnt):
    """Version 2: Deposits food at a node halfway to the anthill."""
    
    def _get_deposit_target(self):
        try:
            path = nx.shortest_path(self.model.graph, self.position, self.model.anthill)
            if len(path) <= 2:
                return self.model.anthill
            halfway_index = len(path) // 2
            return path[halfway_index]
        except nx.NetworkXNoPath:
            return self.model.anthill