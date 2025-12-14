import mesa
import random
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Optional


# Strategy Pattern: Deposit behaviors
class DepositStrategy(ABC):
    """Defines where ants deposit collected food."""
    
    @abstractmethod
    def get_deposit_target(self, ant: 'Ant') -> int:
        pass


class AnthillDepositStrategy(DepositStrategy):
    """Version 1: Return directly to anthill."""
    
    def get_deposit_target(self, ant: 'Ant') -> int:
        return ant.model.anthill


class HalfwayDepositStrategy(DepositStrategy):
    """Version 2: Deposit at halfway point."""
    
    def get_deposit_target(self, ant: 'Ant') -> int:
        try:
            path = nx.shortest_path(ant.model.graph, ant.position, ant.model.anthill)
            if len(path) <= 2:
                return ant.model.anthill
            return path[len(path) // 2]
        except nx.NetworkXNoPath:
            return ant.model.anthill


# Factory Pattern: Ant creation
class AntFactory:
    """Creates ants with appropriate deposit strategies."""
    
    _strategies = {
        1: AnthillDepositStrategy,
        2: HalfwayDepositStrategy,
    }
    
    @classmethod
    def create_ants(cls, model, num_ants: int, version: int):
        strategy_class = cls._strategies.get(version, AnthillDepositStrategy)
        strategy = strategy_class()
        for _ in range(num_ants):
            AntAgent(model, strategy)
    
    @classmethod
    def register_strategy(cls, version: int, strategy_class):
        cls._strategies[version] = strategy_class


class AntAgent(mesa.Agent):
    """Ant agent that searches for food and deposits it."""
    
    def __init__(self, model: mesa.Model, deposit_strategy: DepositStrategy):
        super().__init__(model)
        self.position: int = model.anthill
        self.previous_position: Optional[int] = None
        self.carrying_food: bool = False
        self.path: List[int] = [model.anthill]
        self.target_node: Optional[int] = None
        self.deposit_strategy = deposit_strategy
    
    def step(self):
        if self.carrying_food:
            self._return_with_food()
        else:
            self._search_for_food()
    
    def _search_for_food(self):
        neighbors = list(self.model.graph.neighbors(self.position))
        if not neighbors:
            return
        
        next_node = self._choose_next_node(neighbors)
        self._move_to(next_node)
        self._try_collect_food()
    
    def _choose_next_node(self, neighbors: List[int]) -> int:
        weighted = [
            (n, self.model.graph.edges[self.position, n]['weight'])
            for n in neighbors
            if self.model.graph.edges[self.position, n]['weight'] > 0
            and n not in self.path[-3:]
        ]
        
        if weighted:
            max_weight = max(w for _, w in weighted)
            best = [n for n, w in weighted if w == max_weight]
            return random.choice(best)
        
        available = [n for n in neighbors if n != self.path[-1]] if len(self.path) > 1 else neighbors
        return random.choice(available if available else neighbors)
    
    def _try_collect_food(self):
        if self.model.food_values[self.position] > 0:
            self.carrying_food = True
            self.model.food_values[self.position] -= 1
            self.model.total_food_collected += 1
            self.target_node = self.deposit_strategy.get_deposit_target(self)
    
    def _return_with_food(self):
        if self.position == self.target_node:
            self._deposit_food()
            return
        
        next_node = self._get_next_toward_target()
        self._leave_pheromone(next_node)
        self._move_to(next_node)
        
        if self.position == self.target_node:
            self._deposit_food()
    
    def _get_next_toward_target(self) -> int:
        try:
            path = nx.shortest_path(self.model.graph, self.position, self.target_node)
            return path[1] if len(path) > 1 else self.target_node
        except nx.NetworkXNoPath:
            return random.choice(list(self.model.graph.neighbors(self.position)))
    
    def _leave_pheromone(self, next_node: int):
        edge = (min(self.position, next_node), max(self.position, next_node))
        if self.model.graph.has_edge(*edge):
            self.model.graph.edges[edge]['weight'] += 1
    
    def _deposit_food(self):
        self.carrying_food = False
        self.target_node = None
        self.path = [self.position]
        self.model.food_at_deposits[self.position] = \
            self.model.food_at_deposits.get(self.position, 0) + 1
    
    def _move_to(self, node: int):
        self.previous_position = self.position
        self.position = node
        self.path.append(node)
        if len(self.path) > 20:
            self.path = self.path[-20:]