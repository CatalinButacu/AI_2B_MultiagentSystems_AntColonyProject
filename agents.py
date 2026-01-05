import mesa
import random
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Optional


class DepositStrategy(ABC):
    
    @abstractmethod
    def get_deposit_target(self, ant: 'AntAgent') -> int:
        pass


class AnthillDepositStrategy(DepositStrategy):
    
    def get_deposit_target(self, ant: 'AntAgent') -> int:
        return ant.model.anthill_node


class HalfwayDepositStrategy(DepositStrategy):
    
    def get_deposit_target(self, ant: 'AntAgent') -> int:
        try:
            path_to_anthill = nx.shortest_path(
                ant.model.graph, 
                ant.current_node, 
                ant.model.anthill_node
            )
            if len(path_to_anthill) <= 2:
                return ant.model.anthill_node
            halfway_index = len(path_to_anthill) // 2
            return path_to_anthill[halfway_index]
        except nx.NetworkXNoPath:
            return ant.model.anthill_node


class AntFactory:
    
    strategies = {
        1: AnthillDepositStrategy,
        2: HalfwayDepositStrategy,
    }
    
    @classmethod
    def create_ants(cls, model, number_of_ants: int, version: int):
        strategy_class = cls.strategies.get(version, AnthillDepositStrategy)
        deposit_strategy = strategy_class()
        for _ in range(number_of_ants):
            AntAgent(model, deposit_strategy)
    
    @classmethod
    def register_strategy(cls, version: int, strategy_class):
        cls.strategies[version] = strategy_class


class AntAgent(mesa.Agent):
    
    def __init__(self, model: mesa.Model, deposit_strategy: DepositStrategy):
        super().__init__(model)
        self.current_node: int = model.anthill_node
        self.previous_node: Optional[int] = None
        self.is_carrying_food: bool = False
        self.visited_nodes: List[int] = [model.anthill_node]
        self.target_deposit_node: Optional[int] = None
        self.deposit_strategy = deposit_strategy
    
    def step(self):
        if self.is_carrying_food:
            self.return_with_food()
        else:
            self.search_for_food()
    
    def search_for_food(self):
        neighbor_nodes = list(self.model.graph.neighbors(self.current_node))
        if not neighbor_nodes:
            return
        
        next_node = self.choose_next_node(neighbor_nodes)
        self.move_to_node(next_node)
        self.try_collect_food()
    
    def choose_next_node(self, neighbor_nodes: List[int]) -> int:
        if not self.model.pheromones_enabled:
            return self.choose_random_unexplored_node(neighbor_nodes)
        
        edges_with_pheromones = self.get_pheromone_weighted_edges(neighbor_nodes)
        
        if not edges_with_pheromones:
            return self.choose_random_unexplored_node(neighbor_nodes)
        
        movement_choice = random.random()
        
        if movement_choice < 0.80:
            return self.follow_strongest_pheromone(edges_with_pheromones)
        elif movement_choice < 0.90:
            return self.explore_alternative_path(edges_with_pheromones, neighbor_nodes)
        else:
            return self.choose_random_unexplored_node(neighbor_nodes)
    
    def get_pheromone_weighted_edges(self, neighbor_nodes: List[int]) -> List[tuple]:
        return [
            (node, self.model.graph.edges[self.current_node, node]['weight'])
            for node in neighbor_nodes
            if self.model.graph.edges[self.current_node, node]['weight'] > 0
            and node not in self.visited_nodes[-3:]
        ]
    
    def follow_strongest_pheromone(self, edges_with_pheromones: List[tuple]) -> int:
        highest_weight = max(weight for _, weight in edges_with_pheromones)
        nodes_with_highest_weight = [
            node for node, weight in edges_with_pheromones 
            if weight == highest_weight
        ]
        return random.choice(nodes_with_highest_weight)
    
    def explore_alternative_path(self, edges_with_pheromones: List[tuple], neighbor_nodes: List[int]) -> int:
        if len(edges_with_pheromones) > 1:
            sorted_edges = sorted(edges_with_pheromones, key=lambda x: x[1], reverse=True)
            return sorted_edges[1][0]
        return self.choose_random_unexplored_node(neighbor_nodes)
    
    def choose_random_unexplored_node(self, neighbor_nodes: List[int]) -> int:
        nodes_except_previous = [
            node for node in neighbor_nodes 
            if node != self.visited_nodes[-1]
        ] if len(self.visited_nodes) > 1 else neighbor_nodes
        
        return random.choice(nodes_except_previous if nodes_except_previous else neighbor_nodes)
    
    def try_collect_food(self):
        if self.model.food_at_nodes[self.current_node] > 0:
            self.is_carrying_food = True
            self.model.food_at_nodes[self.current_node] -= 1
            self.model.total_food_collected += 1
            self.target_deposit_node = self.deposit_strategy.get_deposit_target(self)
    
    def return_with_food(self):
        if self.current_node == self.target_deposit_node:
            self.deposit_food()
            return
        
        next_node = self.get_next_node_toward_target()
        self.leave_pheromone_trail(next_node)
        self.move_to_node(next_node)
        
        if self.current_node == self.target_deposit_node:
            self.deposit_food()
    
    def get_next_node_toward_target(self) -> int:
        try:
            path_to_target = nx.shortest_path(
                self.model.graph, 
                self.current_node, 
                self.target_deposit_node
            )
            return path_to_target[1] if len(path_to_target) > 1 else self.target_deposit_node
        except nx.NetworkXNoPath:
            return random.choice(list(self.model.graph.neighbors(self.current_node)))
    
    def leave_pheromone_trail(self, next_node: int):
        if not self.model.pheromones_enabled:
            return
        
        edge = (min(self.current_node, next_node), max(self.current_node, next_node))
        if self.model.graph.has_edge(*edge):
            self.model.graph.edges[edge]['weight'] += 2
    
    def deposit_food(self):
        self.is_carrying_food = False
        self.target_deposit_node = None
        self.visited_nodes = [self.current_node]
        self.model.food_deposited_at_nodes[self.current_node] = \
            self.model.food_deposited_at_nodes.get(self.current_node, 0) + 1
    
    def move_to_node(self, node: int):
        self.previous_node = self.current_node
        self.current_node = node
        self.visited_nodes.append(node)
        if len(self.visited_nodes) > 20:
            self.visited_nodes = self.visited_nodes[-20:]