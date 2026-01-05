import mesa
import networkx as nx
import random
from typing import Dict, List, Tuple


class AntColonyModel(mesa.Model):
    
    def __init__(
        self, 
        num_nodes: int = 21, 
        num_ants: int = 10, 
        decay_rate: float = 0.1, 
        version: int = 1, 
        min_food: int = 1, 
        max_food: int = 5, 
        use_pheromones: bool = True,
        pheromone_follow_prob: float = 0.8,
        clustering: int = 0,
        seed: int = 2025
    ):
        super().__init__(seed=seed)
        random.seed(seed)
        
        self.number_of_nodes = num_nodes
        self.number_of_ants = num_ants
        self.pheromone_decay_rate = decay_rate
        self.ant_version = version
        self.minimum_food_per_node = min_food
        self.maximum_food_per_node = max_food
        self.pheromones_enabled = use_pheromones
        self.pheromone_follow_probability = pheromone_follow_prob
        self.food_clustering_level = clustering
        
        self.graph, self.node_positions = self.build_graph(num_nodes)
        
        self.anthill_node = 0
        self.food_at_nodes = self.initialize_food_distribution(num_nodes)
        self.initial_food_at_nodes = self.food_at_nodes.copy()
        self.total_initial_food = sum(self.food_at_nodes)
        self.food_deposited_at_nodes: Dict[int, int] = {self.anthill_node: 0}
        self.total_food_collected = 0
        
        for source_node, target_node in self.graph.edges():
            self.graph.edges[source_node, target_node]['weight'] = 0.0
        
        from agents import AntFactory
        AntFactory.create_ants(self, num_ants, version)
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Food Remaining": lambda model: sum(model.food_at_nodes),
                "Food Collected": lambda model: model.total_food_collected,
                "Max Edge Weight": lambda model: max(
                    (model.graph.edges[edge]['weight'] for edge in model.graph.edges()), 
                    default=0
                ),
            }
        )
        self.running = True
    
    def build_graph(self, num_nodes: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
        node_positions = {0: (0.5, 0.5)}
        
        for node_id in range(1, num_nodes):
            node_positions[node_id] = (random.random(), random.random())
        
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        
        distance_threshold = 0.15 + (0.1 / (num_nodes ** 0.3))
        for node_a in range(num_nodes):
            for node_b in range(node_a + 1, num_nodes):
                if self.calculate_distance(node_positions[node_a], node_positions[node_b]) < distance_threshold:
                    graph.add_edge(node_a, node_b)
        
        self.ensure_graph_connectivity(graph, node_positions)
        return graph, node_positions
    
    def calculate_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        return ((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)**0.5
    
    def ensure_graph_connectivity(self, graph: nx.Graph, node_positions: Dict[int, Tuple[float, float]]):
        if nx.is_connected(graph):
            return
        
        connected_components = list(nx.connected_components(graph))
        anthill_component = next(component for component in connected_components if 0 in component)
        
        for component in connected_components:
            if component == anthill_component:
                continue
            
            closest_pair = None
            minimum_distance = float('inf')
            for node_in_component in component:
                for node_in_anthill_component in anthill_component:
                    distance = self.calculate_distance(
                        node_positions[node_in_component], 
                        node_positions[node_in_anthill_component]
                    )
                    if distance < minimum_distance:
                        minimum_distance = distance
                        closest_pair = (node_in_component, node_in_anthill_component)
            
            if closest_pair:
                graph.add_edge(*closest_pair)
    
    def initialize_food_distribution(self, num_nodes: int) -> List[int]:
        if self.food_clustering_level == 0:
            return [0] + [
                random.randint(self.minimum_food_per_node, self.maximum_food_per_node) 
                for _ in range(num_nodes - 1)
            ]
        
        food_values = [0] * num_nodes
        
        number_of_clusters = max(1, min(self.food_clustering_level, 8))
        available_nodes = list(range(1, num_nodes))
        
        if len(available_nodes) < number_of_clusters:
            cluster_center_nodes = available_nodes
        else:
            cluster_center_nodes = random.sample(available_nodes, number_of_clusters)
        
        nodes_with_food = set()
        nodes_per_cluster = min(8, max(5, num_nodes // (number_of_clusters * 10)))
        
        for cluster_center in cluster_center_nodes:
            cluster_center_position = self.node_positions[cluster_center]
            nodes_by_distance = []
            
            for node_id in range(1, num_nodes):
                if node_id not in nodes_with_food:
                    distance = self.calculate_distance(
                        self.node_positions[node_id], 
                        cluster_center_position
                    )
                    nodes_by_distance.append((node_id, distance))
            
            nodes_by_distance.sort(key=lambda x: x[1])
            
            actual_cluster_size = min(nodes_per_cluster, len(nodes_by_distance))
            for node_id, _ in nodes_by_distance[:actual_cluster_size]:
                food_values[node_id] = random.randint(
                    self.minimum_food_per_node, 
                    self.maximum_food_per_node
                )
                nodes_with_food.add(node_id)
        
        return food_values
    
    def step(self):
        total_deposited = sum(self.food_deposited_at_nodes.values())
        if total_deposited >= self.total_initial_food:
            self.running = False
            return
        
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        self.apply_pheromone_decay()
    
    def apply_pheromone_decay(self):
        for source_node, target_node in self.graph.edges():
            current_weight = self.graph.edges[source_node, target_node]['weight']
            self.graph.edges[source_node, target_node]['weight'] = max(
                0, 
                current_weight - self.pheromone_decay_rate
            )
    
    def all_food_collected(self) -> bool:
        return sum(self.food_at_nodes) == 0
    
    def get_completion_percentage(self) -> float:
        if self.total_initial_food == 0:
            return 100.0
        remaining_food = sum(self.food_at_nodes)
        return ((self.total_initial_food - remaining_food) / self.total_initial_food) * 100