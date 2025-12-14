import mesa
import networkx as nx
import random
from typing import Dict, List, Tuple


class AntColonyModel(mesa.Model):
    """Ant Colony Optimization simulation model."""
    
    def __init__(
        self, 
        num_nodes: int = 21, 
        num_ants: int = 10, 
        decay_rate: float = 0.1, 
        version: int = 1, 
        min_food: int = 1, 
        max_food: int = 5, 
        seed: int = 2025 # None
    ):
        super().__init__(seed=seed)
        random.seed(seed)
        
        # Store parameters
        self.num_nodes = num_nodes
        self.num_ants = num_ants
        self.decay_rate = decay_rate
        self.version = version
        self.min_food = min_food
        self.max_food = max_food
        
        # Build graph
        self.graph, self.node_positions = self._build_graph(num_nodes)
        
        # Initialize food
        self.anthill = 0
        self.food_values = self._init_food(num_nodes)
        self.initial_food_values = self.food_values.copy()
        self.initial_food = sum(self.food_values)
        self.food_at_deposits: Dict[int, int] = {self.anthill: 0}
        self.total_food_collected = 0
        
        # Initialize edge weights
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['weight'] = 0.0
        
        # Create ants using Factory pattern
        from agents import AntFactory
        AntFactory.create_ants(self, num_ants, version)
        
        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Food Remaining": lambda m: sum(m.food_values),
                "Food Collected": lambda m: m.total_food_collected,
                "Max Edge Weight": lambda m: max(
                    (m.graph.edges[e]['weight'] for e in m.graph.edges()), default=0
                ),
            }
        )
        self.running = True
    
    def _build_graph(self, num_nodes: int) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
        """Creates a spatial graph with proximity-based edges."""
        # Generate random positions
        positions = {0: (0.5, 0.5)}  # Anthill at center
        for i in range(1, num_nodes):
            positions[i] = (random.random(), random.random())
        
        # Create graph and add edges based on distance
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        
        threshold = 0.15 + (0.1 / (num_nodes ** 0.3))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self._distance(positions[i], positions[j]) < threshold:
                    graph.add_edge(i, j)
        
        # Ensure connectivity
        self._ensure_connected(graph, positions)
        return graph, positions
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def _ensure_connected(self, graph: nx.Graph, positions: Dict[int, Tuple[float, float]]):
        """Connects disconnected components to the anthill component."""
        if nx.is_connected(graph):
            return
        
        components = list(nx.connected_components(graph))
        anthill_component = next(c for c in components if 0 in c)
        
        for component in components:
            if component == anthill_component:
                continue
            
            # Find closest pair between components
            best_pair, min_dist = None, float('inf')
            for node_a in component:
                for node_b in anthill_component:
                    dist = self._distance(positions[node_a], positions[node_b])
                    if dist < min_dist:
                        min_dist, best_pair = dist, (node_a, node_b)
            
            if best_pair:
                graph.add_edge(*best_pair)
    
    def _init_food(self, num_nodes: int) -> List[int]:
        """Initializes food values for each node."""
        return [0] + [random.randint(self.min_food, self.max_food) for _ in range(num_nodes - 1)]
    
    def step(self):
        """Advance the simulation by one step."""
        #if self.all_food_collected():
        if self.food_at_deposits.get(self.anthill, 0) >= self.initial_food:
            self.running = False
            return
        
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        self._decay_weights()
    
    def _decay_weights(self):
        """Reduces pheromone weights on all edges."""
        for u, v in self.graph.edges():
            current = self.graph.edges[u, v]['weight']
            self.graph.edges[u, v]['weight'] = max(0, current - self.decay_rate)
    
    def all_food_collected(self) -> bool:
        """Returns True if all food has been collected from nodes."""
        return sum(self.food_values) == 0
    
    def get_completion_percentage(self) -> float:
        """Returns the percentage of food collected."""
        if self.initial_food == 0:
            return 100.0
        remaining = sum(self.food_values)
        return ((self.initial_food - remaining) / self.initial_food) * 100