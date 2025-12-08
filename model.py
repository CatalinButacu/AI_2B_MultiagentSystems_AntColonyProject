"""
Ant Colony Optimization Model
Simulates ant colony food collection with pheromone trails.
Mesa 3.x compatible implementation.
"""
import mesa
import networkx as nx
import random
from agents import AntV1, AntV2


class AntColonyModel(mesa.Model):
    """
    Ant Colony Optimization model with configurable parameters.
    
    Parameters:
        num_nodes: Number of nodes in the graph
        num_ants: Number of ant agents
        decay_rate: Pheromone decay per step (subtracted from weights)
        version: 1 = return to anthill, 2 = deposit halfway
        seed: Random seed for reproducibility
    """
    
    def __init__(self, num_nodes=15, num_ants=10, decay_rate=0.1, version=1, seed=None):
        super().__init__(seed=seed)
        
        self.num_nodes = num_nodes
        self.num_ants = num_ants
        self.decay_rate = decay_rate
        self.version = version
        
        # Create more realistic graph structure
        # For larger graphs, use a combination of strategies for realism
        if num_nodes <= 20:
            # Small graphs: Watts-Strogatz small-world
            k = min(4, num_nodes - 1)
            self.graph = nx.connected_watts_strogatz_graph(num_nodes, k, 0.3, seed=seed)
        else:
            # Large graphs: Use a more realistic approach
            # Create a spatial graph where nodes are positioned in 2D space
            # and connected based on proximity (like real ant foraging terrain)
            import random
            if seed:
                random.seed(seed)
            
            # Generate random 2D positions for ALL nodes including anthill
            positions = {}
            # Place anthill at center
            positions[0] = (0.5, 0.5)
            # Place other nodes randomly
            for i in range(1, num_nodes):
                positions[i] = (random.random(), random.random())
            
            # Create graph with spatial connections
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(num_nodes))
            
            # Connect nodes based on distance threshold
            # Adjust threshold to ensure connectivity but keep it sparse
            distance_threshold = 0.15 + (0.1 / (num_nodes ** 0.3))  # Adaptive threshold
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = ((positions[i][0] - positions[j][0])**2 + 
                           (positions[i][1] - positions[j][1])**2)**0.5
                    if dist < distance_threshold:
                        self.graph.add_edge(i, j)
            
            # Ensure graph is connected
            if not nx.is_connected(self.graph):
                # Find all components
                components = list(nx.connected_components(self.graph))
                # Connect each component to the anthill's component
                anthill_component = None
                for comp in components:
                    if 0 in comp:
                        anthill_component = comp
                        break
                
                for component in components:
                    if component != anthill_component:
                        # Find closest node in component to any node in anthill component
                        min_dist = float('inf')
                        best_pair = None
                        for node_a in component:
                            for node_b in anthill_component:
                                dist = ((positions[node_a][0] - positions[node_b][0])**2 + 
                                       (positions[node_a][1] - positions[node_b][1])**2)**0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    best_pair = (node_a, node_b)
                        if best_pair:
                            self.graph.add_edge(best_pair[0], best_pair[1])
            
            # Store positions for visualization
            self.node_positions = positions
        
        # Node 0 is the anthill
        self.anthill = 0
        
        # Initialize food values (0 for anthill, random 1-5 for others)
        self.food_values = [0] + [random.randint(1, 5) for _ in range(num_nodes - 1)]
        self.initial_food = sum(self.food_values)
        
        # Track food deposited at each node
        self.food_at_deposits = {self.anthill: 0}
        self.total_food_collected = 0
        
        # Initialize edge weights to 0
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['weight'] = 0.0
        
        # Create ants (they auto-register with model.agents)
        AntClass = AntV1 if version == 1 else AntV2
        for _ in range(num_ants):
            AntClass(self)
        
        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Food Remaining": lambda m: sum(m.food_values),
                "Food Collected": lambda m: m.total_food_collected,
                "Max Edge Weight": lambda m: max(
                    (m.graph.edges[e]['weight'] for e in m.graph.edges()), default=0
                ),
            }
        )
    
    def step(self):
        """Advance the model by one step."""
        # Collect data before step
        self.datacollector.collect(self)
        
        # All ants take action (random order)
        self.agents.shuffle_do("step")
        
        # Decay pheromone weights
        self._decay_weights()
    
    def _decay_weights(self):
        """Reduce edge weights by decay rate (minimum 0)."""
        for u, v in self.graph.edges():
            current = self.graph.edges[u, v]['weight']
            self.graph.edges[u, v]['weight'] = max(0, current - self.decay_rate)
    
    def all_food_collected(self):
        """Check if all food has been collected from nodes."""
        return sum(self.food_values) == 0
    
    def get_completion_percentage(self):
        """Return percentage of food collected."""
        if self.initial_food == 0:
            return 100.0
        remaining = sum(self.food_values)
        return ((self.initial_food - remaining) / self.initial_food) * 100