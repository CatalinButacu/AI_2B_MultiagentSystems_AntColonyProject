import mesa
import networkx as nx
import random
from agents import AntV1, AntV2


class AntColonyModel(mesa.Model):
    
    def __init__(self, num_nodes=15, num_ants=10, decay_rate=0.1, version=1, min_food=1, max_food=5, seed=None):
        super().__init__(seed=seed)        
        random.seed(seed)

        self.num_nodes = num_nodes
        self.num_ants = num_ants
        self.decay_rate = decay_rate
        self.version = version
        self.min_food = min_food
        self.max_food = max_food
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_nodes))
        
        distance_threshold = 0.15 + (0.1 / (num_nodes ** 0.3))  
        positions = {}
        positions[0] = (0.5, 0.5)
        for i in range(1, num_nodes):
            positions[i] = (random.random(), random.random())   
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = ((positions[i][0] - positions[j][0])**2 + 
                        (positions[i][1] - positions[j][1])**2)**0.5
                if dist < distance_threshold:
                    self.graph.add_edge(i, j)
        
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            anthill_component = None
            for comp in components:
                if 0 in comp:
                    anthill_component = comp
                    break
            
            for component in components:
                if component != anthill_component:
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
        
        self.node_positions = positions            
        self.anthill = 0        
        self.food_values = [0] + [random.randint(self.min_food, self.max_food) for _ in range(num_nodes - 1)]
        self.initial_food_values = self.food_values.copy()
        self.initial_food = sum(self.food_values)        
        self.food_at_deposits = {self.anthill: 0}
        self.total_food_collected = 0
        
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['weight'] = 0.0
        
        AntClass = AntV1 if version == 1 else AntV2
        for _ in range(num_ants):
            AntClass(self)
        
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
    
    def step(self):
        if self.food_at_deposits.get(self.anthill, 0) >= self.initial_food:
            self.running = False
            return

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)     
        self._decay_weights()

    
    def _decay_weights(self):
        for u, v in self.graph.edges():
            current = self.graph.edges[u, v]['weight']
            self.graph.edges[u, v]['weight'] = max(0, current - self.decay_rate)
    
    def all_food_collected(self):
        return sum(self.food_values) == 0
    
    def get_completion_percentage(self):
        if self.initial_food == 0:
            return 100.0
        remaining = sum(self.food_values)
        return ((self.initial_food - remaining) / self.initial_food) * 100