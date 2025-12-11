"""
Ant Colony Optimization - Solara Visualization
Refactored to use Mesa's SolaraViz.
"""
import solara
import networkx as nx
import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_plot_component
from model import AntColonyModel

def GraphVisualization(model):
    """
    Visualize the ant colony graph.
    This component receives the 'model' instance from SolaraViz.
    """
    # SolaraViz passes the model instance. 
    # If the simulation hasn't started or model is resetting, it might be None or in valid state.
    # However, SolaraViz typically ensures model exists.
    
    # We need a proper figure.
    # Note: In SolaraViz, this function renders reactively.
    
    G = model.graph
    
    # Graph Layout Management
    # Ideally, we want to persist layout across steps for the same model instance.
    # We can attach it to the model if it's not already there.
    if not hasattr(model, '_viz_layout'):
        if hasattr(model, 'node_positions'):
            model._viz_layout = model.node_positions
        else:
            model._viz_layout = nx.spring_layout(G, seed=42, k=1.5/len(G.nodes())**0.5, iterations=50)
    
    pos = model._viz_layout
    
    # Adjust figure size based on graph size
    fig_size = min(12, max(8, 6 + len(G.nodes()) / 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Draw nodes
    base_node_size = max(100, 800 - len(G.nodes()) * 3)
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node == model.anthill:
            node_colors.append('#e94560')  # Red for anthill
            node_sizes.append(base_node_size * 1.5)
        elif model.food_values[node] > 0:
            node_colors.append('#098698')  # Teal for food
            node_sizes.append(base_node_size + model.food_values[node] * 20)
        else:
            node_colors.append('#BAB1AD')  # Gray
            node_sizes.append(base_node_size * 0.7)
            
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          ax=ax, edgecolors='white', linewidths=1.5)
    
    # Draw edges
    max_weight = max([data['weight'] for _, _, data in G.edges(data=True)] or [1])
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight > 0:
            width = min(1 + weight * 2, 8)
            alpha = min(0.3 + weight / max(max_weight, 1), 1.0)
            color = '#4ade80' if weight > 1 else '#AEC0C2'
        else:
            width = 0.5
            alpha = 0.2
            color = '#505A63'
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                              edge_color=color, alpha=alpha, ax=ax)
    
    # Labels
    if len(G.nodes()) <= 50:
        node_labels = {}
        for node in G.nodes():
            if node == model.anthill:
                deposited = model.food_at_deposits.get(model.anthill, 0)
                node_labels[node] = f"🏠\n{deposited}"
            elif model.food_values[node] > 0:
                node_labels[node] = (f"{node}\n{'🍪' * min(model.food_values[node], 3)}" 
                                   if len(G.nodes()) <= 30 else str(model.food_values[node]))
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)

    # Draw Ants
    ant_positions = {}
    for ant in model.agents:
        if ant.position not in ant_positions:
            ant_positions[ant.position] = {'total': 0, 'carrying': 0}
        ant_positions[ant.position]['total'] += 1
        if ant.carrying_food:
            ant_positions[ant.position]['carrying'] += 1
            
    for position, counts in ant_positions.items():
        if position not in pos: continue
        x, y = pos[position]
        
        if position == model.anthill:
            ax.text(x, y - 0.08, f"🐜 {counts['total']}", 
                   fontsize=12, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'), zorder=20)
        else:
            # Simple marker for ants
            if counts['carrying'] > 0:
                # Carrying food: Diamond shape, Brown color, Large
                ax.plot(x, y, marker='D', markersize=10, color='#8B4513', markeredgecolor='white', markeredgewidth=1.5, zorder=15)
            else:
                # Searching: Circle, Black, Large
                ax.plot(x, y, marker='o', markersize=8, color='black', markeredgecolor='white', markeredgewidth=1.5, zorder=15)

    ax.set_title(f"Steps: {model.steps} | Food Collected: {model.total_food_collected}/{model.initial_food}", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    
    solara.FigureMatplotlib(fig)

def Statistics(model):
    """Custom statistics component."""
    with solara.Card("Status"):
        solara.Markdown(f"**Step**: {model.steps}")
        solara.Markdown(f"**Food Remaining**: {sum(model.food_values)}")
        solara.Markdown(f"**Collected**: {model.total_food_collected}")
        
def MainLayout(model):
    """
    Custom layout to place Graph on the left and Charts/Stats on the right.
    """
    with solara.Row(classes=["d-flex", "flex-row", "w-100"]): # Flex row for side-by-side
        with solara.Column(style={"width": "65%", "padding-right": "20px"}):
            GraphVisualization(model)
            
        with solara.Column(style={"width": "35%"}):
            Statistics(model)
            # Render the plot component
            # make_plot_component might return a tuple or list
            if isinstance(plot_component, (tuple, list)):
                plot_component[0](model)
            else:
                plot_component(model)

if __name__ == "__main__":
    # Define Model Parameters using the structure required by SolaraViz
    model_params = {
        "num_nodes": {
            "type": "SliderInt",
            "value": 30,
            "label": "Number of Nodes",
            "min": 10,
            "max": 100,
            "step": 5,
        },
        "num_ants": {
            "type": "SliderInt",
            "value": 15,
            "label": "Number of Ants",
            "min": 5,
            "max": 50,
            "step": 1,
        },
        "decay_rate": {
            "type": "SliderFloat",
            "value": 0.1,
            "label": "Pheromone Decay Rate",
            "min": 0.01,
            "max": 0.5,
            "step": 0.01,
        },
        "version": {
            "type": "Select",
            "value": 1,
            "values": [1, 2],
            "label": "Ant Version (1=Return, 2=Halfway)",
        },
        "min_food": {
            "type": "SliderInt",
            "value": 1,
            "label": "Min Food per Node",
            "min": 0,
            "max": 10,
            "step": 1,
        },
        "max_food": {
            "type": "SliderInt",
            "value": 5,
            "label": "Max Food per Node",
            "min": 1,
            "max": 20,
            "step": 1,
        },
    }

    # Instantiate layout
    # 1. Plot component (Food Collected over time)
    plot_component = make_plot_component(["Food Collected"])

    # 2. Main Page
    page = SolaraViz(
        model=AntColonyModel(),
        components=[MainLayout], # Pass our custom layout wrapper
        model_params=model_params,
        name="Ant Colony Optimization"
    )