"""
Ant Colony Optimization - Solara Visualization
Simple and functional dashboard for ACO simulation.
"""
import solara
import networkx as nx
import matplotlib.pyplot as plt
from model import AntColonyModel
import threading

# Reactive parameters
num_nodes = solara.reactive(50)  # Increased default for more complexity
num_ants = solara.reactive(20)   # More ants for larger graphs
decay_rate = solara.reactive(0.1)
version = solara.reactive(1)
running = solara.reactive(False)
model = solara.reactive(None)
steps = solara.reactive(0)
step_delay = solara.reactive(0.2)  # Faster stepping for larger graphs
graph_layout = solara.reactive(None)  # Cache graph layout to prevent jumping

def initialize_model():
    """Initialize a new model with current parameters"""
    model.value = AntColonyModel(num_nodes.value, num_ants.value, decay_rate.value, version.value)
    steps.value = 0
    running.value = False
    graph_layout.value = None  # Reset layout for new graph

def single_step():
    """Execute a single simulation step"""
    if model.value and not model.value.all_food_collected():
        model.value.step()
        steps.value += 1

@solara.component
def ControlPanel():
    """Control panel for simulation parameters"""
    solara.Markdown("## 🐜 Ant Colony Optimization")
    
    with solara.Column():
        solara.SliderInt("Number of Nodes", value=num_nodes, min=10, max=200, disabled=running.value)
        solara.SliderInt("Number of Ants", value=num_ants, min=5, max=100, disabled=running.value)
        solara.SliderFloat("Decay Rate", value=decay_rate, min=0.01, max=0.5, step=0.01)
        solara.SliderFloat("Step Delay (s)", value=step_delay, min=0.05, max=2.0, step=0.05)
        
        with solara.Row():
            solara.Button("Version 1", color="primary" if version.value == 1 else None, 
                         on_click=lambda: version.set(1), disabled=running.value)
            solara.Button("Version 2", color="primary" if version.value == 2 else None, 
                         on_click=lambda: version.set(2), disabled=running.value)
        
        solara.Markdown(f"**Version {version.value}:** " + 
                       ("Return to anthill" if version.value == 1 else "Deposit halfway"))
        
        with solara.Row():
            solara.Button(label="🔄 Initialize", on_click=initialize_model, 
                         color="success", disabled=running.value)
            if model.value:
                solara.Button(label="▶ Start" if not running.value else "⏸ Pause", 
                            on_click=lambda: running.set(not running.value),
                            color="primary" if not running.value else "warning")
                solara.Button(label="⏭ Step", on_click=single_step, 
                            disabled=running.value or (model.value and model.value.all_food_collected()))
        
        if model.value:
            total_food = sum(model.value.food_values)
            solara.Markdown(f"### Steps: {steps.value}")
            solara.Markdown(f"**Food Remaining:** {total_food} units")
            solara.Markdown(f"**Collected:** {model.value.total_food_collected} units")
            
            if model.value.all_food_collected():
                solara.Success(f"✅ All food collected in {steps.value} steps!")

@solara.component
def GraphVisualization():
    """Visualize the ant colony graph"""
    if model.value is None:
        solara.Info("Click 'Initialize' to start the simulation.")
        return
    
    G = model.value.graph
    
    # Use cached layout or create new one
    if graph_layout.value is None:
        # For large graphs with spatial positions, use those
        if hasattr(model.value, 'node_positions'):
            graph_layout.value = model.value.node_positions
        else:
            # For small graphs, use spring layout
            graph_layout.value = nx.spring_layout(G, seed=42, k=1.5/len(G.nodes())**0.5, iterations=50)
    pos = graph_layout.value
    
    # Adjust figure size based on graph size
    fig_size = min(16, max(10, 8 + len(G.nodes()) / 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Draw nodes with different colors
    # Adaptive node size based on graph size
    base_node_size = max(100, 800 - len(G.nodes()) * 3)
    
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == model.value.anthill:
            node_colors.append('#e94560')  # Red for anthill
            node_sizes.append(base_node_size * 1.5)  # Anthill larger
        elif model.value.food_values[node] > 0:
            node_colors.append('#098698')  # Teal for food nodes
            node_sizes.append(base_node_size + model.value.food_values[node] * 20)
        else:
            node_colors.append('#BAB1AD')  # Gray for empty nodes
            node_sizes.append(base_node_size * 0.7)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          ax=ax, edgecolors='white', linewidths=1.5 if len(G.nodes()) > 50 else 2)
    
    # Draw edges with varying thickness and color based on weight
    max_weight = max([data['weight'] for _, _, data in G.edges(data=True)] or [1])
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight > 0:
            width = min(1 + weight * 2, 10)  # Scale width, cap at 10
            alpha = min(0.3 + weight / max(max_weight, 1), 1.0)
            color = '#4ade80' if weight > 1 else '#AEC0C2'  # Green for high weight
        else:
            width = 0.5
            alpha = 0.2
            color = '#505A63'
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                              edge_color=color, alpha=alpha, ax=ax)
    
    # Draw node labels with food values (only for smaller graphs or important nodes)
    show_all_labels = len(G.nodes()) <= 50
    node_labels = {}
    for node in G.nodes():
        if node == model.value.anthill:
            deposited = model.value.food_at_deposits.get(model.value.anthill, 0)
            node_labels[node] = f"🏠\n{deposited}"
        elif show_all_labels or model.value.food_values[node] > 0:
            food = model.value.food_values[node]
            if food > 0:
                node_labels[node] = f"{node}\n{'🍪' * min(food, 3)}" if len(G.nodes()) <= 30 else f"{food}"
            elif show_all_labels:
                node_labels[node] = f"{node}"
    
    label_font_size = max(6, 10 - len(G.nodes()) / 20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=label_font_size, 
                           font_weight='bold', ax=ax)
    
    # Draw edge weights only for significant edges and smaller graphs
    if len(G.nodes()) <= 100:
        edge_labels = {(u, v): f"{data['weight']:.1f}" 
                       for u, v, data in G.edges(data=True) if data['weight'] > 0.5}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=max(6, 8 - len(G.nodes()) / 30), ax=ax)
    
    # Draw ants at their current positions
    # Count ants at each position
    ant_positions = {}
    for ant in model.value.agents:
        if ant.position not in ant_positions:
            ant_positions[ant.position] = {'total': 0, 'carrying': 0}
        ant_positions[ant.position]['total'] += 1
        if ant.carrying_food:
            ant_positions[ant.position]['carrying'] += 1
    
    ant_marker_size = max(6, 12 - len(G.nodes()) / 30)
    
    for position, counts in ant_positions.items():
        if position not in pos:
            continue
            
        x, y = pos[position]
        
        # For anthill, show count as text instead of individual markers
        if position == model.value.anthill:
            ax.text(x, y - 0.08, f"🐜 {counts['total']}", 
                   fontsize=max(10, 14 - len(G.nodes()) / 30),
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='black', alpha=0.8),
                   zorder=15)
        else:
            # For other nodes, show individual ant markers (up to a limit)
            ants_at_node = [ant for ant in model.value.agents if ant.position == position]
            display_limit = min(len(ants_at_node), 5)  # Show max 5 individual ants
            
            for i, ant in enumerate(ants_at_node[:display_limit]):
                offset_scale = 0.02 if len(G.nodes()) > 50 else 0.05
                offset = offset_scale * (i - display_limit / 2)
                color = '#8B4513' if ant.carrying_food else '#000000'
                marker = 'D' if ant.carrying_food else 'o'
                ax.plot(x + offset, y + offset, marker=marker, markersize=ant_marker_size, 
                       color=color, markeredgecolor='white', markeredgewidth=1, zorder=10)
            
            # If more than display limit, show count
            if len(ants_at_node) > display_limit:
                ax.text(x, y - 0.06, f"+{len(ants_at_node) - display_limit}", 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                       zorder=11)
    
    ax.set_title(f"Ant Colony Optimization - Version {model.value.version}\n"
                f"Step {steps.value} | Decay Rate: {model.value.decay_rate}",
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    solara.FigureMatplotlib(fig)

@solara.component
def Statistics():
    """Display statistics about the simulation"""
    if model.value is None:
        return
    
    with solara.Card("📊 Statistics"):
        ants_carrying = sum(1 for ant in model.value.agents if ant.carrying_food)
        
        solara.Markdown(f"- **Ants carrying food:** {ants_carrying}/{len(model.value.agents)}")
        solara.Markdown(f"- **Food remaining:** {sum(model.value.food_values)}")
        solara.Markdown(f"- **Total nodes:** {model.value.num_nodes}")

@solara.component
def AutoStepper():
    """Handle automatic stepping when running."""
    
    def do_step():
        if model.value and not model.value.all_food_collected():
            model.value.step()
            steps.value += 1
    
    def schedule_step():
        if running.value and model.value and not model.value.all_food_collected():
            do_step()
            # Schedule next step
            timer = threading.Timer(step_delay.value, schedule_step)
            timer.daemon = True
            timer.start()
    
    def start_auto_stepping():
        if running.value and model.value and not model.value.all_food_collected():
            schedule_step()
        return lambda: None  # Cleanup function
    
    # Use effect to start/stop auto-stepping
    solara.use_effect(start_auto_stepping, dependencies=[running.value])

@solara.component
def Page():
    """Main page component"""
    
    AutoStepper()  # Render the AutoStepper component to activate its effects
    
    with solara.Row():
        with solara.Column(style={"width": "30%"}):
            ControlPanel()
            Statistics()
        with solara.Column(style={"width": "70%"}):
            GraphVisualization()