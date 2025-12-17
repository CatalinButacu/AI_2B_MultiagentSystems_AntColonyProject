import solara
import networkx as nx
import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_plot_component
from model import AntColonyModel
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Configure matplotlib to prevent memory leaks
plt.rcParams['figure.max_open_warning'] = 5
import matplotlib
matplotlib.use('Agg')

# Intrinsic State (shared & immutable configuration)
@dataclass(frozen=True)
class VisualizationConfig:
    color_anthill: str = '#e94560'
    color_food: str = '#098698'
    color_empty: str = '#BAB1AD'
    color_edge_strong: str = '#4ade80'
    color_edge_weak: str = '#AEC0C2'
    color_edge_none: str = '#505A63'
    max_fig_size: int = 10
    min_fig_size: int = 6
    base_node_size: int = 50

# Shared flyweight instance
shared_config = VisualizationConfig()


# Extrinsic State (model data)
def GraphVisualization(model):
    
    G = model.graph
    
    if not hasattr(model, '_viz_layout'):
        model._viz_layout = getattr(model, 'node_positions', nx.spring_layout(G, seed=42, k=1.5/len(G.nodes())**0.5, iterations=50))
    
    pos = model._viz_layout
    
    fig_size = min(shared_config.max_fig_size, max(shared_config.min_fig_size, 6 + len(G.nodes()) / 30))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    num_nodes = len(G.nodes())
    base_node_size = max(20, 800 / (1 + num_nodes * 0.02))
    
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node == model.anthill:
            node_colors.append(shared_config.color_anthill)
            node_sizes.append(base_node_size * 1.5)
        elif model.food_values[node] > 0:
            node_colors.append(shared_config.color_food)
            food_bonus = model.food_values[node] * max(2, 20 / (1 + num_nodes * 0.01))
            node_sizes.append(base_node_size + food_bonus)
        else:
            node_colors.append(shared_config.color_empty)
            node_sizes.append(base_node_size * 0.7)
            
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          ax=ax, edgecolors='white', linewidths=1.5)
    
    max_weight = max([data['weight'] for _, _, data in G.edges(data=True)] or [1])
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight > 0:
            width = min(1 + weight * 2, 8)
            alpha = min(0.3 + weight / max(max_weight, 1), 1.0)
            color = shared_config.color_edge_strong if weight > 1 else shared_config.color_edge_weak
        else:
            width = 0.5
            alpha = 0.2
            color = shared_config.color_edge_none
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                              edge_color=color, alpha=alpha, ax=ax)
    
    if len(G.nodes()) <= 50:
        node_labels = {}
        for node in G.nodes():
            if node == model.anthill:
                deposited = model.food_at_deposits.get(model.anthill, 0)
                node_labels[node] = f"HOME\n{deposited}"
            else:
                initial = model.initial_food_values[node]
                current = model.food_values[node]
                node_labels[node] = f"id{node}\n{current}/{initial}"
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)

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
            ax.text(x, y - 0.08, f"ANT x{counts['total']}", 
                   fontsize=12, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'), zorder=30)
        else:
            if counts['carrying'] > 0:
                ax.plot(x, y, marker='D', markersize=10, color='#8B4513', markeredgecolor='white', markeredgewidth=1.5, zorder=15, alpha=0.7)
            else:
                ax.plot(x, y, marker='o', markersize=8, color='black', markeredgecolor='white', markeredgewidth=1.5, zorder=15, alpha=0.7)



    ax.set_title(f"Collected: {model.total_food_collected} of {model.initial_food}", fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=shared_config.color_anthill, markersize=8, label='Anthill'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=shared_config.color_food, markersize=8, label='Food'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=shared_config.color_empty, markersize=8, label='Empty'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Ant searching'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#8B4513', markersize=5, label='Ant carrying'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
              framealpha=0.6, edgecolor='gray', 
              handlelength=1, handleheight=0.7, 
              borderpad=0.3, labelspacing=0.3)
    
    ax.axis('off')
    plt.tight_layout()    
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def MainLayout(model):
    
    with solara.Row(gap="20px", style={"width": "75vw", "height": "80vh"}):
        with solara.Column(gap="15px", style={"flex": "1", "padding": "10px"}):
            GraphVisualization(model)            
            
        with solara.Column(gap="15px", style={"flex": "1", "padding": "10px"}):
            if isinstance(plot_component, (tuple, list)):
                plot_component[0](model)
            else:
                plot_component(model)   
            
            AntPosTable(model)      


def AntPosTable(model):
    with solara.Card("Ant Activity"):
        ants = list(model.agents)
        rows = []
        ct = 4
        
        for i in range(0, len(ants), ct):
            group = ants[i:i+ct]
            cells = []
            for j, ant in enumerate(group, start=i+1):
                if ant.carrying_food:
                    status = "<span style='color:#4ade80'>Carrying</span>"
                else:
                    status = "Searching"
                if ant.previous_position is not None and ant.previous_position != ant.position:
                    prev = "H" if ant.previous_position == model.anthill else str(ant.previous_position)
                    curr = "H" if ant.position == model.anthill else str(ant.position)
                    pos = f"{prev}→{curr}"
                else:
                    pos = "H" if ant.position == model.anthill else str(ant.position)
                cells.append(f"<td style='color:#e94560'>{j}</td><td>{pos}</td><td>{status}</td>")
            
            while len(cells) < ct:
                cells.append("<td></td><td></td><td></td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        
        html = f"""
        <table style='font-size: 11px; border-collapse: collapse; width: 100%; text-align: center;'>
            <thead><tr>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>"""
        solara.HTML(tag="div", unsafe_innerHTML=html)

if __name__ == "__main__":
    
    model_params = {
        "num_nodes": {
            "type": "SliderInt",
            "value": 81,
            "label": "Number of Nodes",
            "min": 10,
            "max": 500,
            "step": 5,
        },
        "num_ants": {
            "type": "SliderInt",
            "value": 12,
            "label": "Number of Ants",
            "min": 4,
            "max": 100,
            "step": 5,
        },
        "use_pheromones": {
            "type": "Checkbox",
            "value": True,
            "label": "Use Pheromones",
        },
        "decay_rate": {
            "type": "SliderFloat",
            "value": 0.33,
            "label": "Pheromone Decay Rate",
            "min": 0.01,
            "max": 0.5,
            "step": 0.01,
        },
        "pheromone_follow_prob": {
            "type": "SliderFloat",
            "value": 0.8,
            "label": "Pheromone Follow Probability",
            "min": 0.6,
            "max": 1.0,
            "step": 0.05,
        },
        "clustering": {
            "type": "SliderInt",
            "value": 4,
            "label": "Food Clustering (0=everywhere)",
            "min": 0,
            "max": 8,
            "step": 1,
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
    
    plot_component = make_plot_component(["Food Collected"])
    initial_params = {key: param["value"] for key, param in model_params.items()}
    
    page = SolaraViz(
        model=AntColonyModel(**initial_params),
        components=[MainLayout],
        model_params=model_params,
        name="Topic 4 - Ant Colony Optimization"
        #render_interval=1
    )