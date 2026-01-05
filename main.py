import solara
import networkx as nx
import matplotlib.pyplot as plt
from mesa.visualization import SolaraViz, make_plot_component
from model import AntColonyModel
from dataclasses import dataclass
from matplotlib.lines import Line2D

plt.rcParams['figure.max_open_warning'] = 5
import matplotlib
matplotlib.use('Agg')


@dataclass(frozen=True)
class VisualizationConfig:
    anthill_color: str = '#2c3e50'
    food_node_color: str = '#27ae60'
    empty_node_color: str = '#bdc3c7'
    strong_pheromone_color: str = '#3498db'
    weak_pheromone_color: str = '#95a5a6'
    no_pheromone_color: str = '#ecf0f1'
    ant_searching_color: str = '#34495e'
    ant_carrying_color: str = '#e74c3c'
    maximum_figure_size: int = 10
    minimum_figure_size: int = 6
    base_node_size: int = 30


visualization_config = VisualizationConfig()


def get_node_layout(model, graph):
    if not hasattr(model, 'cached_layout'):
        model.cached_layout = getattr(
            model, 
            'node_positions', 
            nx.spring_layout(graph, seed=42, k=1.5/len(graph.nodes())**0.5, iterations=50)
        )
    return model.cached_layout


def calculate_node_appearance(model, node, total_nodes, calculated_node_size):
    if node == model.anthill_node:
        return visualization_config.anthill_color, calculated_node_size * 1.5
    
    if model.food_at_nodes[node] > 0:
        food_size_bonus = model.food_at_nodes[node] * max(2, 20 / (1 + total_nodes * 0.01))
        return visualization_config.food_node_color, calculated_node_size + food_size_bonus
    
    return visualization_config.empty_node_color, calculated_node_size * 0.7


def get_edge_style(edge_weight, maximum_edge_weight):
    if edge_weight > 0:
        edge_width = min(0.5 + edge_weight * 0.5, 3.0)
        edge_alpha = min(0.4 + edge_weight / max(maximum_edge_weight, 1) * 0.4, 0.8)
        edge_color = (
            visualization_config.strong_pheromone_color 
            if edge_weight > 1 
            else visualization_config.weak_pheromone_color
        )
    else:
        edge_width = 0.5
        edge_alpha = 0.35
        edge_color = '#95a5a6'
    
    return edge_width, edge_alpha, edge_color


def draw_graph_edges(graph, node_positions, axes):
    all_edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    maximum_edge_weight = max(all_edge_weights) if all_edge_weights else 1
    
    for source_node, target_node, edge_data in graph.edges(data=True):
        edge_width, edge_alpha, edge_color = get_edge_style(edge_data['weight'], maximum_edge_weight)
        
        nx.draw_networkx_edges(
            graph, 
            node_positions, 
            edgelist=[(source_node, target_node)], 
            width=edge_width, 
            edge_color=edge_color, 
            alpha=edge_alpha, 
            ax=axes
        )


def build_node_labels(model, graph):
    node_labels = {}
    for node in graph.nodes():
        if node == model.anthill_node:
            food_deposited = model.food_deposited_at_nodes.get(model.anthill_node, 0)
            node_labels[node] = f"HOME\n{food_deposited}"
        else:
            initial_food = model.initial_food_at_nodes[node]
            current_food = model.food_at_nodes[node]
            node_labels[node] = f"id{node}\n{current_food}/{initial_food}"
    return node_labels


def count_ants_at_nodes(model):
    ants_at_each_node = {}
    for ant in model.agents:
        if ant.current_node not in ants_at_each_node:
            ants_at_each_node[ant.current_node] = {'total': 0, 'carrying': 0}
        ants_at_each_node[ant.current_node]['total'] += 1
        if ant.is_carrying_food:
            ants_at_each_node[ant.current_node]['carrying'] += 1
    return ants_at_each_node


def draw_ant_markers(model, node_positions, axes, ants_at_each_node):
    for node, ant_counts in ants_at_each_node.items():
        if node not in node_positions: 
            continue
        x_position, y_position = node_positions[node]
        
        if node == model.anthill_node:
            axes.text(
                x_position, 
                y_position - 0.06, 
                f"{ant_counts['total']}", 
                fontsize=9, 
                ha='center', 
                va='center', 
                fontweight='normal',
                bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '#95a5a6', 'linewidth': 0.5}, 
                zorder=30
            )
        elif ant_counts['carrying'] > 0:
            axes.plot(
                x_position, y_position, marker='o', markersize=4, 
                color=visualization_config.ant_carrying_color, markeredgecolor='none', zorder=15, alpha=0.8
            )
        else:
            axes.plot(
                x_position, y_position, marker='o', markersize=3, 
                color=visualization_config.ant_searching_color, markeredgecolor='none', zorder=15, alpha=0.7
            )


def create_legend_elements():
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visualization_config.anthill_color, markersize=6, label='Anthill', markeredgecolor='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visualization_config.food_node_color, markersize=6, label='Food', markeredgecolor='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visualization_config.empty_node_color, markersize=6, label='Empty', markeredgecolor='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visualization_config.ant_searching_color, markersize=4, label='Searching', markeredgecolor='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=visualization_config.ant_carrying_color, markersize=4, label='Carrying', markeredgecolor='none'),
    ]


def graph_visualization(model):
    graph = model.graph
    node_positions = get_node_layout(model, graph)
    
    figure_size = min(
        visualization_config.maximum_figure_size, 
        max(visualization_config.minimum_figure_size, 6 + len(graph.nodes()) / 30)
    )
    figure, axes = plt.subplots(figsize=(figure_size, figure_size))
    
    total_nodes = len(graph.nodes())
    calculated_node_size = max(8, 400 / (1 + total_nodes * 0.04))
    
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        color, size = calculate_node_appearance(model, node, total_nodes, calculated_node_size)
        node_colors.append(color)
        node_sizes.append(size)
            
    nx.draw_networkx_nodes(
        graph, node_positions, node_size=node_sizes, node_color=node_colors, 
        ax=axes, edgecolors='#34495e', linewidths=0.5, alpha=0.9
    )
    
    draw_graph_edges(graph, node_positions, axes)
    
    if len(graph.nodes()) <= 30:
        node_labels = build_node_labels(model, graph)
        nx.draw_networkx_labels(graph, node_positions, labels=node_labels, 
                               font_size=7, font_weight='normal', font_family='sans-serif', ax=axes)

    ants_at_each_node = count_ants_at_nodes(model)
    draw_ant_markers(model, node_positions, axes, ants_at_each_node)

    axes.set_title(f"Food Collected: {model.total_food_collected} / {model.total_initial_food}", 
                   fontsize=10, fontweight='normal', color='#2c3e50', pad=10)
    axes.legend(
        handles=create_legend_elements(), loc='upper right', fontsize=7, 
        framealpha=0.9, edgecolor='#ecf0f1', fancybox=False,
        handlelength=1.2, handleheight=0.8, 
        borderpad=0.4, labelspacing=0.4
    )
    
    axes.axis('off')
    plt.tight_layout()    
    solara.FigureMatplotlib(figure)
    plt.close(figure)


def main_layout(model):
    with solara.Row(gap="20px", style={"width": "75vw", "height": "80vh"}):
        with solara.Column(gap="15px", style={"flex": "1", "padding": "10px"}):
            graph_visualization(model)            
            
        with solara.Column(gap="15px", style={"flex": "1", "padding": "10px"}):
            if isinstance(food_collection_plot, (tuple, list)):
                food_collection_plot[0](model)
            else:
                food_collection_plot(model)   
            
            ant_activity_table(model)


def format_ant_position(ant, model):
    if ant.previous_node is not None and ant.previous_node != ant.current_node:
        previous_label = "H" if ant.previous_node == model.anthill_node else str(ant.previous_node)
        current_label = "H" if ant.current_node == model.anthill_node else str(ant.current_node)
        return f"{previous_label}→{current_label}"
    return "H" if ant.current_node == model.anthill_node else str(ant.current_node)


def format_ant_status(ant):
    if ant.is_carrying_food:
        return "<span style='color:#4ade80'>Carrying</span>"
    return "Searching"


def build_table_row(ants_in_row, row_start, model, ants_per_row):
    table_cells = []
    for ant_index, ant in enumerate(ants_in_row, start=row_start + 1):
        status_text = format_ant_status(ant)
        position_text = format_ant_position(ant, model)
        table_cells.append(
            f"<td style='color:#e94560'>{ant_index}</td>"
            f"<td>{position_text}</td>"
            f"<td>{status_text}</td>"
        )
    
    while len(table_cells) < ants_per_row:
        table_cells.append("<td></td><td></td><td></td>")
    
    return f"<tr>{''.join(table_cells)}</tr>"


def ant_activity_table(model):
    with solara.Card("Ant Activity"):
        all_ants = list(model.agents)
        table_rows = []
        ants_per_row = 4
        
        for row_start in range(0, len(all_ants), ants_per_row):
            ants_in_row = all_ants[row_start:row_start + ants_per_row]
            table_rows.append(build_table_row(ants_in_row, row_start, model, ants_per_row))
        
        table_html = f"""
        <table style='font-size: 11px; border-collapse: collapse; width: 100%; text-align: center;'>
            <thead><tr>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
                <th>Ant</th><th>Tr</th><th>St</th>
            </tr></thead>
            <tbody>{''.join(table_rows)}</tbody>
        </table>"""
        solara.HTML(tag="div", unsafe_innerHTML=table_html)


if __name__ == "__main__":
    
    model_parameters = {
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
    
    food_collection_plot = make_plot_component(["Food Collected"])
    initial_parameter_values = {key: param["value"] for key, param in model_parameters.items()}
    
    page = SolaraViz(
        model=AntColonyModel(**initial_parameter_values),
        components=[main_layout],
        model_params=model_parameters,
        name="Topic 4 - Ant Colony Optimization"
    )