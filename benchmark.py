from model import AntColonyModel
import matplotlib.pyplot as plt
import numpy as np


def run_simulation(num_nodes, num_ants, use_pheromones, version, seed, decay_rate):
    model = AntColonyModel(
        num_nodes=num_nodes,
        num_ants=num_ants,
        decay_rate=decay_rate,
        version=version,
        min_food=2,
        max_food=5,
        use_pheromones=use_pheromones,
        pheromone_follow_prob=0.8,
        clustering=4,
        seed=seed
    )
    
    steps = 0
    max_steps = 10000
    
    while model.running and steps < max_steps:
        model.step()
        steps += 1
    
    return steps


def run_comprehensive_benchmark():
    graph_sizes = list(range(50, 501, 50))
    num_ants = 20
    num_runs = 10
    
    scenarios = [
        {"name": "V1 Random", "version": 1, "use_pheromones": False, "decay_rate": 1.0  },
        {"name": "V1 Pher (0.10)", "version": 1, "use_pheromones": True, "decay_rate": 0.10},
        {"name": "V1 Pher (0.25)", "version": 1, "use_pheromones": True, "decay_rate": 0.25},
        {"name": "V1 Pher (0.33)", "version": 1, "use_pheromones": True, "decay_rate": 0.33},
        {"name": "V2 Random", "version": 2, "use_pheromones": False, "decay_rate": 1.0},
        {"name": "V2 Pher (0.10)", "version": 2, "use_pheromones": True, "decay_rate": 0.10},
        {"name": "V2 Pher (0.25)", "version": 2, "use_pheromones": True, "decay_rate": 0.25},
        {"name": "V2 Pher (0.33)", "version": 2, "use_pheromones": True, "decay_rate": 0.33},
    ]
    
    results = {s["name"]: {} for s in scenarios}
    
    print("="*100)
    print("BENCHMARK \n Scenarios: Random, Pheromone (0.10/0.25/0.33) for both V1 (Direct Return) and V2 (Halfway)")
    print("="*100)
    
    for num_nodes in graph_sizes:
        print(f"\nTesting {num_nodes} nodes...")
        
        for scenario in scenarios:
            steps_list = []
            for run in range(num_runs):
                steps = run_simulation(
                    num_nodes=num_nodes,
                    num_ants=num_ants,
                    use_pheromones=scenario["use_pheromones"],
                    version=scenario["version"],
                    seed=1000 + run,
                    decay_rate=scenario["decay_rate"]
                )
                steps_list.append(steps)
            
            avg = sum(steps_list) / len(steps_list)
            results[scenario["name"]][num_nodes] = avg
    
    print_results_table(results, graph_sizes, scenarios)
    return results


def print_results_table(results, graph_sizes, scenarios):
    print("\n" + "="*120)
    print("RESULTS TABLE: Average Steps to Completion")
    print("="*120)
    
    header = f"{'Nodes':<8}"
    for s in scenarios:
        header += f"{s['name']:<12}"
    print(header)
    print("-"*120)
    
    for num_nodes in graph_sizes:
        row = f"{num_nodes:<8}"
        for s in scenarios:
            avg = results[s["name"]][num_nodes]
            row += f"{avg:<12.0f}"
        print(row)
    
    print("="*120)


def plot_results(results, graph_sizes, scenarios):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    v1_scenarios = [s for s in scenarios if s["name"].startswith("V1")]
    v2_scenarios = [s for s in scenarios if s["name"].startswith("V2")]
    
    for scenario in v1_scenarios:
        values = [results[scenario["name"]][n] for n in graph_sizes]
        ax1.plot(graph_sizes, values, label=scenario["name"])
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Steps to Completion')
    ax1.set_title('Version 1: Direct Return to Anthill')
    ax1.legend()
    ax1.grid(True)
    
    for scenario in v2_scenarios:
        values = [results[scenario["name"]][n] for n in graph_sizes]
        ax2.plot(graph_sizes, values, label=scenario["name"])
    
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Steps to Completion')
    ax2.set_title('Version 2: Halfway Deposit Strategy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'benchmark_results.png'")
    plt.show()


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    first_scenario = list(results.keys())[0]
    completed_sizes = sorted(results[first_scenario].keys())
    
    plot_results(results, completed_sizes, [
        {"name": "V1 Random", "version": 1},
        {"name": "V1 Pher (0.10)", "version": 1},
        {"name": "V1 Pher (0.25)", "version": 1},
        {"name": "V1 Pher (0.33)", "version": 1},
        {"name": "V2 Random", "version": 2},
        {"name": "V2 Pher (0.10)", "version": 2},
        {"name": "V2 Pher (0.25)", "version": 2},
        {"name": "V2 Pher (0.33)", "version": 2},
    ])
