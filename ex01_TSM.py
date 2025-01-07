import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_tsp_file(file_path):
    """
    Parses a TSPLIB format .tsp file to extract the node coordinates.
    """
    nodes = {}
    order = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        coord_section = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                coord_section = True
                continue
            if line.startswith("EOF"):
                break
            if coord_section:
                parts = line.strip().split()
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[node_id] = (x, y)
                order.append(node_id)
    return nodes, order

def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points.
    """
    return round(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

def calculate_total_distance(tour, nodes):
    """
    Calculates the total distance of the given tour.
    """
    total_distance = 0
    for i in range(len(tour)):
        total_distance += euclidean_distance(nodes[tour[i]], nodes[tour[(i + 1) % len(tour)]])
    return total_distance

def nearest_neighbor(nodes):
    """
    Generates an initial tour using the Nearest Neighbor heuristic.
    """
    unvisited = set(nodes.keys())
    current = next(iter(nodes))  # Start from the first node
    tour = [current]
    unvisited.remove(current)
    while unvisited:
        next_node = min(unvisited, key=lambda node: euclidean_distance(nodes[current], nodes[node]))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour



def swap_edges(tour, i, j):
    """
    Perform a 2-Opt swap by reversing the segment between indices i+1 and j.
    """
    i += 1
    while i < j:
        tour[i], tour[j] = tour[j], tour[i]
        i += 1
        j -= 1
def two_opt_optimized(tour, nodes, max_iterations=1000):
    """
    Optimized 2-Opt heuristic to minimize the total tour length.
    Uses input data from files as the initial tour.
    """
    n = len(tour)  # Number of nodes in the tour
    found_improvement = True
    iterations = 0
    current_length = calculate_total_distance(tour, nodes)

    while found_improvement and iterations < max_iterations:
        found_improvement = False
        for i in range(n - 1):
            for j in range(i + 2, n):  # Ensure valid 2-Opt pairs
                # Calculate the change in distance for swapping edges (i, i+1) and (j, j+1)
                length_delta = (
                    -euclidean_distance(nodes[tour[i]], nodes[tour[i + 1]])
                    - euclidean_distance(nodes[tour[j]], nodes[tour[(j + 1) % n]])
                    + euclidean_distance(nodes[tour[i]], nodes[tour[j]])
                    + euclidean_distance(nodes[tour[i + 1]], nodes[tour[(j + 1) % n]])
                )

                # If a swap reduces the total length, perform the swap
                if length_delta < 0:
                    swap_edges(tour, i, j)
                    current_length += length_delta
                    found_improvement = True

        iterations += 1

    # Return the optimized tour and its length
    final_length = calculate_total_distance(tour, nodes)
    print(f"2-Opt completed: Iterations = {iterations}, Final Distance = {final_length:.1f}")
    return tour, final_length



def create_grid(nodes, cell_size):
    """
    Creates a grid-based spatial index for the nodes.
    """
    grid = defaultdict(list)
    for node_id, (x, y) in nodes.items():
        cell = (int(x // cell_size), int(y // cell_size))
        grid[cell].append(node_id)
    return grid

def find_neighbors(grid, nodes, cell_size, x, y):
    """
    Finds neighbors of a point (x, y) in the grid.
    """
    neighbors = []
    cell_x, cell_y = int(x // cell_size), int(y // cell_size)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = (cell_x + dx, cell_y + dy)
            neighbors.extend(grid.get(cell, []))
    return neighbors

def two_opt_optimized_partial(tour, nodes, cell_size=10, max_iterations=1000):
    """
    Optimized 2-Opt heuristic using partial distance computation and grid-based locality.
    """
    grid = create_grid(nodes, cell_size)
    improved = True
    edge_swaps = 0
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if j - i == 1:  # Consecutive edges, no change
                    continue

                # Only check edges that are spatially near
                x, y = nodes[tour[i]]
                neighbors = find_neighbors(grid, nodes, cell_size, x, y)
                if tour[j] not in neighbors:
                    continue

                # Compute distances only for the affected edges
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % len(tour)]
                original_cost = euclidean_distance(nodes[a], nodes[b]) + euclidean_distance(nodes[c], nodes[d])
                new_cost = euclidean_distance(nodes[a], nodes[c]) + euclidean_distance(nodes[b], nodes[d])

                # Debugging: Log distances and swap decision
                #print(f"Checking swap: {a}-{b} with {c}-{d} | Original Cost: {original_cost}, New Cost: {new_cost}")

                # If swap reduces the cost, apply it
                if new_cost < original_cost:
                    #print(f"Applying swap between {a}-{b} and {c}-{d}")
                    tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    edge_swaps += 1
                    improved = True
        iterations += 1

    # Debugging: Log final distance
    final_distance = calculate_total_distance(tour, nodes)
    print(f"Final 2-Opt Distance: {final_distance}, Swaps: {edge_swaps}")
    return tour, edge_swaps




def run_experiments(file_path, bound):
    """
    Runs both heuristics on a given TSP instance and compares results.
    """
    nodes, input_order = parse_tsp_file(file_path)

    # Nearest-Neighbor Heuristic
    print("Running Nearest Neighbor...")
    start_time = time.time()
    nn_tour = nearest_neighbor(nodes)
    nn_time = time.time() - start_time
    nn_distance = calculate_total_distance(nn_tour, nodes)
    nn_ratio = nn_distance / bound

    # 2-Opt Using Input Order
    print("Running 2-Opt on input order...")
    start_time = time.time()
    #opt_tour_input, edge_swaps_input = two_opt_optimized_partial(input_order, nodes)
    opt_tour_input, edge_swaps_input = two_opt_optimized(input_order, nodes)
    opt_time_input = time.time() - start_time
    opt_distance_input = calculate_total_distance(opt_tour_input, nodes)
    opt_ratio_input = opt_distance_input / bound

    # 2-Opt Using NN Tour
    print("Running 2-Opt on NN tour...")
    start_time = time.time()
    #opt_tour_nn, edge_swaps_nn = two_opt_optimized_partial(nn_tour, nodes)
    opt_tour_nn, edge_swaps_nn = two_opt_optimized(nn_tour, nodes)
    opt_time_nn = time.time() - start_time
    opt_distance_nn = calculate_total_distance(opt_tour_nn, nodes)
    opt_ratio_nn = opt_distance_nn / bound

    return {
        "NN Distance": nn_distance,
        "NN Time": nn_time,
        "NN Ratio": nn_ratio,
        "2-Opt Input Distance": opt_distance_input,
        "2-Opt Input Time": opt_time_input,
        "2-Opt Input Ratio": opt_ratio_input,
        "2-Opt Input Swaps": edge_swaps_input,
        "2-Opt NN Distance": opt_distance_nn,
        "2-Opt NN Time": opt_time_nn,
        "2-Opt NN Ratio": opt_ratio_nn,
        "2-Opt NN Swaps": edge_swaps_nn,
    }

def plot_tour(nodes, tour, title):
    """
    Plots a given tour.
    """
    x = [nodes[node][0] for node in tour + [tour[0]]]  # Add start node at the end to close the loop
    y = [nodes[node][1] for node in tour + [tour[0]]]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o')
    for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
        plt.text(x_coord, y_coord, f"{tour[idx]}" if idx < len(tour) else "", fontsize=9)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig(f"{title}.png")

# Example Visualization Code
def visualize_tours(file_path):
    """
    Visualizes tours (a-e) for a given TSP instance.
    """
    # Parse data
    nodes, input_order = parse_tsp_file(file_path)

    # (a) Tour from Nearest Neighbor Heuristic
    nn_tour = nearest_neighbor(nodes)
    plot_tour(nodes, nn_tour, title=f"(a) Nearest Neighbor Tour for {file_path}")

    # (b) Initial Tour (Input Order)
    plot_tour(nodes, input_order, title=f"(b) Initial Tour (Input Order) for {file_path}")

    # (c) Intermediate Tour (Halfway through edge swaps)
    """
    opt_tour_input, edge_swaps_input = two_opt_optimized_partial(input_order, nodes)
    halfway_tour = input_order.copy()
    _, halfway_edge_swaps = two_opt_optimized_partial(input_order, nodes, max_iterations=edge_swaps_input // 2)
    plot_tour(nodes, halfway_tour, title=f"(c) Intermediate Tour for {file_path}")

    # (d) Final Tour from Input Order with 2-Opt
    plot_tour(nodes, opt_tour_input, title=f"(d) Final 2-Opt Tour (Input Order) for {file_path}")
    """
    # (e) Final Tour from NN Tour with 2-Opt
    #opt_tour_nn, _ = two_opt_optimized_partial(nn_tour, nodes)
    opt_tour_nn, _ = two_opt_optimized(nn_tour, nodes)
    plot_tour(nodes, opt_tour_nn, title=f"(e) Final 2-Opt Tour (NN Tour) for {file_path}")


# Dictionary mapping instance names to their bounds
instance_bounds = {
    "ar9152": 837479,
    "bm33708": 959011,
    "ca4663": 1290319,
    "ch71009": 4565452,
    "dj38": 6656,
    "eg7146": 172386,
    "fi10639": 520527,
    "gr9882": 300899,
    "ho14473": 177092,
    "ei8246": 206171,
    "it16862": 557315,
    "ja9847": 491924,
    "kz9976": 1061881,
    "lu980": 11340,
    "mo14185": 427377,
    "nu3496": 96132,
    "mu1979": 86891,
    "pm8079": 114855,
    "qa194": 9352,
    "rw1621": 26051,
    "sw24978": 855597,
    "tz6117": 394718,
    "uy734": 79114,
    "vm22775": 569288,
    "wi29": 27603,
    "ym7663": 238314,
    "zi929": 95345
}

def main():
    user_choice = input("type 1 for 25 results, 2 to compute and plot for 3 instances: ")
    if user_choice == "1":
        instances = {
        "dj38.tsp": 6656,
        "qa194.tsp": 9352,
        "lu980.tsp": 11340
        # Add more instances and bounds here
        }

        results = []
        for file_path, bound in instances.items():
            print(f"Running experiments on {file_path} with bound {bound}...")
            result = run_experiments(file_path, bound)
            result["Instance"] = file_path
            results.append(result)
            print(f"Results for {file_path}:", result)
            print("\n")

        # Print results in table format
        print("Instance", "NN Distance", "NN Time", "NN Ratio", 
            "2-Opt Input Distance", "2-Opt Input Time", "2-Opt Input Ratio", "2-Opt Input Swaps",
            "2-Opt NN Distance", "2-Opt NN Time", "2-Opt NN Ratio", "2-Opt NN Swaps", sep="\t")
        for r in results:
            print(r["Instance"], r["NN Distance"], f"{r['NN Time']:.4f}", f"{r['NN Ratio']:.4f}",
                r["2-Opt Input Distance"], f"{r['2-Opt Input Time']:.4f}", f"{r['2-Opt Input Ratio']:.4f}", r["2-Opt Input Swaps"],
                r["2-Opt NN Distance"], f"{r['2-Opt NN Time']:.4f}", f"{r['2-Opt NN Ratio']:.4f}", r["2-Opt NN Swaps"], sep="\t")
        
    elif user_choice == "2":
        print("Computing and plotting for 3 instances...")
        # Visualize for dj38.tsp, qa194.tsp, lu980.tsp
        for instance in ["dj38.tsp", "qa194.tsp", "lu980.tsp"]:
            print(f"Visualizing tours for {instance}...")
            visualize_tours(instance)


    else:
        user_choice = input("type 1 for 25 results, 2 to compute and plot for 3 instances: ")
    

if __name__ == "__main__":
    main()
