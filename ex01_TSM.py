import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import csv

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

def two_opt_optimized(tour, nodes, max_iterations=1000, time_limit=600):
    """
    Optimized 2-Opt heuristic to minimize the total tour length.
    Stops execution after a specified time limit.
    """
    tour = tour.copy()  # Create a copy of the input tour to avoid modifying the original
    n = len(tour)  # Number of nodes in the tour
    found_improvement = True
    iterations = 0
    current_length = calculate_total_distance(tour, nodes)
    start_time = time.time()
    count_edgeswap = 0
    while found_improvement and iterations < max_iterations:
        if time.time() - start_time > time_limit:
            print(f"Time limit of {time_limit} seconds reached for 2-Opt.")
            break

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
                    count_edgeswap += 1
                    current_length += length_delta
                    found_improvement = True
        iterations += 1

    # Return the optimized tour and its length
    final_length = calculate_total_distance(tour, nodes)
    print(f"2-Opt completed: Iterations = {iterations}, Final Distance = {final_length:.1f}")
    return tour, final_length, count_edgeswap

def two_opt_optimized_intermediate(tour, nodes, max_edgeswap = 1000, time_limit=600):
    """
    Optimized 2-Opt heuristic to minimize the total tour length.
    Stops execution after a specified time limit.
    """
    tour = tour.copy()  # Create a copy of the input tour to avoid modifying the original
    n = len(tour)  # Number of nodes in the tour
    found_improvement = True
    iterations = 0
    current_length = calculate_total_distance(tour, nodes)
    start_time = time.time()
    count_edgeswap = 0
    while found_improvement:
        if time.time() - start_time > time_limit:
            print(f"Time limit of {time_limit} seconds reached for 2-Opt.")
            break
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
                    count_edgeswap += 1
                    current_length += length_delta
                    found_improvement = True
                    if count_edgeswap >= max_edgeswap:  # Stop after a certain number of edge swaps
                        inter_length = calculate_total_distance(tour, nodes)
                        return tour, inter_length, count_edgeswap 
        iterations += 1

    # Return the optimized tour and its length
    final_length = calculate_total_distance(tour, nodes)
    print(f"2-Opt completed: Iterations = {iterations}, Final Distance = {final_length:.1f}")
    return tour, final_length, count_edgeswap



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
    opt_tour_input, edge_swaps_input = two_opt_optimized(input_order, nodes, time_limit=600)
    opt_time_input = time.time() - start_time
    opt_distance_input = calculate_total_distance(opt_tour_input, nodes)
    opt_ratio_input = opt_distance_input / bound

    # 2-Opt Using NN Tour
    print("Running 2-Opt on NN tour...")
    start_time = time.time()
    opt_tour_nn, edge_swaps_nn = two_opt_optimized(nn_tour, nodes, time_limit=600)
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

def plot_tour(nodes, tour, instance_name, tour_type):
    """
    Plots a given tour and saves the figure with a valid filename in the /plots directory.
    """
    title = f"{instance_name} - {tour_type}"  # Create a title using instance name and tour type

    x = [nodes[node][0] for node in tour + [tour[0]]]  # Add start node at the end to close the loop
    y = [nodes[node][1] for node in tour + [tour[0]]]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    # Create the /plots directory if it doesn't exist
    plots_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the plot in the /plots directory with a simplified filename
    plt.savefig(os.path.join(plots_directory, f"{instance_name}_{tour_type}.png"))  # Save under /plots
    plt.close()  # Close the figure to free up memory




def visualize_tours(file_path):
    """
    Visualizes tours (a-e) for a given TSP instance.
    """
    # Parse data
    nodes, input_order = parse_tsp_file(file_path)
    instance_name = os.path.basename(file_path).replace('.tsp', '')  # Get the instance name without extensionnodes, input_order = parse_tsp_file(file_path)
    # (a) Tour from Nearest Neighbor Heuristic
    nn_tour = nearest_neighbor(nodes)
    plot_tour(nodes, nn_tour, instance_name, "Nearest Neighbor Tour")

    # (b) Initial Tour (Input Order)
    plot_tour(nodes, input_order, instance_name, "Initial Tour")

    # (c) Intermediate Tour (Halfway through edge swaps)
    opt_tour_input, _,edge_swaps_input = two_opt_optimized(input_order, nodes)
    print(f"Edge swaps for full tour: {edge_swaps_input}")
    halfway_tour, _ ,edge_swaps_inter= two_opt_optimized_intermediate(input_order, nodes, max_edgeswap=edge_swaps_input //2)

    print(f"Edge swaps for intermediate tour: {edge_swaps_inter}")
    plot_tour(nodes, halfway_tour, instance_name, "Intermediate Tour")

    # (d) Final Tour from Input Order with 2-Opt
    plot_tour(nodes, opt_tour_input, instance_name, "Final Tour (Input Order)")

    # (e) Final Tour from NN Tour with 2-Opt
    opt_tour_nn, _ ,_= two_opt_optimized(nn_tour, nodes)
    plot_tour(nodes, opt_tour_nn, instance_name, "Final Tour (NN Tour)")


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
        
        input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')
        tsp_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.tsp')]
        
        if not tsp_files:
            print("No .tsp files found in the /inputs directory.")
            return

        # Match files with bounds
        instances = {}
        for file_path in tsp_files:
            instance_name = os.path.splitext(os.path.basename(file_path))[0]
            if instance_name in instance_bounds:
                instances[file_path] = instance_bounds[instance_name]
            else:
                print(f"Warning: No bound found for {instance_name}. Skipping this file.")

        if not instances:
            print("No valid TSP instances found.")
            return

        results = []
        for file_path, bound in instances.items():
            print(f"Running experiments {instance_name} with bound {bound}...")
            result = run_experiments(file_path, bound)
            result["Instance"] = os.path.basename(file_path)
            results.append(result)
            print(f"Results for {file_path}:", result)
            print("\n")

        # Save results to a CSV file in the current directory
        output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
        with open(output_csv, mode='w', newline='') as csvfile:
            fieldnames = [
                "Instance", "NN Distance", "NN Time", "NN Ratio",
                "2-Opt Input Distance", "2-Opt Input Time", "2-Opt Input Ratio", "2-Opt Input Swaps",
                "2-Opt NN Distance", "2-Opt NN Time", "2-Opt NN Ratio", "2-Opt NN Swaps"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_csv}")

    elif user_choice == "2":
        print("Computing and plotting for 3 instances...")
        # Visualize for dj38.tsp, qa194.tsp, lu980.tsp
        input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')  # Ensure this is defined
        for instance in ["dj38.tsp", "qa194.tsp", "lu980.tsp"]:
            instance_path = os.path.join(input_directory, instance)  # Create full path
            print(f"Visualizing tours for {instance_path}...")
            visualize_tours(instance_path)  # Pass the full path

    else:
        user_choice = input("type 1 for 25 results, 2 to compute and plot for 3 instances: ")


if __name__ == "__main__":
    main()
