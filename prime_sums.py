
import argparse
import os
import hashlib
import networkx as nx
import numpy as np
import sympy

def generate_prime_graph(primes):
    """
    Generate the set of edge pairs and adjacency matrix based on the given prime number set.
    
    Args:
        primes (set): A set of prime numbers including 1.
    
    Returns:
        edges (set): A dictionary with edge type keys ('red', 'blue') and sets of directed edge pairs.
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
        in_degrees (dict): A dictionary mapping each prime to its in-degree.
        out_degrees (dict): A dictionary mapping each prime to its out-degree.
        edge_labels (dict): A dictionary mapping each edge to a label string representing its sum.
    """
    # Initialize the directed graph
    G = nx.DiGraph()
    G.add_nodes_from(primes)
    
    # Compute edges based on given rules
    edges = {'red': set(), 'blue': set()}
    edge_labels = {}
    primes_without_one = primes - {1}
    prime_list = sorted(primes_without_one)
    
    for p in prime_list:
        for q in prime_list:
            for r in prime_list:
                if q < r and q + r == p:  # Rule 1: p = q + r
                    edges['red'].add((q, p))
                    edges['red'].add((r, p))
                    edge_labels.setdefault((q, p), []).append(f"{q}+{r}")
                    edge_labels.setdefault((r, p), []).append(f"{q}+{r}")
                if q < r and q + r + 1 == p:  # Rule 2: p = q + r + 1
                    edges['blue'].add((q, p))
                    edges['blue'].add((r, p))
                    edges['blue'].add((1, p))
                    edge_labels.setdefault((q, p), []).append(f"{q}+{r}+1")
                    edge_labels.setdefault((r, p), []).append(f"{q}+{r}+1")
                    edge_labels.setdefault((1, p), []).append(f"{q}+{r}+1")
    
    # Create adjacency matrix
    full_prime_list = [1] + prime_list
    index_map = {val: idx for idx, val in enumerate(full_prime_list)}
    adjacency_matrix = np.zeros((len(full_prime_list), len(full_prime_list)), dtype=int)

    for color in edges:
        for (src, dest) in edges[color]:
            adjacency_matrix[index_map[src], index_map[dest]] = 1
    
    # Compute in-degree and out-degree
    in_degrees = {p: 0 for p in primes}
    out_degrees = {p: 0 for p in primes}
    
    for _, dest in edges['red'].union(edges['blue']):
        in_degrees[dest] += 1
    for src, _ in edges['red'].union(edges['blue']):
        out_degrees[src] += 1
    
    # Convert edge labels into a formatted string
    formatted_edge_labels = {k: "\n".join(v) for k, v in edge_labels.items()}
    
    return edges, adjacency_matrix, in_degrees, out_degrees, formatted_edge_labels

def generate_dot_file(primes, edges, in_degrees, out_degrees, edge_labels):
    """
    Generate a Graphviz .dot file from the node set and edge set,
    and render it to an SVG file using the dot command.
    
    Args:
        primes (set): A set of prime numbers including 1.
        edges (dict): A dictionary with edge type keys ('red', 'blue') and sets of directed edge pairs.
        in_degrees (dict): A dictionary mapping each prime to its in-degree.
        out_degrees (dict): A dictionary mapping each prime to its out-degree.
        edge_labels (dict): A dictionary mapping each edge to a label string representing its sum.
    
    Returns:
        svg_filename (str): The name of the generated SVG file.
    """
    # Create a unique identifier for the set of primes
    primes_sorted = sorted(primes)
    primes_str = '_'.join(map(str, primes_sorted))
    hash_object = hashlib.md5(primes_str.encode())
    unique_id = hash_object.hexdigest()
    
    # Define filenames
    dot_filename = f"graph_{unique_id}.dot"
    svg_filename = f"graph_{unique_id}.svg"
    
    # Write to the .dot file
    with open(dot_filename, "w") as f:
        f.write("digraph PrimeGraph {\n")
        f.write("    rankdir=LR;\n")  # Widen the graph layout
        
        for node in primes_sorted:
            f.write(f'    {node} [shape=ellipse, label=<'
                    f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                    f'<TR><TD><FONT POINT-SIZE="16"><B>{node}</B></FONT></TD></TR>'
                    f'<TR><TD><FONT POINT-SIZE="12">In: {in_degrees[node]}</FONT></TD></TR>'
                    f'<TR><TD><FONT POINT-SIZE="12">Out: {out_degrees[node]}</FONT></TD></TR>'
                    f'</TABLE>>, fontsize=14];\n')
        
        for src, dest in sorted(edges['red']):
            label = edge_labels.get((src, dest), "")
            f.write(f'    {src} -> {dest} [color=red, penwidth=2.0, label="{label}", fontsize=10];\n')
        
        for src, dest in sorted(edges['blue']):
            label = edge_labels.get((src, dest), "")
            f.write(f'    {src} -> {dest} [color=blue, penwidth=2.0, label="{label}", fontsize=10];\n')
        
        f.write("}\n")
    
    print(f"Graphviz .dot file generated: {dot_filename}")
    
    # Render the .dot file to an SVG file using the dot command
    command = f"dot -Tsvg {dot_filename} -o {svg_filename}"
    os.system(command)
    
    print(f"SVG file generated: {svg_filename}")
    return svg_filename



if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate a prime sum graph.")
    parser.add_argument(
        '--primes',
        metavar='N',
        type=str,
        required=True,
        help='A list of prime numbers (excluding 1) or an upper bound in the form "<=N" to generate primes up to N.'
    )
    args = parser.parse_args()
    
    # Determine whether input is a direct list or a range
    prime_set = set()
    
    if args.primes.startswith("≤") or args.primes.startswith("<="):
        # Extract the upper bound from the string
        upper_bound = int(args.primes.lstrip("≤<="))
        prime_set = set(sympy.primerange(2, upper_bound + 1))  # Generate primes from 2 to upper_bound
    else:
        # Assume the input is a space-separated list of integers
        prime_set = set(map(int, args.primes.split()))
    
    # Always add 1 to the set
    prime_set.add(1)
    
    # Generate the graph
    edges, adjacency_matrix, in_degrees, out_degrees, edge_labels = generate_prime_graph(prime_set)
    print('Edges:', edges)
    print('Adjacency Matrix:\n', adjacency_matrix)
    isolated_nodes = [node for i, node in enumerate(sorted(prime_set)) 
                  if np.sum(adjacency_matrix[i, :]) == 0 and np.sum(adjacency_matrix[:, i]) == 0]

    if isolated_nodes:
        print("Isolated nodes:", isolated_nodes)
    else:
        print("No isolated nodes found.")
        
    dot_filename = generate_dot_file(prime_set, edges, in_degrees, out_degrees, edge_labels)
