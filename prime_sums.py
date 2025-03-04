import argparse
import os
import hashlib
import networkx as nx
import numpy as np
import sympy

def generate_prime_graph(primes):
    """
    Generate the set of edge pairs and adjacency matrix based on the given prime number set.
    """
    G = nx.DiGraph()
    G.add_nodes_from(primes)
    
    edges = {'red': set(), 'blue': set()}
    edge_labels = {}
    primes_without_one = primes - {1}
    prime_list = sorted(primes_without_one)
    
    for p in prime_list:
        for q in prime_list:
            for r in prime_list:
                if q < r and q + r == p:
                    edges['red'].add((q, p))
                    edges['red'].add((r, p))
                    edge_labels.setdefault((q, p), []).append(f"{q}+{r}")
                    edge_labels.setdefault((r, p), []).append(f"{q}+{r}")
                if q < r and q + r + 1 == p:
                    edges['blue'].add((q, p))
                    edges['blue'].add((r, p))
                    edges['blue'].add((1, p))
                    edge_labels.setdefault((q, p), []).append(f"{q}+{r}+1")
                    edge_labels.setdefault((r, p), []).append(f"{q}+{r}+1")
                    edge_labels.setdefault((1, p), []).append(f"{q}+{r}+1")
    
    full_prime_list = [1] + prime_list
    index_map = {val: idx for idx, val in enumerate(full_prime_list)}
    adjacency_matrix = np.zeros((len(full_prime_list), len(full_prime_list)), dtype=int)

    for color in edges:
        for (src, dest) in edges[color]:
            adjacency_matrix[index_map[src], index_map[dest]] = 1
    
    in_degrees = {p: 0 for p in primes}
    out_degrees = {p: 0 for p in primes}
    
    for _, dest in edges['red'].union(edges['blue']):
        in_degrees[dest] += 1
    for src, _ in edges['red'].union(edges['blue']):
        out_degrees[src] += 1
    
    formatted_edge_labels = {k: "\n".join(v) for k, v in edge_labels.items()}
    
    return edges, adjacency_matrix, in_degrees, out_degrees, formatted_edge_labels

def generate_dot_file(primes, edges, in_degrees, out_degrees, edge_labels, output_format, generate_dot):
    """
    Generate a Graphviz .dot file and render it to SVG or PNG.
    """
    primes_sorted = sorted(primes)
    primes_str = '_'.join(map(str, primes_sorted))
    hash_object = hashlib.md5(primes_str.encode())
    unique_id = hash_object.hexdigest()
    
    dot_filename = f"graph_{unique_id}.dot"
    output_filename = f"graph_{unique_id}.{output_format}"
    
    if generate_dot:
        with open(dot_filename, "w") as f:
            f.write("digraph PrimeGraph {\n")
            f.write("    rankdir=LR;\n")
            
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
    
        os.system(f"dot -T{output_format} {dot_filename} -o {output_filename}")
        print(f"Graph output file generated: {output_filename}")
    return output_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a prime sum graph.")
    parser.add_argument('--primes', metavar='N', type=str, required=True, 
                        help='A list of prime numbers or an upper bound in the form "<=N".')
    parser.add_argument('--output-format', choices=['svg', 'png'], default='svg',
                        help='Specify output format (svg/png). Default is svg.')
    parser.add_argument('--no-dot', action='store_true',
                        help='If set, does not generate the .dot file.')
    args = parser.parse_args()
    
    prime_set = set()
    if args.primes.startswith("≤") or args.primes.startswith("<="):
        upper_bound = int(args.primes.lstrip("≤<="))
        prime_set = set(sympy.primerange(2, upper_bound + 1))
    else:
        prime_set = set(map(int, args.primes.split()))
    
    prime_set.add(1)
    
    edges, adjacency_matrix, in_degrees, out_degrees, edge_labels = generate_prime_graph(prime_set)
    # print('Edges:', edges)
    print('Adjacency Matrix:\n', adjacency_matrix)
    
    sink_nodes = [node for node in sorted(prime_set) if out_degrees[node] == 0]
    print("Sink nodes (leaf nodes):", sink_nodes)
    
    output_filename = generate_dot_file(prime_set, edges, in_degrees, out_degrees, edge_labels, 
                                        args.output_format, not args.no_dot)
