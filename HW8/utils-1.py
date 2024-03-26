filename = 'web-Google_10k.txt'
with open(filename,'r') as input_file: 
    # The first 4 lines are metadata about the graph that you do not need 
    # After the metadata, the next lines are edges given in the format: `node1\tnode2\n` where node1 points to node2
    lines = [item.replace('\n','').split('\t') for item in input_file] 
    edges = [[int(item[0]),int(item[1])] for item in lines[4:]]

    nodes_with_duplicates = [node for pair in edges for node in pair]
    nodes = sorted(set(nodes_with_duplicates))

    # There are 10K unique nodes, but the nodes are not numbered from 0 to 10K!!! 
    # E.g. there is a node with the ID 916155 
    # So you might find these dictionaries useful in the rest of the assignment
    node_index = {node: index for index, node in enumerate(nodes)}
    index_node = {index: node for node, index in node_index.items()}