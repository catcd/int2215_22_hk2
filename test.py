import networkx as nx

g = nx.DiGraph()

g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.add_edge(3, 5)
g.add_edge(4, 6)

h = list(nx.nodes(nx.dfs_tree(g, 1)))[1:]

print(1)
