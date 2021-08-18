#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
from operator import itemgetter

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
import pylab


# In[3]:


got_graph=nx.Graph()


# In[4]:


with open('got_nodes.txt', 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprehension and list slicing to remove the header row)
    node = [n for n in nodereader][1:]
    
nodes = [n[0] for n in node] # Get a list of only the node names

with open('got_edges.txt', 'r') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e[0:3]) for e in edgereader][1:] # Retrieve the edges along with the weights


# In[5]:


print(len(nodes))
print(len(edges))


# In[6]:


got_graph.add_nodes_from(nodes) # Add nodes to the Graph
got_graph.add_weighted_edges_from(edges,weight='Weight') # Add edges and edge weights to the Graph 
print(nx.info(got_graph)) # Print information about the Graph


# In[7]:


label_dict={}

for n in node: # Loop through the list, one row at a time
    label_dict[n[0]] = n[1]
 
  


# In[8]:


nx.set_node_attributes(got_graph, label_dict, 'label')


# In[9]:


for n in got_graph.nodes:
    print(n, got_graph.nodes[n]['label'])


# In[10]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")


# ### DEGREE ANALYSIS

# In[11]:


#  degree method 
got_graph_degree = list(dict(got_graph.degree()).values())

print(got_graph_degree)


# In[12]:


# basic statistics
print(np.mean(got_graph_degree))
print(np.median(got_graph_degree))
print(np.std(got_graph_degree))
print(np.max(got_graph_degree))
print(np.min(got_graph_degree))


# In[13]:


def make_histogram(aGraph):     
    fig = pylab.figure()
    hist = nx.degree_histogram(aGraph)
    pylab.bar(range(len(hist)), hist, align = 'center')
    pylab.xlim((0, len(hist)))
    pylab.xlabel("Degree of node")
    pylab.ylabel("Number of nodes")
    return fig
make_histogram(got_graph)


# The degree centrality values are normalized by dividing by the maximum possible degree in a simple graph n-1 where n is the number of nodes in G.

# In[14]:


degree_centrality = nx.degree_centrality(got_graph)
print(degree_centrality)


# In[15]:


#density (p=density)
density=nx.density(got_graph)
print('Density: {}'.format(density))


# In[16]:


#betweenness centrality
betweenness_centrality=nx.betweenness_centrality(got_graph)
print(betweenness_centrality)


# In[17]:


# closeness centrality
closeness_centrality = nx.closeness_centrality(got_graph)
print(closeness_centrality)


# In[18]:


#Eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(got_graph)
print(eigenvector_centrality)


# In[19]:


hub_degree = sorted(degree_centrality.items(),key = lambda x:x[1], reverse=True)[0]
hub_betweenness = sorted(betweenness_centrality.items(),key = lambda x:x[1], reverse=True)[0]
hub_closeness = sorted(closeness_centrality.items(),key = lambda x:x[1], reverse=True)[0]
hub_eigenvector = sorted(eigenvector_centrality.items(),key = lambda x:x[1], reverse=True)[0]
print('degree:      ',hub_degree)
print('betweenness: ',hub_betweenness)
print('closeness:   ',hub_closeness)
print('eigenvector: ',hub_eigenvector)

#thats enough, dont have to calculate katz and pagerank for undirected graph.


# ### ECDF and ECCDF

# In[20]:


# ECDF in linear scale
cdf_function = ECDF(got_graph_degree)
x = np.unique(got_graph_degree)
y = cdf_function(x)
fig_cdf_function = plt.figure(figsize=(8,5)) 
axes = fig_cdf_function.gca()
axes.plot(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECDF in loglog scale
fig_cdf_function = plt.figure(figsize=(8,5))
axes = fig_cdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECCDF in loglog scale
y = 1-cdf_function(x)
fig_ccdf_function = plt.figure(figsize=(8,5))
axes = fig_ccdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECCDF',size = 30)


# ### HUBS

# Nodes with high degree. Fix the quantile in the CDF. given  𝑞∈[0,1]  find the degree  𝑘  such that  𝐹𝑋(𝑘)=𝑞 . We use the Numpy function percentile.  𝑞=0.95

# In[21]:


percentile_98 = np.percentile(got_graph_degree,98)
print(percentile_98)


# Now we can identify the hubs by using the list comprehension

# In[22]:


hub_nodi = [k for k,v in dict(got_graph.degree()).items() if v>= percentile_98]
print(hub_nodi)


# In[23]:


print(len(hub_nodi))
print(list(hub_nodi))


# In[24]:


#### Isolates
print(list(nx.isolates(got_graph)))


# ### Connectivity

# In[25]:


print(nx.is_connected(got_graph))
print(nx.number_connected_components(got_graph))


# In[26]:


nx.diameter(got_graph)
#The maximum shortest distance between a pair of nodes in a graph 


# In[27]:


(nx.average_shortest_path_length(got_graph)) 
#average of shortest paths between all possible pairs of nodes 


# ### Random networks: the Erdos-Renyi model

# From
# $$ <k> = p (N-1)$$
# we obtain $p = \frac{<k>}{N-1}$

# In[28]:


mean_degree_got=np.mean(got_graph_degree)
p= mean_degree_got/(got_graph.order()-1)
p #same as the density of the network


# In[29]:


random_graph = nx.fast_gnp_random_graph(got_graph.order(),p)


# In[30]:


print('Number of nodes: {}'.format(random_graph.order()))
print('Number of links: {}'.format(random_graph.size()))


# In[31]:


random_degree = list(dict(random_graph.degree()).values())
np.mean(random_degree) 


# In[87]:


cdf_got_graph = ECDF(got_graph_degree)
x_sw = np.unique(got_graph_degree)
y_sw = cdf_got_graph(x_sw)
cdf_random = ECDF(random_degree)
x_random = np.unique(random_degree)
y_random = cdf_random(x_random)
fig_cdf_sw = plt.figure(figsize=(16,9))
assi = fig_cdf_sw.gca()
assi.set_xscale('log')
assi.set_yscale('log')
assi.loglog(x_sw,1-y_sw,marker='o',ms=8, linestyle='--', label='Real Network')
assi.plot(x_random,1-y_random,marker='+',ms=10, linestyle='--',label='Random Network')
assi.set_xlabel('Degree',size=30)
assi.set_ylabel('ECCDF', size = 30)
assi.legend(loc="upper right")


# ## Triangles

# Note: When computing triangles for the entire graph each triangle is counted three times, once at each node. 
# Note: Self loops are ignored.

# In[33]:


print('game of thrones graph - dictionary keyed by nodes: number of triangles {}'.format(nx.triangles(got_graph)))
print('game of thrones graph - number of triangles of node labelled Jon: {}'.format(nx.triangles(got_graph,'Jon')))
print('game of thrones graph - list of the number of triangles of all nodes: {}'.format(list(nx.triangles(got_graph).values())))


# ### Transitivity - Global Clustering Coefficient
# 
# a measure of the degree to which nodes in a graph tend to cluster together.

# In[34]:


transitivity=nx.transitivity(got_graph)
print(transitivity)
#transitivty gives more weights to high degree nodes


# ### Local clustering coefficient

# In[35]:


got_local_clustering= nx.clustering(got_graph)
got_local_clustering['Jon']


# In[36]:


greatestlocalclustering = sorted(got_local_clustering.items(),key = lambda x:x[1], reverse=True)
greatestlocalclustering


# Local clustering is interesting, rough dependence on degree in real network, 
# vertices with higher degree having lower local clustering coef. on average
# Local ccmeasures influence.
# Betweennes and local CC are STRONGLY CORRELATED
# 
# when the neighbors of a node are not connected to one another we say the network structure contains STRUCTURAL HOLES

# ### Average Clustering Coefficient

# In[37]:


print('average local clustering: {}'.format(nx.average_clustering(got_graph)))


# In[38]:


print('average local clustering: {}'.format(nx.average_clustering(random_graph)))


# ### Correlation

# Pearson correlation coefficient and p-value for testing non-correlation. 
# The Pearson correlation coefficient measures the linear relationship between two datasets 
# It varies between -1 and +1 with 0 implying no correlation

# In[39]:


print(scipy.stats.pearsonr(list(degree_centrality.values()),list(eigenvector_centrality.values())))
print(scipy.stats.pearsonr(list(betweenness_centrality.values()),list(got_local_clustering.values())))

#buranın interpretation ını öğren


# #### Degree Assortativity

# In[40]:


print(nx.degree_assortativity_coefficient(got_graph)) 
#buraya weighti ekle


# In[41]:


nx.write_gexf(G=got_graph,path='got_graph.gexf')


# ## COMMUNITY

# ### Communitiy detection with Greedy Algorithm

# In[42]:


import networkx.algorithms.community as nx_comm


# In[43]:


list_com_sets_greedy = list(nx_comm.greedy_modularity_communities(got_graph))
print(list_com_sets_greedy)


# In[44]:


partition_greedy = {}
for i, comm in enumerate(list_com_sets_greedy):
    print("Community:", i)
    print(i,comm)
    for n in comm:
        partition_greedy[n]=i


# In[45]:


print(partition_greedy)


# In[46]:


nx.set_node_attributes(got_graph, partition_greedy, "community_greedy")


# In[47]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")


# ### Communitiy detection with Louvain Algorithm

# In[48]:


import community as community_louvain
import matplotlib.cm as cm


# In[49]:


partition_library = community_louvain.best_partition(got_graph)


# In[50]:


print(partition_library)


# In[51]:


nx.set_node_attributes(got_graph, partition_library, "community_library")


# In[52]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")


# In[53]:


# draw the graph with partition_greedy
pos = nx.spring_layout(got_graph)

cmap = cm.get_cmap('viridis', max(partition_greedy.values()) + 1)
nx.draw_networkx_nodes(got_graph, pos, partition_greedy.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition_greedy.values()))
nx.draw_networkx_edges(got_graph, pos, alpha=0.5)
plt.show()


# In[54]:


# draw the graph with partition_library
pos = nx.spring_layout(got_graph)

cmap = cm.get_cmap('viridis', max(partition_library.values()) + 1)
nx.draw_networkx_nodes(got_graph, pos, partition_library.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition_library.values()))
nx.draw_networkx_edges(got_graph, pos, alpha=0.5)
plt.show()


# ### Communities ---EVALUATION

# In[55]:


comms = set(partition_library.values())
comms


# In[56]:


list_community_sets_library = [ set() for i in range(len(comms)) ]


# In[57]:


for n, comm in partition_library.items():
    list_community_sets_library[comm].add(n)

list_community_sets_library


# In[58]:


for my_list in [list_com_sets_greedy,  list_community_sets_library]:
    
    #print("Coverage")
    print("Coverage", nx_comm.coverage(got_graph, my_list))
    #print("Modularity")
    print("Modularity", nx_comm.modularity(got_graph, my_list, weight='weight'))
    #print("Performance")
    print("Performance", nx_comm.performance(got_graph, my_list))
    
    print("---")


# ### Communitiy detection with k_clique_communities 
# for overlapping communities

# In[59]:


from networkx.algorithms.community import k_clique_communities


# In[60]:


print("min size: 4", list(nx_comm.k_clique_communities(got_graph, 4)) ) # k (int) – Size of smallest clique)
print('---------------')
print("min size: 6", list(nx_comm.k_clique_communities(got_graph, 6)) ) # k (int) – Size of smallest clique)


# In[61]:


list_community_sets_kclique = nx_comm.k_clique_communities(got_graph, 4)


# In[62]:


map_4clique = {}

for i, kclique in enumerate(list_community_sets_kclique):
    print("Community:", i)
    print(i,kclique)
    for n in kclique:
        map_4clique[n]=i


# In[63]:


map_4clique


# In[64]:


nx.set_node_attributes(got_graph, map_4clique, "k4_clique_communities")


# In[65]:


list_community_sets_kclique = nx_comm.k_clique_communities(got_graph, 6)

map_6clique = {}

for i, kclique in enumerate(list_community_sets_kclique):
    print("Community:", i)
    print(i,kclique)
    for n in kclique:
        map_6clique[n]=i
nx.set_node_attributes(got_graph, map_6clique, "k6_clique_communities")


# In[66]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")


# ### Size distribution of communities

# In[67]:


list_community_sets_library


# In[68]:


pairs = []
for index, nodes in enumerate(list_community_sets_library):
    print(index,len(nodes))
    comm_size = (index,len(nodes))
    pairs.append(comm_size)


# In[69]:


pairs


# In[70]:


community_index = []
number_of_nodes = []

for index, n in pairs:
    community_index.append(str(index))
    number_of_nodes.append(n)   
    
    
plt.bar(community_index,number_of_nodes)
plt.xlabel("Community")
plt.ylabel("Number of nodes")


# ### Centrality in communities

# In[71]:


list_community_sets_library


# In[72]:


for comm in list_community_sets_library:
    subgraph = got_graph.subgraph(comm)
    print(subgraph.order())


# In[73]:


centr_comm = {}
# node -> centrality in the community subgraph


# In[74]:


for comm in list_community_sets_library:
    subgraph = got_graph.subgraph(comm)
    print(subgraph.order())
    print(nx.degree_centrality(subgraph))
    print("---")
    
    node_degrees  = nx.degree_centrality(subgraph)
    for n,d in node_degrees.items():
        centr_comm[n] = d


# In[75]:


centr_comm


# In[76]:


nx.set_node_attributes(got_graph, centr_comm, "centr_comm")
#important nodes in communites, centralities inside the community


# In[77]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")


# ### Bridges

# Yields e (edge) – An edge in the graph whose removal disconnects the graph 
# (or causes the number of connected components to increase)

# In[78]:


nx.has_bridges(got_graph.to_undirected())


# In[79]:


nx.set_edge_attributes(got_graph, 0, name="is_bridge")


# In[80]:


for br in nx.bridges(got_graph.to_undirected(), root=None):
    #print("edge (src,target):", br)
    src,target = br
    got_graph[src][target]['is_bridge'] = 1 


# ### Local Bridges
# 
# A local bridge is an edge whose endpoints have no common neighbors. That is, the edge is not part of a triangle in the graph
# 
# The span of a local bridge is the shortest path length between the endpoints if the local bridge is removed.

# In[84]:


nx.set_edge_attributes(got_graph, 0, name="is_local_bridge")

for br in nx.local_bridges(got_graph, with_span=True, weight='None'):
    #print("edge (src,target, span):", br)
    src, target, span = br
    got_graph[src][target]['is_local_bridge'] = 1


# In[82]:


nx.write_gexf(G=got_graph,path="got_graph.gexf")

