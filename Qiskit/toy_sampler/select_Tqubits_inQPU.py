#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit_ibm_runtime import QiskitRuntimeService
import networkx as nx
from pprint import pprint
import pickle,os
import numpy as np
import copy

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-nt','--nodeThresh',type=float, nargs=2, default=[30., 0.90], help="minimal node [T/ns, readoutFid]  threshold")
    parser.add_argument('-et','--edgeThresh',type=float, default=0.98, help="minimal edge fidelity threshold")
    parser.add_argument('-gt','--graphThresh',type=float, default=0.96, help="minimal graph fidelity threshold")
    parser.add_argument('-b','--backend',default="ibm_hanoi", help="tasks")    
    parser.add_argument( "-P","--pickleCalib", action='store_true', default=False, help="will read backend calibration from pickle")
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")

    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if 1:
        assert args.backend in ["ibmq_qasm_simulator",
                        "ibmq_kolkata","ibmq_mumbai","ibm_algiers","ibm_hanoi", "ibm_cairo",# 27 qubits
                                "ibm_brisbane", 'ibm_nazca','ibm_sherbrooke' ,'ibm_cusco' # 127 qubits
                                 ]
    return args


#...!...!....................
# Function to acquire credentials and get the specified backend
def get_backend(backend_name):
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print('got backend:',backend)
    return backend

#...!...!....................
# Function to download calibration data for the specified backend
def get_calibration(backend):
    properties = backend.properties()
    print('got properties for ',backend)
    return properties

def save_graph_to_file(G, filename):
    with open(filename, 'wb') as file:
        pickle.dump(G, file)
        
def read_graph_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    

#...!...!....................
# Function to extract connectivity with fidelity from calibration data
def extract_connectivity_with_fidelity(properties,gateN):
    G = nx.Graph()
    #  {'qubits': [19, 16], 'gate': 'cx', 'parameters': [{'date': datetime.datetime(2023, 11, 27, 23, 4, 3, tzinfo=tzlocal()), 'name': 'gate_error', 'unit': '', 'value': 0.006633318897474982}, {'date': datetime.datetime(2023, 11, 25, 11, 33, 6, tzinfo=tzlocal()), 'name': 'gate_length', 'unit': 'ns', 'value': 327.1111111111111}], 'name': 'cx19_16'}
    for gate in properties.gates:
        if gate.gate == gateN:
            #print('gg',gate.to_dict()); aa
            #pprint(gate.to_dict());bb
            q1, q2 = gate.qubits
            fidelity = 1-properties.gate_error(gateN, [q1, q2])
            G.add_edge(q1, q2, weight=1, cxfidel=fidelity)
            # weights=1 are better for displaying the graph
    print('full graph has %d %s-edges '%(G.number_of_edges(), gateN))

    #.... for all nodes add readout fidelity
    #{'T1': (0.00012812375155067278, datetime.datetime(2023, 11, 27, 21, 39, 18, tzinfo=tzlocal())), 'T2': (0.00012847631635998057, datetime.datetime(2023, 11, 27, 21, 41, 9, tzinfo=tzlocal())), 'frequency': (5095809071.851239, datetime.datetime(2023, 11, 28, 9, 18, 45, tzinfo=tzlocal())), 'anharmonicity': (-341017528.47322655, datetime.datetime(2023, 11, 28, 9, 18, 45, tzinfo=tzlocal())), 'readout_error': (0.009000000000000008, datetime.datetime(2023, 11, 27, 21, 35, 32, tzinfo=tzlocal())), 'prob_meas0_prep1': (0.0134, datetime.datetime(2023, 11, 27, 21, 35, 32, tzinfo=tzlocal())), 'prob_meas1_prep0': (0.0046000000000000485, datetime.datetime(2023, 11, 27, 21, 35, 32, tzinfo=tzlocal())), 'readout_length': (8.177777777777777e-07, datetime.datetime(2023, 11, 27, 21, 35, 32, tzinfo=tzlocal()))}
    for N in G.nodes():
        #print('N=',N)
        qProp=properties.qubit_property(N)
        '''
        print(type(qProp),list(qProp));
        for key in qProp:
            print(key,qProp[key])
        '''        
        m0p1=qProp['prob_meas0_prep1'][0]        
        Tm_us= min(qProp['T1'][0], qProp['T2'][0] ) *1e6

        #print('q:%2d Tm/ns=%8.1f  m0p1=%8.3f'%(N,Tm_us,m0p1))
        G.nodes[N]['Tmin']=Tm_us
        G.nodes[N]['Rfid']=1-m0p1
    return G


#...!...!....................
def filter_graph_by_node_attribute(G, thrD):
    print('FGNA: thr:',thrD)
    badNL=[]
    n0N=G.number_of_nodes()
    for N, attr in G.nodes(data=True):
        isGood=True
        for key in thrD:
            #print('kk',N,key,attr.get(key, 0),thrD[key])
            if attr.get(key, 0) < thrD[key]: isGood=False
        if isGood: continue
        badNL.append(N)
        #print('drop node:',N,attr)
    print('removing %d nodes:%s'%(len(badNL),badNL))
    G.remove_nodes_from(badNL)
    print('filtered graph with node thres:%s  has %d nodes of %d'%(thrD,G.number_of_nodes(),n0N))
    

#...!...!....................
def filter_graph_by_edge_attribute(G, attributeN, threshold):
    # Create a new graph of the same type as the input graph
    new_graph = type(G)()

    # Add all nodes to the new graph (assumes nodes don't have the attribute we're filtering by)
    new_graph.add_nodes_from(G.nodes(data=True))

    # Iterate over all edges and their attributes in the original graph
    for u, v, attr in G.edges(data=True):
        # Check if the edge attribute meets the threshold
        if attr.get(attributeN, 0) >= threshold:
            # Add the edge to the new graph
            new_graph.add_edge(u, v, **attr)
            
    print('filtered graph with edge fidelity>%.3f  has %d edges of %d'%(threshold,new_graph.number_of_edges(),G.number_of_edges()))
    return new_graph


#...!...!....................
# Function to find nodes with a degree of 3
def find_nodes_with_degree_three(G,thrs=0.9):
    print('\nsearch for Ts in G, cxfidel_prod thrs=%.3f'%thrs)
    NL = []

    for N, degree in dict(G.degree()).items():
        if degree != 3: continue
        #print('Tnode',N)
        fidProd=1.
        for neighbor in G.neighbors(N):
            edge_data = G.get_edge_data(N, neighbor)
            cxfidel = edge_data.get('cxfidel', None)
            #print('cxfidel=',cxfidel)
            fidProd*=cxfidel
        #print('see fidPros=',fidProd)
        if fidProd<thrs: continue
        G.nodes[N]['Tfidel']=fidProd
        NL.append(N)
    print('found %d T-nodes'%len(NL))
    if args.verb>1:          
        for N in NL: print_node_attributes(G, N)
    return NL


#...!...!....................
def cut_good_subgraph(G , Tthres):
    '''
    find  a node with degree 3  with enough fidelity
    Graph (a): Find a node with degree 3, create a subgraph containing this node and its connected neighbors.
    Graph (b): Create a new graph from G and then remove the nodes and edges that are in graph (a).
    '''
    
    N=None
    for node, degree in G.degree():
        if degree != 3: continue
        val = G.nodes[node].get('Tfidel', None)
        if val==None or val< Tthres: continue
        N = node
        break

    if N is None:
        print("No node with exactly 3 edges found.")
        return None, G
    # raise ValueError("aaa bbb")

    # Create graph (a)
    neighbors = list(G.neighbors(N))
    subNL = neighbors + [N]
    graph_a = G.subgraph(subNL).copy()

    # Create graph (b)
    #graph_b = nx.Graph(G)  # Assuming G is a Graph, use DiGraph if G is directed
    graph_b = G.copy()
    graph_b.remove_nodes_from(subNL)

    print('new isolate T-graph nodes: %s, left %d edges'%( list(graph_a.nodes),graph_b.number_of_edges()))
    return graph_a, graph_b


def print_graph_edges_with_weights(G):
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', None)  # Get the weight of the edge, default to None if not present
        print(f"Edge ({u}, {v}): Weight = {weight}")

def print_all_edge_attributes(G):
    print('all_edge_attributes')
    for u, v, attributes in G.edges(data=True):
        print(f"Edge ({u}, {v}): {attributes}")
        
def print_node_attributes(G, N):
    if N in G:
        attributes = G.nodes[N]
        print(f"Attributes of node {N}: {attributes}")
    else:
        print(f"Node {N} is not in the graph.")


#...!...!....................
def print_graph_nodes_with_edges(G,txt=''):
    # print the adjacency list
    #for line in nx.generate_adjlist(G):   print(line)
    nN=G.number_of_nodes()
    nE=G.number_of_edges()
    print(txt+' %d nodes, %d edges'%(nN,nE))
    for N in G.nodes():
        # Build a dictionary for the current node with neighbors and weights
        attr = G.nodes[N]
        #print('nn', N,attributes)
        edges = {}
        for neighbor in G.neighbors(N):
            # Check if the edge has a weight attribute
            weight = G[N][neighbor].get('weight', None)
            edges[neighbor] = weight
   
        print("[%d] %s edges:%s"%(N,attr,edges))


#...!...!....................
def print_final_qubits_map(G):
    outLL=[]
    for N in G.nodes():
        nEdge=G.degree(N)
        if nEdge!=3: continue
        qL = [N]
        for neighbor in G.neighbors(N): qL.append(neighbor)
        #print('qubits:',qL)
        outLL.append(qL)
    print('final %s %d locations, qubitIDs, multiQMap='%(args.backend ,len(outLL)),outLL)

    
#...!...!....................
# Function to plot the connectivity graph with specific nodes colored
def plot_connectivity_graph(G, pos, Tthres=0.98, tit='aa', outF='aa.png'):
    # Create a plot with one subplot
    # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    
    fig, ax = plt.subplots()
    #ax.set_xlim(-1.1,1.1);     ax.set_ylim(-1.1,1.1)
  
    fig.set_size_inches(18,8)
    # Define node colors
    node_colors = []
    for N in G.nodes():
        val = G.nodes[N].get('Tfidel', None)
        nEdge=G.degree(N)
        #print('vv',val,N,Tthres)
        if val==None or nEdge!=3:
            ncol='white'
        elif val< Tthres:
            ncol='yellow'
        else:
            ncol='lime'
        node_colors.append(ncol)
    
    #print('nncc',node_colors)
    # Draw the graph on the Axes
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, edgecolors='black')

    # Annotate edges with CX fidelities
    edge_cxfidel = nx.get_edge_attributes(G, 'cxfidel')
    # Format the fidelities to 2 decimal places for display
    formatted_edge_labels = {(e[0], e[1]): f"{edge_cxfidel[e]:.3f}" for e in edge_cxfidel}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=formatted_edge_labels)
  
    # Annotate with the 'tfidel' attribute
    #    for node, (x, y) in pos.items():
    for node in G:
        (x, y)=pos[node]
        #print('nnn',node,x,y)
        val = G.nodes[node].get('Tfidel', None)
        if val==None: continue
        ax.text(x+0.03, y + 0.10, s='%.3f'%val, horizontalalignment='center', fontsize=10,color='m')

    tit1='%s , %s'%(args.backend,tit)
    ax.set(title=tit1)
    # Save the plot to the specified file
    plt.savefig(outF, format='png', bbox_inches='tight')
    plt.close()


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    np.set_printoptions(precision=3)

    gateN='cx'
    if args.backend in [ 'ibm_brisbane', 'ibm_sherbrooke','ibm_nazca','ibm_cusco' ]: gateN='ecr'
    
    qpuP='qpu_%s.pkl'%args.backend
    qpuF=os.path.join(args.outPath,qpuP)
    if args.pickleCalib:
        qpu_graph0=read_graph_from_file(qpuF)
    else:  # talk to IBMQ        
        backend = get_backend(args.backend)
        properties = get_calibration(backend)
        qpu_graph0 = extract_connectivity_with_fidelity(properties, gateN)
        save_graph_to_file(qpu_graph0,qpuF)

    print_graph_nodes_with_edges(qpu_graph0,txt='\nM:input full graph')
    
    # Use spectral layout for the positions
    positions = nx.spectral_layout(qpu_graph0)
    #print('pos range',positions) ; aa
    # >>> pos range {19: array([-0.55079778,  0.69133138]), 16: array([-0.29589719,  0.63172437]),

    outF=os.path.join(args.outPath,'inp0all.png')
    plot_connectivity_graph(qpu_graph0, positions, Tthres=0,tit='all nodes', outF=outF)
    qpu_graph = copy.deepcopy(qpu_graph0)

    if 1:
        #... remove nodes with bad nodes
        nodeThrD={'Tmin':args.nodeThresh[0],'Rfid':args.nodeThresh[1]}
        filter_graph_by_node_attribute(qpu_graph,thrD=nodeThrD)
        print_graph_nodes_with_edges(qpu_graph,txt='\nM:good nodes')
        outF=os.path.join(args.outPath,'inp1nodes.png')
        plot_connectivity_graph(qpu_graph, positions, Tthres=0,tit='good nodes', outF=outF)

    #... remove bad CX gates    
    qpu_graph=filter_graph_by_edge_attribute(qpu_graph, 'cxfidel', threshold=args.edgeThresh)
    print_graph_nodes_with_edges(qpu_graph,txt='\nM:good edges graph')
    outF=os.path.join(args.outPath,'inp2edges.png')
    plot_connectivity_graph(qpu_graph, positions, Tthres=0,tit='good edges', outF=outF)
     
    #print_graph_nodes_with_edges(qpu_graph)
    #print_graph_edges_with_weights(gpu_graph)
    if args.verb>1: print_all_edge_attributes(qpu_graph)
    nodes3E = find_nodes_with_degree_three(qpu_graph)
    
    print('M: itarating and plotting')
    Tthres=args.graphThresh
    res_graph=qpu_graph
    mapL=[]
    tit=args.backend+'    good candidates'
    nIter=0
    while res_graph:
        if args.verb>1: print_graph_nodes_with_edges(res_graph,txt='\nM:res-g')
        outF=os.path.join(args.outPath,'ires%d.png'%len(mapL))
        plot_connectivity_graph(res_graph, positions, Tthres=Tthres,tit=tit, outF=outF)
        
        tit='iter=%d'%nIter
        t_graph, res_graph=cut_good_subgraph(res_graph, Tthres=Tthres)
        nIter+=1
        if t_graph==None: break
        
        if args.verb>1:  print_all_edge_attributes(t_graph)
        mapL.append(t_graph)
        #exit(0)

    nMap=len(mapL)
    print('found %d locations for T'%nMap)
    if nMap<=0 : exit(0)
    u_graph=mapL[0]
    for i in range(1,nMap):
        u_graph=nx.union(u_graph,mapL[i])
    outF=os.path.join(args.outPath,'optMap_%d_%s.png'%(nMap,args.backend))
    tit='final choice of %d locations'%nMap
    plot_connectivity_graph(u_graph,  positions, tit=tit, Tthres=Tthres,outF=outF)   
    
    
    print_final_qubits_map(u_graph)
