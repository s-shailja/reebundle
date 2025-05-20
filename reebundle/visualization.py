import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import plotly.graph_objects as go


def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

def graph_vis(G, node_loc, streamlines):
    """Function to visualize the graph with node locations. 
    Input: Graph G and Node location hash map
    """
    # 3d spring layout
    pos = node_loc
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w",label = None)
    for i in range(len(streamlines)):
        xdata = []
        ydata = []
        zdata = []
        for j in streamlines[i]:
            xdata.append(j[0])
            ydata.append(j[1])
            zdata.append(j[2])
        ax.plot3D(xdata,ydata,zdata,color= '#eb7a30', lw = 2, alpha = 0.05);
    # Plot the nodes
    ax.scatter(*node_xyz.T, s=400, ec="w", color = 'r', zorder=100)
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Plot the edges
    weight_labels = list(edge_labels.values())
    count = 0
    for vizedge in edge_xyz:
        wt = weight_labels[count]/max(weight_labels)*5
        ax.plot(*vizedge.T, color='#000000',
                lw = wt,
                zorder = 50,
               label = str(weight_labels[count]))
        count+=1
    _format_axes(ax)
    fig.tight_layout()
    plt.axis("off")   
    plt.show()

def plot_reeb_graph_3d(G, node_loc, streamlines):
    """
    Plot the Reeb graph in 3D using Plotly.
    Parameters
    ----------
    G : networkx.Graph
        The Reeb graph to plot.
    """
    # Extract node and edge positions from the layout
    node_xyz = np.array([node_loc[v] for v in sorted(G)])
    edge_xyz = np.array([(node_loc[u], node_loc[v]) for u, v in G.edges()])
    edge_labels = nx.get_edge_attributes(G, "weight")
    weight_labels = list(edge_labels.values())
    
    # 3d spring layout
    pos = node_loc
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Separate X,Y,Z coordinates for Plotly
    x_nodes = [node_loc[i][0] for i in G.nodes()]
    y_nodes = [node_loc[i][1] for i in G.nodes()]
    z_nodes = [node_loc[i][2] for i in G.nodes()]
    edge_list = G.edges()
    x_edges, y_edges, z_edges = [], [], []
    for edge in edge_list:
        x_edges.append([node_loc[edge[0]][0], node_loc[edge[1]][0], None])
        y_edges.append([node_loc[edge[0]][1], node_loc[edge[1]][1], None])
        z_edges.append([node_loc[edge[0]][2], node_loc[edge[1]][2], None])
    traces = []
    for i in range(len(x_edges)):
        trace_edges = go.Scatter3d(
            x=x_edges[i],
            y=y_edges[i],
            z=z_edges[i],
            mode='lines',
            line=dict(color='black', width=weight_labels[i]*10),
        )
        traces.append(trace_edges)
    xdata, ydata, zdata = [], [], []
    for streamline in streamlines:
        for point in streamline:
            xdata.append(point[0])
            ydata.append(point[1])
            zdata.append(point[2])
    trace_data = go.Scatter3d(
        x=xdata,
        y=ydata,
        z=zdata,
        mode='markers',
        marker=dict(symbol='circle', color='#eb7a30', size=1),
        hoverinfo=None
    )
    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(symbol='circle', color='red', size=10, line=dict(color='red', width=0.5)),
        hoverinfo='text'
    )
    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')
    layout = go.Layout(
        title='Interactive Visualization for Reeb Graph',
        width=650,
        height=625,
        showlegend=False,
        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
        margin=dict(t=100),
        hovermode='closest'
    )
    data = traces + [trace_nodes, trace_data]
    fig = go.Figure(data=data, layout=layout)
    fig.show()
