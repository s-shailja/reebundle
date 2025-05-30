import nibabel as nib
import networkx as nx
import reebundle.construct as rc


def test_rg_nodes_edges():
    eps = 2.5
    alpha = 3
    delta = 5
    trkpathI = "tests/data/"
    file = "CA.trk"
    streamlines = nib.streamlines.load(trkpathI + file).streamlines
    h, node_loc = rc.constructRobustReeb(streamlines, eps, alpha, delta)
    assert isinstance(h, nx.Graph)
    assert isinstance(node_loc, dict)
    assert len(h.nodes) == 9
    assert len(h.edges) == 8
