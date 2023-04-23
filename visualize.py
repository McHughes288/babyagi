import argparse
from dash import Dash, html
import dash_cytoscape as cyto
from graphviz import Digraph
from utils import maybe_load_checkpoint

def visualize_tree_simple(root):
    """Use in a notebook to see how the nodes of the tree connect together"""
    dot = Digraph()

    def add_nodes(node):
        dot.node(str(node), str(node.content[0:10]))
        for child in node.children:
            dot.edge(str(node), str(child))
            add_nodes(child)

    add_nodes(root)
    return dot


def visualize_tree(root):
    app = Dash(__name__)

    def generate_elements(node, depth=0, parent_id=None):
        elements = []
        node_id = f"{depth}-{node.title}"
        elements.append({"data": {"id": node_id, "label": node.content[0:10]}})

        if parent_id is not None:
            elements.append({"data": {"source": parent_id, "target": node_id}})

        for child in node.children:
            elements.extend(generate_elements(child, depth + 1, node_id))

        return elements

    elements = generate_elements(root)

    app.layout = html.Div([
        html.P("Dash Cytoscape:"),
        cyto.Cytoscape(
            id='cytoscape',
            elements=elements,
            layout={'name': 'breadthfirst'},
            style={'width': '400px', 'height': '500px'}
        )
    ])

    app.run_server(debug=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="path to checkpoint dir")
    args = parser.parse_args()

    objective_node, _ = maybe_load_checkpoint(args.checkpoint_dir)

    visualize_tree(objective_node)


if __name__ == "__main__":
    main()