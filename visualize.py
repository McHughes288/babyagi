import argparse
from dash import Dash, html, dcc
import dash_cytoscape as cyto
from graphviz import Digraph
from utils import maybe_load_checkpoint

import json
from dash.dependencies import Input, Output

def NamedDropdown(name, **kwargs):
    return html.Div(
        style={'margin': '10px 0px'},
        children=[
            html.P(
                children=f'{name}:',
                style={'margin-left': '3px'}
            ),

            dcc.Dropdown(**kwargs)
        ]
    )
def DropdownOptionsList(*args):
    return [{'label': val.capitalize(), 'value': val} for val in args]

def visualize_tree_simple(root):
    """Use in a notebook to see how the nodes of the tree connect together"""
    dot = Digraph()

    def add_nodes(node):
        dot.node(str(node), str(node.task_id))
        for child in node.children:
            dot.edge(str(node), str(child))
            add_nodes(child)

    add_nodes(root)
    return dot


def visualize_tree(root):
    app = Dash(__name__)

    def generate_elements(node, depth=0, parent_id=None):
        elements = []
        node_id = f"{depth}-{node.task_id}"
        elements.append({
            "data": {
                "id": node_id,
                "label": node.task_id,
                "task_name": node.task_name,
                "result": node.result,
                "state": node.state
            }
        })
        if parent_id is not None:
            elements.append({"data": {"source": parent_id, "target": node_id}})

        for child in node.children:
            elements.extend(generate_elements(child, depth + 1, node_id))

        return elements

    elements = generate_elements(root)

    styles = {
        'json-output': {
            'overflow-y': 'scroll',
            'height': 'calc(50% - 25px)',
            'border': 'thin lightgrey solid'
        },
        'tab': {'height': 'calc(98vh - 80px)'}
    }

    app.layout = html.Div([
        html.Div(className='eight columns', children=[
            cyto.Cytoscape(
                id='cytoscape',
                elements=elements,
                # stylesheet=default_stylesheet,
                layout={'name': 'breadthfirst'},
                style={
                    'height': '95vh',
                    'width': 'calc(100% - 500px)',
                    'float': 'left'
                },
                responsive=True
            )
        ]),
        html.Div(className='four columns', children=[
            dcc.Tabs(id='tabs', children=[
                dcc.Tab(label='Control Panel', children=[
                    NamedDropdown(
                        name='Layout',
                        id='dropdown-layout',
                        options=DropdownOptionsList(
                            'breadthfirst',
                            'grid',
                            'circle',
                            'concentric',
                            'cose',
                        ),
                        value='breadthfirst',
                        clearable=False
                    ),
                ]),
                dcc.Tab(label='JSON', children=[
                    html.Div(style=styles['tab'], children=[
                        html.P('Node Object JSON:'),
                        html.Pre(
                            id='tap-node-json-output',
                            style=styles['json-output']
                        ),
                        html.P('Edge Object JSON:'),
                        html.Pre(
                            id='tap-edge-json-output',
                            style=styles['json-output']
                        )
                    ])
                ])
            ]),
        ])
    ])
    @app.callback(Output('tap-node-json-output', 'children'), [Input('cytoscape', 'tapNode')])
    def display_tap_node(data):
        return json.dumps(data, indent=2)


    @app.callback(Output('tap-edge-json-output', 'children'), [Input('cytoscape', 'tapEdge')])
    def display_tap_edge(data):
        return json.dumps(data, indent=2)


    @app.callback(Output('cytoscape', 'layout'), [Input('dropdown-layout', 'value')])
    def update_cytoscape_layout(layout):
        return {'name': layout}
    
    app.run_server(debug=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="path to checkpoint dir")
    args = parser.parse_args()

    objective_node, _ = maybe_load_checkpoint(args.checkpoint_dir)

    visualize_tree(objective_node)


if __name__ == "__main__":
    main()