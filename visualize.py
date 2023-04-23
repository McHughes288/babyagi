import argparse
from dash import Dash, html, dcc
import dash_cytoscape as cyto
from utils import maybe_load_checkpoint

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

    tab_style = {'height': 'calc(98vh - 80px)'}
    
    def text_box_style(height="15%"):

        style = {
            'overflow-y': 'scroll',
            'height': height,
            'border': 'thin lightgrey solid',
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word',
            'overflow-x': 'hidden'
        }
        return style

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
                dcc.Tab(label='Content', children=[
                    html.Div(style=tab_style, children=[
                        html.P('Task:'),
                        html.Pre(
                            id='tap-node-task',
                            style=text_box_style("60px") # 4 lines
                        ),
                        html.P('Result:'),
                        html.Pre(
                            id='tap-node-result',
                            style=text_box_style("45%")
                        ),
                        html.P('state:'),
                        html.Pre(
                            id='tap-node-state',
                            style=text_box_style("20px")
                        ),
                        html.P('Edge Content:'),
                        html.Pre(
                            id='tap-edge-json-output',
                            style=text_box_style("75px") # 5 lines
                        )
                    ])
                ]),
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
            ]),
        ])
    ])

    @app.callback(Output('tap-node-task', 'children'), [Input('cytoscape', 'tapNode')])
    def display_task(data):
        task_name = ""
        if data:
            task_name = data["data"].get("task_name", "")
        return task_name
    
    @app.callback(Output('tap-node-result', 'children'), [Input('cytoscape', 'tapNode')])
    def display_result(data):
        result = ""
        if data:
            result = data["data"].get("result", "")
        return result
    
    @app.callback(Output('tap-node-state', 'children'), [Input('cytoscape', 'tapNode')])
    def display_state(data):
        state = ""
        if data:
            state = data["data"].get("state", "")
        return state

    @app.callback(Output('tap-edge-json-output', 'children'), [Input('cytoscape', 'tapEdge')])
    def display_tap_edge(data):
        if data:
            source_name = data["sourceData"].get("task_name", "")
            target_name = data["targetData"].get("task_name", "")
            return f"- TASK: {source_name}\n- SUBTASK: {target_name}"
        else:
            return ""


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