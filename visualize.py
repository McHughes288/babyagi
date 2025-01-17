import argparse

import dash_cytoscape as cyto
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from utils import maybe_load_checkpoint

state2colour = {
    "new": "grey",
    "queued": "yellow",
    "cancelled": "red",
    "complete": "green",
}


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Dropdown(**kwargs),
        ],
    )


def DropdownOptionsList(*args):
    return [{"label": val.capitalize(), "value": val} for val in args]


def visualize_tree(checkpoint_dir):
    app = Dash(__name__)

    def generate_elements(node, depth=0, parent_id=None):
        elements = []
        stylesheet = []
        node_id = f"{depth}-{node.task_id}"
        elements.append(
            {
                "data": {
                    "id": node_id,
                    "label": int(node.task_id) + 1,
                    "task_name": node.task_name,
                    "result": node.result,
                    "state": node.state,
                }
            }
        )
        stylesheet.append(
            {
                "selector": f'node[id = "{node_id}"]',
                "style": {
                    "background-color": state2colour[node.state],
                    "opacity": 0.7,
                    "label": "data(label)",
                    "font-size": 20,
                },
            }
        )

        if parent_id is not None:
            elements.append({"data": {"source": parent_id, "target": node_id}})

        for child in node.children:
            child_elements, child_stylesheet = generate_elements(
                child, depth + 1, node_id
            )
            elements.extend(child_elements)
            stylesheet.extend(child_stylesheet)

        return elements, stylesheet

    def get_elements_from_checkpoint(checkpoint_dir):
        objective_node, _ = maybe_load_checkpoint(checkpoint_dir)
        if objective_node is not None:
            elements, stylesheet = generate_elements(objective_node)
        else:
            elements = []
            stylesheet = []
        return elements, stylesheet
    
    elements, stylesheet = get_elements_from_checkpoint(checkpoint_dir)

    stylesheet.append(
        {
            "selector": ":selected",
            "style": {
                "border-width": 2,
                "border-color": "black",
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "black",
                "font-size": 20,
                "z-index": 9999,
            },
        }
    )

    tab_style = {"height": "calc(98vh - 80px)"}

    def text_box_style(height="15%"):
        style = {
            "overflow-y": "scroll",
            "height": height,
            "border": "thin lightgrey solid",
            "white-space": "pre-wrap",
            "word-wrap": "break-word",
            "overflow-x": "hidden",
        }
        return style

    # fmt: off
    app.layout = html.Div([
        dcc.Interval(
            id='interval-component',
            interval=5*1000, # in milliseconds
            n_intervals=0
        ),
        html.Div(className="eight columns", children=[
            cyto.Cytoscape(
                id="cytoscape",
                elements=elements,
                stylesheet=stylesheet,
                layout={"name": "breadthfirst"},
                style={
                    "height": "95vh",
                    "width": "calc(100% - 500px)",
                    "float": "left"
                },
                responsive=True
            )
        ]),
        html.Div(className="four columns", children=[
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label="Content", children=[
                    html.Div(style=tab_style, children=[
                        html.P("Task:"),
                        html.Pre(
                            id="tap-node-task",
                            style=text_box_style("60px")  # 4 lines
                        ),
                        html.P("Result:"),
                        html.Pre(
                            id="tap-node-result",
                            style=text_box_style("45%")
                        ),
                        html.P("state:"),
                        html.Pre(
                            id="tap-node-state",
                            style=text_box_style("20px")
                        ),
                        html.P("Edge Content:"),
                        html.Pre(
                            id="tap-edge-json-output",
                            style=text_box_style("75px")  # 5 lines
                        )
                    ])
                ]),
                dcc.Tab(label="Control Panel", children=[
                    NamedDropdown(
                        name="Layout",
                        id="dropdown-layout",
                        options=DropdownOptionsList(
                            "breadthfirst",
                            "grid",
                            "circle",
                            "concentric",
                            "cose",
                        ),
                        value="breadthfirst",
                        clearable=False
                    ),
                ]),
            ]),
        ])
    ])
    # fmt: on

    @app.callback(Output("tap-node-task", "children"), [Input("cytoscape", "tapNode")])
    def display_task(data):
        task_name = ""
        if data:
            task_name = data["data"].get("task_name", "")
        return task_name

    @app.callback(
        Output("tap-node-result", "children"), [Input("cytoscape", "tapNode")]
    )
    def display_result(data):
        result = ""
        if data:
            result = data["data"].get("result", "")
        return result

    @app.callback(Output("tap-node-state", "children"), [Input("cytoscape", "tapNode")])
    def display_state(data):
        state = ""
        if data:
            state = data["data"].get("state", "")
        return state

    @app.callback(
        Output("tap-edge-json-output", "children"), [Input("cytoscape", "tapEdge")]
    )
    def display_tap_edge(data):
        if data:
            source_name = data["sourceData"].get("task_name", "")
            target_name = data["targetData"].get("task_name", "")
            return f"- TASK: {source_name}\n- SUBTASK: {target_name}"
        else:
            return ""

    @app.callback(Output("cytoscape", "layout"), [Input("dropdown-layout", "value")])
    def update_cytoscape_layout(layout):
        return {"name": layout}
    
    @app.callback(Output('cytoscape', 'elements'), Output('cytoscape', 'stylesheet'),
              Input('interval-component', 'n_intervals'))
    def update_metrics(n):
        elements, stylesheet = get_elements_from_checkpoint(checkpoint_dir)
        return elements, stylesheet

    app.run_server(debug=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="path to checkpoint dir")
    args = parser.parse_args()

    visualize_tree(args.checkpoint_dir)


if __name__ == "__main__":
    main()
