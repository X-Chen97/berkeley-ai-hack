import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import random
import io
import sys
import re
from pymatgen.core import Lattice, Structure
import crystal_toolkit.components as ctc
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
import torch

# Scatter plot

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

test_comps_energies = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [1.4, -0.5, -0.25, 0.5],
    "composition": ["LiFePO4", "LiMn2O4", "LiMnO3", "LiTiO4"],
    "below_threshold": ["False", "True", "True", "False"]
})

random_energies = []
for i in range(10):
    random_energies.append(random.uniform(-1.5, 1.5))

fig = px.scatter(
    test_comps_energies,
    x="x",
    y="y",
    color="below_threshold",
    custom_data=["composition"],
    title="Is your new composition thermodynamically stable?"
)

fig.update_layout(clickmode='event+select')

fig.update_traces(marker_size=20)

#from huggingface_hub import login

# use_huggingface = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if use_huggingface is False:
#    raise NotImplementedError()

# Login to Hugging Face
# os.system("huggingface-cli login --token $HUGGINGFACE_TOKEN")

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Roboto:wght@900&display=swap",
]

# model and tokenizer details
# tokenizer = ...
# model = ...
# model.to(device)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container(
    [
        html.H1(
            "Batteries by LLM",
            className="text-center mt-4",
            style={"font-family": "Roboto", "color": "green", "font-size": "48px"},
        ),
        html.H2(
            "CalHacks Team: Materials AI",
            className="text-center",
            style={"font-family": "Roboto", "color": "black", "font-size": "24px"},
        ),
        html.Div(
            "We use batteries in many of our day-to-day devices, and they are becoming increasingly important as we transition to emissions-free technology, like EVs. There are still a number of concerns for existing batteries (flammability, poor efficiency compared to other energy conversion methods) which motivates the continuing search for better battery materials. Leveraging the code-generating power of LLMs, we provide battery researchers with a convenient interface to query the viability of their candidate material."
        ),
        html.Div(
            "To screen for candidate materials that have desirable properties, computational materials scientists use a first-principles (meaning the only input is structural information about a given compound) modeling technique known as density functional theory (DFT). "
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Textarea(
                            id="chemical_formula",
                            value="LiFePO4",
                            style={"width": "15%", "height": "150px"},
                        ),
                        dcc.Textarea(
                            id="space_group",
                            value="Pnma",
                            style={"width": "15%", "height": "150px"},
                        ),
                        dcc.Textarea(
                            id="lattice_parameter",
                            value="8.4",
                            style={"width": "15%", "height": "150px"},
                        ),
                        html.Button("Submit", id="my-button", className="mt-2"),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="my-output",
                            style={
                                "width": "100%",
                                "height": "200px",
                                "border": "1px solid",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            id="eval-output",
                            style={
                                "width": "100%",
                                "height": "200px",
                                "border": "1px solid",
                                "padding": "10px",
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [html.Div(id="structure-output")],
                    width=4,
                ),
            ],
            className="align-items-center mt-4",
        ),
        dcc.Graph(
            id='compositions-energies',
            figure=fig
        ),

        # html.Div(className='row', children=[
        #     html.Div([
        #         dcc.Markdown("""
        #             **Hover Data**
        #
        #             Mouse over values in the graph.
        #         """),
        #         html.Pre(id='hover-data', style=styles['pre'])
        #     ], className='three columns'),
        #
        #     html.Div([
        #         dcc.Markdown("""
        #             **Click Data**
        #
        #             Click on points in the graph.
        #         """),
        #         html.Pre(id='click-data', style=styles['pre']),
        #     ], className='three columns'),
        #
        #     html.Div([
        #         dcc.Markdown("""
        #             **Selection Data**
        #
        #             Choose the lasso or rectangle tool in the graph's menu
        #             bar and then select points in the graph.
        #
        #             Note that if `layout.clickmode = 'event+select'`, selection data also
        #             accumulates (or un-accumulates) selected data if you hold down the shift
        #             button while clicking.
        #         """),
        #         html.Pre(id='selected-data', style=styles['pre']),
        #     ], className='three columns'),
        #
        #     html.Div([
        #         dcc.Markdown("""
        #             **Zoom and Relayout Data**
        #
        #             Click and drag on the graph to zoom or click on the zoom
        #             buttons in the graph's menu bar.
        #             Clicking on legend items will also fire
        #             this event.
        #         """),
        #         html.Pre(id='relayout-data', style=styles['pre']),
        #     ], className='three columns')
        # ]),
        dcc.Textarea(
            id="new_chemical_formula",
            value="LiFePO4",
            style={"width": "15%", "height": "150px"},
        ),
    ],
    fluid=True,
)


# @callback(
#     Output('hover-data', 'children'),
#     Input('compositions-energies', 'hoverData'))
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)
#
#
# @callback(
#     Output('click-data', 'children'),
#     Input('compositions-energies', 'clickData'))
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)
#
#
# @callback(
#     Output('selected-data', 'children'),
#     Input('compositions-energies', 'selectedData'))
# def display_selected_data(selectedData):
#     return json.dumps(selectedData, indent=2)
#
#
# @callback(
#     Output('relayout-data', 'children'),
#     Input('compositions-energies', 'relayoutData'))
# def display_relayout_data(relayoutData):
#     return json.dumps(relayoutData, indent=2)

# Input:
# Composition, Symmetry, lattic parameter (angle+length), space group

def build_poscar_from_user_input():
    # ... LLM generate
    return

def build_pymatgen_structure_from_poscar(poscar):
    # ...
    return structure

def call_llm(question):
    #when have model

    inputs = tokenizer(
        question, return_tensors="pt"
    )
    inputs.to(device)
    outputs = model.generate(**inputs, max_length=1024)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return answer


# @app.callback(
#     Output("my-output", "children"),
#     Output("eval-output", "children"),
#     Output("structure-output", "children"),
#     Input("my-button", "n_clicks"),
#     State("chemical_formula", "value"),
#     # State("space_group", "value"),
#     # State("lattice_parameter", "value")
# )
# def update_output(n_clicks, value):
#     if n_clicks is None:
#         # button has not been clicked yet
#         return "", "", ""
#     else:
#         # call LLM
#         sample_poscar = call_llm(value)
#         structure = build_pymatgen_structure_from_poscar(sample_poscar)
#         code_out = str(structure)
#         structure_component = ctc.StructureMoleculeComponent(structure)
#         samp_comm = re.findall("```(.*?)```", sample_script, re.DOTALL)
#         return (
#             code_out,
#             structure_component.layout(),
#         )

@app.callback(
    Output("compositions-energies", "figure"),
    Input("new_chemical_formula", "value")
)
def update_scatter_plot(value):
    new_energy = random.choice(random_energies)
    test_comps_energies.loc[len(test_comps_energies.index)] = {
        "x": test_comps_energies.iloc[-1]['x'] + 1,
        "y": new_energy,
        "composition": [value],
        "below_threshold": "True" if new_energy < 0 else "False"
    }
    fig = px.scatter(test_comps_energies, x="x", y="y", color="below_threshold", custom_data=["composition"])
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=20)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=80, host="0.0.0.0")
