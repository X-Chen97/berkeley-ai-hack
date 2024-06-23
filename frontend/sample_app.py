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
from mp_api.client import MPRester
from pymatgen.core import Lattice, Structure
import crystal_toolkit.components as ctc
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
import torch

app = dash.Dash(prevent_initial_callbacks=True)


#from huggingface_hub import login

# use_huggingface = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if use_huggingface is False:
#    raise NotImplementedError()

# Login to Hugging Face
# os.system("huggingface-cli login --token $HUGGINGFACE_TOKEN")

# Scatter plot

###############################################################################
###############################################################################
###########################    Testing           ##############################
###############################################################################
###############################################################################

test_comps_energies = pd.DataFrame({
    "x": ["LiFePO4", "LiMn2O4", "LiMnO3", "LiTiO4"],
    "y": [1.4, -0.5, -0.25, 0.5],
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
    title="Is your new composition thermodynamically stable?"
)

fig.update_layout(clickmode='event+select')

fig.update_traces(marker_size=20)

# now we give a list of structures to pick from
structures = [
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.25, 0.5]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.25, 0.25, 0.5]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.25, 0.52]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.25, 0.25, 0.5]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.52, 0.25, 0.5]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.25, 0.5, 0.25]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.52, 0.5]]),
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
]

# we show the first structure by default
structure_component = ctc.StructureMoleculeComponent(structures[0], id="my_structure")

# model and tokenizer details
# tokenizer = ...
# model = ...
# model.to(device)

###############################################################################
###############################################################################
###########################    Layout            ##############################
###############################################################################
###############################################################################

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Roboto:wght@900&display=swap",
]

# and we create a button for user interaction
my_button = html.Button("Swap Structure", id="change_structure_button")

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
                        dbc.Row([
                            html.H3("Chemical Formula", className="text-center"),
                            dcc.Textarea(
                                id="chemical_formula",
                                value="LiFePO4",
                                style={"width": "50%", "height": "50px"}
                            )
                        ], justify="center"),
                        dbc.Row([
                            html.H3("Space Group", className="text-center"),
                            dcc.Textarea(
                                id="space_group",
                                value="Pnma",
                                style={"width": "50%", "height": "50px"},
                            ),
                        ], justify="center"),
                        dbc.Row([
                            html.H3("Lattice Parameter", className="text-center"),
                            dcc.Textarea(
                                id="lattice_parameter",
                                value="8.4",
                                style={"width": "50%", "height": "50px"},
                            ),
                        ], justify="center"),
                        html.Div([
                            html.Button(
                                "Submit",
                                id="add-new-candidate-button",
                                className="text-center",
                                n_clicks=0
                            )
                        ], style={'margin-bottom': '10px',
                              'textAlign':'center',
                              'width': '220px',
                              'margin':'auto'}
                        )
                    ],
                    width=2
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                structure_component.layout(),
                                dcc.Slider(
                                    id="relaxation_slider",
                                    min=0,
                                    max=len(structures)-1,
                                    value=0
                                )
                            ], id="structure_output")
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id='compositions-energies',
                            figure=fig
                        ),
                        # dcc.Textarea(
                        #     id="new_chemical_formula",
                        #     value="LiFePO4",
                        #     style={"width": "15%", "height": "150px"},
                        # )
                    ]
                )

            ],
            className="align-items-center mt-4",
        )
    ],
    fluid=True,
)

###############################################################################
###############################################################################
###########################    Function Pipeline ##############################
###############################################################################
###############################################################################

# Input:
# Composition, Symmetry, lattic parameter (angle+length), space group
def call_llm(question):
    #when have model

    inputs = tokenizer(
        question, return_tensors="pt"
    )
    inputs.to(device)
    outputs = model.generate(**inputs, max_length=1024)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return answer

def build_pymatgen_structure_from_poscar(poscar_file):

    return structure

###############################################################################
###############################################################################
###########################    Call Backs        ##############################
###############################################################################
###############################################################################

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

ctc.register_crystal_toolkit(app=app, layout=app.layout)

# for the interactivity, we use a standard Dash callback
@app.callback(
    Output(structure_component.id(), "data"),
    [Input("relaxation_slider", "value")],
    prevent_initial_call=True
)
def update_structure(slider_value):
    # with MPRester("wOqaB77pMUmeQ7fY59PZCxdVeLd30eM2") as mpr:
    #     mp_id = mpr.materials.search()
    return structures[int(slider_value)]

@app.callback(
    Output("compositions-energies", "figure"),
    Input("add-new-candidate-button", "n_clicks"),
    [
        State("chemical_formula", "value")
    ],
    prevent_initial_call=True
)
def update_scatter_plot(n_clicks, value):
    new_energy = random.choice(random_energies)
    test_comps_energies.loc[len(test_comps_energies.index)] = {
        "x": value,
        "y": new_energy,
        "below_threshold": "True" if new_energy < 0 else "False"
    }
    fig = px.scatter(test_comps_energies, x="x", y="y", color="below_threshold")
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=20)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=80, host="0.0.0.0")
