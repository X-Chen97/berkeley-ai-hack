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

app = dash.Dash(prevent_initial_callbacks=True)

# now we have two entries in our app layout,
# the structure component's layout and the button

#from huggingface_hub import login

# use_huggingface = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if use_huggingface is False:
#    raise NotImplementedError()

# Login to Hugging Face
# os.system("huggingface-cli login --token $HUGGINGFACE_TOKEN")

# Scatter plot

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

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
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    Structure(Lattice.cubic(5), ["K", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
]

# we show the first structure by default
structure_component = ctc.StructureMoleculeComponent(structures[0], id="my_structure")

# and we create a button for user interaction
my_button = html.Button("Swap Structure", id="change_structure_button")

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
                        html.Button(
                            "Submit",
                            id="add-new-candidate-button",
                            className="mt-2",
                            n_clicks=0
                            ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [html.Div([structure_component.layout(), my_button], id="structure_output")],
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

ctc.register_crystal_toolkit(app=app, layout=app.layout)

# for the interactivity, we use a standard Dash callback
@app.callback(
    Output(structure_component.id(), "data"),
    [Input("change_structure_button", "n_clicks")],
    prevent_initial_call=True
)
def update_structure(n_clicks):
    return structures[n_clicks % 2]


if __name__ == "__main__":
    app.run_server(debug=True, port=80, host="0.0.0.0")
