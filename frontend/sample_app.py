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
# from mp_api.client import MPRester
from pymatgen.core import Lattice, Structure
import crystal_toolkit.components as ctc
import os

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, prepare_model_for_kbit_training
# import torch

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

with open('../data/formula_spacegroup_description_map.json', 'r') as fp:
    formula_spacegroup_description_map = json.load(fp)

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
    color="below_threshold"
)

fig.update_layout(
    clickmode='event+select',
    title={'text': "Is your candidate thermodynamically stable?",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )

fig.update_traces(marker_size=20)

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
            "We developed an AI-powered web app that uses large language models (LLMs) to predict and visualize the stability of new lithium-based battery materials. By inputting a material's formula and space group, our system generates a structural model using LLMs, optimizes it, and evaluates its thermodynamic stability, helping to identify promising materials for safer and more efficient batteries. This tool leverages AI to accelerate discoveries in sustainable energy technology."
        ),
        html.P(
            "Description generated by ChatGPT",
            className="text-center",
            style={'font-size': "10px"}
        ),
        html.Hr(style={
            "border": 0,
            "clear":"both",
            "display":"block",
            "width": "96%",
            "background-color":"#FFFF00",
            "height": "1px"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([
                            html.H3("Chemical Formula", className="text-center"),
                            dcc.Textarea(
                                id="chemical_formula",
                                value="CsLi(B3O5)2",
                                style={"width": "50%", "height": "50px"}
                            )
                        ], justify="center"),
                        dbc.Row([
                            html.H3("Space Group", className="text-center"),
                            dcc.Textarea(
                                id="space_group",
                                value="I-42d",
                                style={"width": "50%", "height": "50px"},
                            ),
                        ], justify="center"),
                        # dbc.Row([
                        #     html.H3("Lattice Parameter", className="text-center"),
                        #     dcc.Textarea(
                        #         id="lattice_parameter",
                        #         value="8.4",
                        #         style={"width": "50%", "height": "50px"},
                        #     ),
                        # ], justify="center"),
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
                              'margin-top':'10px',
                              'margin-bottom': '25px',
                              'margin-left': '60px'}
                        ),
                        dbc.Row([
                            html.H3("Description generated by Robocrystallographer", className="text-center"),
                            html.P(
                                "A. Ganose & A. Jain (2019). Robocrystallographer: Automated crystal structure text descriptions and analysis. MRS Communications, 9(3), 874-881.",
                                className="text-center",
                                style={'font-size': "10px"}
                            ),
                            dcc.Textarea(
                                id="robocryst_description",
                                value="CsLiB₆O₁₀ crystallizes in the tetragonal I̅42d space group. Cs¹⁺ is bonded in a 8-coordinate geometry to eight O²⁻ atoms. There are four shorter (3.29 Å) and four longer (3.59 Å) Cs-O bond lengths. Li¹⁺ is bonded to four equivalent O²⁻ atoms to form distorted LiO₄ trigonal pyramids that share corners with four equivalent BO₄ tetrahedra. All Li-O bond lengths are 1.97 Å. There are two inequivalent B³⁺ sites. In the first B³⁺ site, B³⁺ is bonded to four O²⁻ atoms to form BO₄ tetrahedra that share corners with two equivalent LiO₄ trigonal pyramids. All B-O bond lengths are 1.48 Å. In the second B³⁺ site, B³⁺ is bonded in a trigonal planar geometry to three O²⁻ atoms. There are a spread of B-O bond distances ranging from 1.36-1.40 Å. There are three inequivalent O²⁻ sites. In the first O²⁻ site, O²⁻ is bonded in a bent 120 degrees geometry to two equivalent Cs¹⁺ and two equivalent B³⁺ atoms. In the second O²⁻ site, O²⁻ is bonded in a bent 120 degrees geometry to one Cs¹⁺ and two B³⁺ atoms. In the third O²⁻ site, O²⁻ is bonded in a trigonal planar geometry to one Li¹⁺ and two B³⁺ atoms.",
                                style={"width": "80%", "height": "200px"},
                            ),
                        ], justify="center"),

                    ],
                    width=3
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H3("Crystal structure generated by fine-tuned LLM", className="text-center"),
                                html.P(
                                    "Visualization interface provided by Materials Project",
                                    className="text-center",
                                    style={'font-size': "15px"}
                                ),
                                html.P(
                                    "A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. S., G. Ceder, and K. A. Persson (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. MRS APL Materials, 1, 011002.",
                                    className="text-center",
                                    style={'font-size': "10px"}
                                ),
                                structure_component.layout(),
                                dcc.Slider(
                                    id="relaxation_slider",
                                    min=0,
                                    max=len(structures)-1,
                                    value=0,
                                    marks={
                                        0: "unrelaxed",
                                        len(structures)-1: "relaxed"
                                    }
                                )
                            ],
                            id="structure_output",
                            # style={'margin-bottom': '10px',
                            #       'textAlign':'center',
                            #       'width': '500px',
                            #       'margin-top':'10px',
                            #       'margin-bottom': '25px',
                            #       'margin-left': '60px'}
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id='compositions-energies',
                            figure=fig
                        ),
                    ], width=5
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

@app.callback(
    Output("robocryst_description", "value"),
    Input("add-new-candidate-button", "n_clicks"),
    [
        State("chemical_formula", "value"),
        State("space_group", "value")
    ],
    prevent_initial_call=True
)
def update_robocyrsdescr(n_clicks, chemical_formula_value, space_group_value):
    key = f"{chemical_formula_value}_{space_group_value}"
    if key in formula_spacegroup_description_map:
        description = formula_spacegroup_description_map[key]
    else:
        description = "Could not find corresponding description! Either try again, or create your own!"
    return description

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
