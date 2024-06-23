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
from pprint import pprint

from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset
from dotenv import load_dotenv
import requests
# hf_api_key = ""
# HfFolder.save_token(hf_api_key)
# hf_api = HfApi()
# dataset = load_dataset(dataset_name)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, prepare_model_for_kbit_training
# import torch

app = dash.Dash(prevent_initial_callbacks=True)

use_huggingface = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if use_huggingface is False:
    raise NotImplementedError()

load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")
HfFolder.save_token(hf_api_key)
hf_api = HfApi()
dataset = load_dataset('MaterialsAI/robocr_poscar_2col')

# Scatter plot

###############################################################################
###############################################################################
###########################    Testing           ##############################
###############################################################################
###############################################################################

with open('../data/formula_spacegroup_description_map.json', 'r') as fp:
    formula_spacegroup_description_map = json.load(fp)

formula_spacegroup_description_map['Li4MnCo2NiO8_P2/m'] = "Li₄MnCo₂NiO₈ is Caswellsilverite-derived structured and crystallizes in the monoclinic P2/m space group. There are three inequivalent Li¹⁺ sites. In the first Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with three equivalent MnO₆ octahedra, corners with three equivalent NiO₆ octahedra, an edgeedge with one MnO₆ octahedra, an edgeedge with one NiO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 1-9°. There are a spread of Li-O bond distances ranging from 2.01-2.29 Å. In the second Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with six equivalent CoO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with four equivalent MnO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles are 11°. There are two shorter (2.13 Å) and four longer (2.14 Å) Li-O bond lengths. In the third Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with six equivalent CoO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with four equivalent NiO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 0-6°. There are four shorter (2.14 Å) and two longer (2.21 Å) Li-O bond lengths. Mn⁴⁺ is bonded to six O²⁻ atoms to form MnO₆ octahedra that share corners with six equivalent LiO₆ octahedra, edges with two equivalent MnO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 1-9°. There are two shorter (1.93 Å) and four longer (1.97 Å) Mn-O bond lengths. Co³⁺ is bonded to six O²⁻ atoms to form CoO₆ octahedra that share corners with six LiO₆ octahedra, edges with two equivalent MnO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with two equivalent NiO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 0-11°. There are a spread of Co-O bond distances ranging from 1.85-2.13 Å. Ni²⁺ is bonded to six O²⁻ atoms to form NiO₆ octahedra that share corners with six equivalent LiO₆ octahedra, edges with two equivalent NiO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 7-9°. There are four shorter (2.07 Å) and two longer (2.12 Å) Ni-O bond lengths. There are four inequivalent O²⁻ sites. In the first O²⁻ site, O²⁻ is bonded to three Li¹⁺, two equivalent Mn⁴⁺, and one Co³⁺ atom to form a mixture of corner and edge-sharing OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-8°. In the second O²⁻ site, O²⁻ is bonded to three Li¹⁺, one Mn⁴⁺, and two equivalent Co³⁺ atoms to form OLi₃MnCo₂ octahedra that share corners with six OLi₃MnCo₂ octahedra and edges with twelve OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-11°. In the third O²⁻ site, O²⁻ is bonded to three Li¹⁺, one Co³⁺, and two equivalent Ni²⁺ atoms to form a mixture of corner and edge-sharing OLi₃CoNi₂ octahedra. The corner-sharing octahedral tilt angles range from 0-8°. In the fourth O²⁻ site, O²⁻ is bonded to three Li¹⁺, two equivalent Co³⁺, and one Ni²⁺ atom to form OLi₃Co₂Ni octahedra that share corners with six OLi₃MnCo₂ octahedra and edges with twelve OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-11°."

# now we give a list of structures to pick from
with open('../script/trajectory.json', 'r') as fp:
    structures = [Structure.from_dict(s) for s in json.load(fp)]

# we show the first structure by default
generated_structure_component = ctc.StructureMoleculeComponent(
    structures[0],
    id="generated_structure"
)
actual_structure_component = ctc.StructureMoleculeComponent(
    Structure.from_file("../data/Li4MnCo2NiO8_POSCAR"),
    id="actual_structure"
)

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
            "We developed an AI-powered web app that uses large language models (LLMs) to predict and visualize the stability of new lithium-based battery materials. By inputting a candidate material's formula and space group, our system generates a structural model using LLMs, optimizes it, and evaluates its thermodynamic stability, helping to identify promising materials for safer and more efficient batteries. Battery scientists and researchers can now leverage the generative power of LLMs to accelerate discoveries in sustainable energy technology."
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
                                value="Li4MnCo2NiO8",
                                style={"width": "50%", "height": "50px"}
                            )
                        ], justify="center"),
                        dbc.Row([
                            html.H3("Space Group", className="text-center"),
                            dcc.Textarea(
                                id="space_group",
                                value="P2/m",
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
                                value="Li₄MnCo₂NiO₈ is Caswellsilverite-derived structured and crystallizes in the monoclinic P2/m space group. There are three inequivalent Li¹⁺ sites. In the first Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with three equivalent MnO₆ octahedra, corners with three equivalent NiO₆ octahedra, an edgeedge with one MnO₆ octahedra, an edgeedge with one NiO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 1-9°. There are a spread of Li-O bond distances ranging from 2.01-2.29 Å. In the second Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with six equivalent CoO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with four equivalent MnO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles are 11°. There are two shorter (2.13 Å) and four longer (2.14 Å) Li-O bond lengths. In the third Li¹⁺ site, Li¹⁺ is bonded to six O²⁻ atoms to form LiO₆ octahedra that share corners with six equivalent CoO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with four equivalent NiO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 0-6°. There are four shorter (2.14 Å) and two longer (2.21 Å) Li-O bond lengths. Mn⁴⁺ is bonded to six O²⁻ atoms to form MnO₆ octahedra that share corners with six equivalent LiO₆ octahedra, edges with two equivalent MnO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 1-9°. There are two shorter (1.93 Å) and four longer (1.97 Å) Mn-O bond lengths. Co³⁺ is bonded to six O²⁻ atoms to form CoO₆ octahedra that share corners with six LiO₆ octahedra, edges with two equivalent MnO₆ octahedra, edges with two equivalent CoO₆ octahedra, edges with two equivalent NiO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 0-11°. There are a spread of Co-O bond distances ranging from 1.85-2.13 Å. Ni²⁺ is bonded to six O²⁻ atoms to form NiO₆ octahedra that share corners with six equivalent LiO₆ octahedra, edges with two equivalent NiO₆ octahedra, edges with four equivalent CoO₆ octahedra, and edges with six LiO₆ octahedra. The corner-sharing octahedral tilt angles range from 7-9°. There are four shorter (2.07 Å) and two longer (2.12 Å) Ni-O bond lengths. There are four inequivalent O²⁻ sites. In the first O²⁻ site, O²⁻ is bonded to three Li¹⁺, two equivalent Mn⁴⁺, and one Co³⁺ atom to form a mixture of corner and edge-sharing OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-8°. In the second O²⁻ site, O²⁻ is bonded to three Li¹⁺, one Mn⁴⁺, and two equivalent Co³⁺ atoms to form OLi₃MnCo₂ octahedra that share corners with six OLi₃MnCo₂ octahedra and edges with twelve OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-11°. In the third O²⁻ site, O²⁻ is bonded to three Li¹⁺, one Co³⁺, and two equivalent Ni²⁺ atoms to form a mixture of corner and edge-sharing OLi₃CoNi₂ octahedra. The corner-sharing octahedral tilt angles range from 0-8°. In the fourth O²⁻ site, O²⁻ is bonded to three Li¹⁺, two equivalent Co³⁺, and one Ni²⁺ atom to form OLi₃Co₂Ni octahedra that share corners with six OLi₃MnCo₂ octahedra and edges with twelve OLi₃Mn₂Co octahedra. The corner-sharing octahedral tilt angles range from 0-11°.",
                                style={"width": "80%", "height": "200px"},
                            ),
                        ], justify="center"),
                        dbc.Row([
                            html.H3("Atomic coordinates generated by LLM", className="text-center"),
                            html.P(
                                "LLM-generated atomic coordinates",
                                className="text-center",
                                style={'font-size': "10px"}
                            ),
                            dcc.Textarea(
                                id="generated_poscar",
                                value="",
                                style={"width": "80%", "height": "200px"},
                            ),
                        ], justify="center", style={'margin-top': '10px'}),
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
                                generated_structure_component.layout(),
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
                            id="generated_structure_output",
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H3("Actual crystal structure from Materials Project", className="text-center"),
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
                                actual_structure_component.layout(),
                            ],
                            id="actual_structure",
                            style={
                                'margin-bottom': '40px',
                                'margin-left': '30px'
                            }
                        )
                    ],
                    width=4,
                )
            ],
            className="align-items-center mt-4",
        ),
        # dbc.Row([
        #     dbc.Col(
        #         [
        #             dcc.Graph(
        #                 id='compositions-energies',
        #                 figure=fig
        #             ),
        #         ]
        #     )
        # ])
    ],
    fluid=True,
)

###############################################################################
###############################################################################
###########################    Function Pipeline ##############################
###############################################################################
##############################################################################
def call_llm(input):

    url = f"http://146.152.224.107:8017/generate"
    response = requests.post(
        url,
        json={
            'inputs': input,
            'parameters': {"max_new_tokens": 1024}
        }
    )
    generated_poscar = response.json()

    return generated_poscar['generated_text']

def call_gnn_relaxer(input):
    url = f"http://146.152.224.107:8019/predict_predict_post"
    response = requests.post(
        url,
        json={
            'inputs': input
        }
    )
    trajectory_energy = response.json()
    return trajectory_energy

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

@app.callback(
    Output("generated_poscar", "value"),
    Input("add-new-candidate-button", "n_clicks"),
    State("robocryst_description", "value"),
    prevent_initial_call=True
)
def run_llm_inference(n_clicks, value):
    poscar = call_llm(value)
    stability = call_gnn_relaxer(poscar)
    print("Length of Trajectory")
    return poscar

ctc.register_crystal_toolkit(app=app, layout=app.layout)

# for the interactivity, we use a standard Dash callback
@app.callback(
    Output(generated_structure_component.id(), "data"),
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
