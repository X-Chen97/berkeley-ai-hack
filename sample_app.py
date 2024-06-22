import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import io
import sys
import re
from pymatgen.core import Lattice, Structure
import crystal_toolkit.components as ctc
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
import torch

#from huggingface_hub import login

# use_huggingface = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            "CLIMATE",
            className="text-center mt-4",
            style={"font-family": "Roboto", "color": "green", "font-size": "48px"},
        ),
        html.H2(
            "Team Best",
            className="text-center",
            style={"font-family": "Roboto", "color": "black", "font-size": "24px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Textarea(
                            id="my-input",
                            value="copper",
                            style={"width": "100%", "height": "150px"},
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
    ],
    fluid=True,
)

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


@app.callback(
    Output("my-output", "children"),
    Output("eval-output", "children"),
    Output("structure-output", "children"), 
    Input("my-button", "n_clicks"),
    State("my-input", "value"),
)
def update_output(n_clicks, value):
    if n_clicks is None:
        # button has not been clicked yet
        return "", "", ""
    else:
        # call LLM
        sample_poscar = call_llm(value)
        structure = build_pymatgen_structure_from_poscar(sample_poscar)
        code_out = str(structure)
        structure_component = ctc.StructureMoleculeComponent(structure)
        samp_comm = re.findall("```(.*?)```", sample_script, re.DOTALL)
        return (
            code_out,
            structure_component.layout(),
        )


if __name__ == "__main__":
    app.run_server(debug=True, port=80, host="0.0.0.0")