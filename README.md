# berkeley-ai-hack

Repo for Materials LLM!

## Inspiration
As our climate changes, we need to continue adopting sustainable technologies for energy conversion. As materials scientists, we want to leverage cutting-edge scientific approaches to find the right materials for these needs. To mitigate carbon pollution, industries need to transition to batteries, the most efficient of which are lithium-ion batteries. As researchers, we use first principles atomistic modeling to understand materials properties, such as conductivity and stability; however, building these models is time consuming and presents significant learning curves for non-computational scientists. To bridge the gap between ideation and discovery for new lithium-ion battery opportunities, we fine-tuned an LLM model to automatically suggest structural input files for first principles modeling techniques from natural text descriptions. Our web app allows users to visualize the suggested structure, which is further optimized to reflect a real-world material.

We believe that our tool will help scientists and researchers discover the best lithium-ion battery electrolytes to compete in the energy industry!

## What it does
Researchers input a new lithium-ion battery electrolyte material and its space group. A natural text description is generated via a rule-based crystallography tool. This description is fed into a Llama model that is fine-tuned to generate the initial suggested structure file (in VASP POSCAR format) for atomistic modeling. This structure is then relaxed using a DFT-based GNN model to reflect the closest physically possible structure. The LLM-suggested structure and the DFT-relaxed structure are visualized for comparison. Finally, the stability (i.e., its viability as a lithium-ion electrolyte) of this candidate material is calculated and depicted graphically, which a user can then compare to other materials' stabilities.

## How we built it
We queried the Materials Project for a dataset of battery materials and released publicly on HuggingFaceHub. Using a Gaudi card from the Intel Cloud, we fine-tuned a Llama-2-7B model that converts natural text to modeling input files that reflect atomic positions (POSCAR file). We then used existing pre-trained GNN models (multi-atomic cluster expansion AKA MACE) to relax the LLM-generated structure to be more physically reasonable. We built a Dash app for users to explore the lithium-ion battery dataset and our structures generated by LLMs.

Out dataset and model can be found at Hugging Face Hub!
Dataset: [MaterialsAI/robocr_poscar_2col](https://huggingface.co/datasets/MaterialsAI/robocr_poscar_2col)
Model: [MaterialsAI/robocr_poscar_2col_llama](https://huggingface.co/MaterialsAI/robocr_poscar_2col_llama)
We run our LLM on *One Intel Gaudi* at http://146.152.224.107:8017/docs

## Challenges we ran into
Materials dataset complexity is not immediately useful for LLM inference, so this required deep domain expertise in crystallography as well as proper LLM prompt engineering. DFT input files for complex materials contain numerous atomic coordinates that are based on crystallographic rules and set based on symmetry of crystals; however, for LLM doing text-based prediction accurately setting these sites is challenging. This difficulty is why we included a relaxation step to modify the LLM-generated structures to something more physically reasonable.

## Accomplishments that we're proud of
Contributing to the discovery of new lithium-ion battery materials to fight climate change. Fusing generative AI, LLMs, and complex atomistic modeling. Building an intuitive user interface that will help scientists overcome learning barriers for materials modeling. Developing a complex end-to-end pipeline from unstructured text input to LLM to GNN to final materials visualization and evaluation that is usable by scientists.

## What we learned
Generative AI can be effectively adopted into a specific scientific domains. LLMs can be used as a starting point to generate possible materials structures, even if they require further optimization. We learned how to leverage Intel Development Cloud and Gaudi cards to fine-tune LLMs and perform further inference.

## What's next for Batteries by LLM
We want to extend our pipeline beyond Li-ion materials to Na- and Mg-based batteries, as well as other sustainable technology sectors like solar energy and soft materials.

## Built With
```dash```
```gaudi```
```huggingface```
```intel```
```llama```
```python```
```vasp```
