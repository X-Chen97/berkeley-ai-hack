import os
import sys
import json
import torch
import argparse
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', message='Too many lattice symmetries was found')

from pymatgen.ext.matproj import MPRester

# IOs
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

# House code
sys.path.append('./')
from relaxer import TrajectoryObserver, M3gnetRelaxer, ChgnetRelaxer, MaceRelaxer


mpr = MPRester(api_key="dPcAQJZ6y1NZidGuvTerIPPFXHtsOb3E")


def has_common_element(list1, list2):
    
    return not set(list1).isdisjoint(list2)


def read_json(fjson):
    
    with open(fjson) as f:
        return json.load(f)


def write_json(d, fjson):
    
    with open(fjson, 'w') as f:
        json.dump(d, f)

    return


def get_ehull(mpr, vasp_entry, mp_entries, compatibility):
    
    mp_entries = compatibility.process_entries(mp_entries)

    # phase diagram
    PD = PhaseDiagram(mp_entries + [vasp_entry])
    decomp_info = PD.get_decomp_and_e_above_hull(vasp_entry, allow_negative=True)
    e_above_hull = decomp_info[1]

    return e_above_hull


def get_test_structure(mpid='mp-510462'):
    
    mp_data = mpr.get_entry_by_material_id(material_id=[test_id])
    test_structure = mp_data[1].structure
    
    return test_structure


def get_chemsys_mpdata(structure):
    
    U_els = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9,
             'Mo': 4.38, 'Ni': 6.2, 'V': 4.2, 'W': 6.2}
    
    species = []
    potcar_spec = []

    for i in set(structure.species):
        species.append(i.name)

    chemical_space = '-'.join(species)
    mp_data = mpr.get_entries_in_chemsys(chemical_space)
    
    hubbards = {}
    if has_common_element(list(U_els.keys()), species):
        for specie in species:
            if specie in U_els.keys():
                hubbards[specie] = U_els[specie]
            else:
                hubbards[specie] = 0

    for d in mp_data:
        for j in d.parameters['potcar_spec']:
            if not j in potcar_spec:
                potcar_spec.append(j)
        if len(potcar_spec) == len(species):
            break

    return mp_data, potcar_spec, hubbards

def get_relaxation_result(structure):
        
    mp_data, potcar_spec, hubbards = get_chemsys_mpdata(structure)
    
    atoms = structure.to_ase_atoms()
    relaxer = MaceRelaxer()
    result = relaxer.relax(atoms, fmax=0.5)
    result['parameters'] = mp_data[0].parameters
    result['parameters']['potcar_spec'] = potcar_spec
    result['parameters']['hubbards'] = hubbards
    
    lattice = result['final_structure'].lattice.matrix
    species = [specie for specie in result['final_structure'].species]
    coords = [site.frac_coords for site in result['final_structure'].sites]
    structure = Structure(lattice=lattice, species=species, coords=coords)

    gga_entry = ComputedStructureEntry(structure=structure,
                                       energy=result['final_energy'],
                                       entry_id='llm_generation',
                                       composition=structure.composition.remove_charges(),
                                       parameters=result['parameters']
                                       )

    compat = MaterialsProject2020Compatibility()
    gga_entry = compat.process_entry(gga_entry)
    ehull = get_ehull(mpr, gga_entry, mp_data, compat)
    
    return result, ehull


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Get structure information.')
    
    parser.add_argument('-s', '--structure', type=str, required=True, 
                        help='Structure directory to process')

    # Parse the arguments
    args = parser.parse_args()

    # Access the structure information
    structure = args.structure
    structure = Structure.from_file(structure)
    
    result, ehull = get_relaxation_result(structure)
    
    trajectory = []
    aaa = AseAtomsAdaptor()
    
    for i in result['trajectory'].atoms_trajectory:
        relaxed_structure = aaa.get_structure(i)
        trajectory.append(relaxed_structure.as_dict())
        
    trajectory.appennd(ehull*1000)
    
    print(result)
    print("{} meV/atom".format(ehull*1000))

    write_json(trajectory, './trajectory.json')
    result['final_structure'].to(filename='./final_POSCAR', fmt='POSCAR')
