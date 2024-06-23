"""
Purpose: inherit relaxers to make fixed-atom relaxer.
Note: Force calculated on fixed atom is set to 0.
Need to generate forces using trajectory data set.
"""

import io
import sys
import torch
import inspect
import pickle
import logging
import contextlib
import numpy as np
from copy import deepcopy

# IOs
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor

# ASE
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.optimize import FIRE, BFGS, LBFGS
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter, ExpCellFilter

# Calculators
from chgnet.model import CHGNet
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp
from chgnet.model.dynamics import CrystalFeasObserver

# Relaxers
from chgnet.model.dynamics import StructOptimizer
from m3gnet.models import Relaxer

# To remove m3gnet error
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
}

sys.path.append("../")

class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """
    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.atoms_trajectory = []
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """The logic for saving the properties of an Atoms during the relaxation."""
        
        self.atoms_trajectory.append(deepcopy(self.atoms))
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def to_dict(self) -> dict:
        """
        Change to dictionary.
        Note that Numpy array is not JSON serializable.
        Need additional encoder.
        """

        out_dict = {"energy": self.energies,
                    "forces": self.forces,
                    "stresses": self.stresses,
                    "atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers()
                    }

        return out_dict

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MaceRelaxer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model='saved',
        optimizer_class="FIRE",
        device='cpu'
    ) -> None:
        """MACE Relaxer

        Args:
            model (CHGNet):
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class
        self.device = device

        if model == 'mp2M':
            self.calculator = mace_mp(model="https://tinyurl.com/y7uhwpje", device=self.device)
        elif model == 'saved':
            self.calculator = MACECalculator(model_paths='../pretrained_model/2023-12-03-mace-128-L1_epoch-199.model',
                                             device=device,
                                             datatype = 'float64')
        else:
            raise ValueError("Model not found.")
        print(f"MACE will run on {self.device}")

    def relax(
        self,
        atoms,
        fmax=0.1,
        steps=500,
        relax_cell=False,
        save_path=None,
        loginterval = 1,
        verbose=True,
        fixatoms=[],
        **kwargs,
    ):
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal feas
                Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            fixatoms (list): Index of atoms to fix during relaxation.
                Default = []
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        if isinstance(atoms, Structure):
            atoms = atoms.to_ase_atoms()

        atoms.calc = self.calculator  # assign model used to predict forces
        constraint = FixAtoms(indices=fixatoms)
        atoms.set_constraint(constraint)
        setted_constraint = atoms.constraints[0].get_indices()

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if relax_cell:
                atoms = FrechetCellFilter(atoms)

            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)

        return {"final_structure": struct, "final_energy": atoms.get_total_energy() , "trajectory": obs}


class ChgnetRelaxer(StructOptimizer):

    def __init__(self):

        super().__init__()
        torch.set_default_dtype(torch.float32)

    def relax(
            self,
            atoms=Atoms,
            fmax=0.1,
            steps=500,
            relax_cell=False,
            ase_filter="FrechetCellFilter",
            save_path=None,
            loginterval=1,
            crystal_feas_save_path=None,
            verbose=True,
            fixatoms=[],
            **kwargs,
    ):
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            ase_filter (str | ase.filters.Filter): The filter to apply to the atoms
                object for relaxation. Default = FrechetCellFilter
                Used to default to ExpCellFilter but was removed due to bug reported in
                https://gitlab.com/ase/ase/-/issues/1321 and fixed in
                https://gitlab.com/ase/ase/-/merge_requests/3024.
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal feas
                Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            fixatoms (list): Index of atoms to fix during relaxation. Not
                compatible with relax_cell = True.
                Default = []
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        try:
            import ase.filters as filter_classes
            from ase.filters import Filter

        except ImportWarning:
            import ase.constraints as filter_classes
            from ase.constraints import Filter

            if ase_filter == "FrechetCellFilter":
                ase_filter = "ExpCellFilter"
            print(
                "Failed to import ase.filters. Default filter to ExpCellFilter. "
                "For better relaxation accuracy with the new FrechetCellFilter,"
                "Run pip install git+https://gitlab.com/ase/ase"
            )
        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filter_classes, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filter_classes, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces
        constraint = FixAtoms(indices=fixatoms)
        atoms.set_constraint(constraint)
        setted_constraint = atoms.constraints[0].get_indices()

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)

            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for key in struct.site_properties:
            struct.remove_site_property(property_name=key)
        struct.add_site_property(
            "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
        )
        return {"final_structure": struct, "trajectory": obs}


class M3gnetRelaxer(Relaxer):

    def __init__(self, relax_cell=False):

        super().__init__(relax_cell=relax_cell)

    def relax(
            self,
            atoms=Atoms,
            fmax=0.1,
            steps=500,
            traj_file=None,
            interval=1,
            verbose=False,
            fixatoms=[],
            **kwargs,
    ):
        """
        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            **kwargs:
        Returns:
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        constraint = FixAtoms(indices=fixatoms)
        atoms.set_constraint(constraint)
        setted_constraint = atoms.constraints[0].get_indices()

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = FrechetCellFilter(atoms)
            optimizer = self.opt_class(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms

        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }

