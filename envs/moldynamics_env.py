from collections import OrderedDict

import backoff
import gym
import numpy as np
import torch

from ase.db import connect
from sqlite3 import DatabaseError
from schnetpack.data.atoms import AtomsConverter

import warnings
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# For backoff exceptions
def on_giveup(details):
    print("Giving Up after {} tries. Time elapsed: {:.3f} :(".format(details['tries'], details['elapsed']))


class MolecularDynamics(gym.Env):
    metadata = {"render_modes":["human"], "name":"md_v0"}
    
    def __init__(self,
                 db_path,
                 converter, 
                 timelimit=10,
                 done_on_timelimit=False,
                 sample_initial_conformations=True,
                 num_initial_conformations=50000,
                 inject_noise=False,
                 noise_std=0.1,
                 remove_hydrogen=False):
        self.db_path = db_path
        self.converter = converter
        self.TL = timelimit
        self.done_on_timelimit = done_on_timelimit
        self.sample_initial_conformations = sample_initial_conformations
        self.inject_noise = inject_noise
        self.noise_std = noise_std
        self.remove_hydrogen = remove_hydrogen
        
        self.db_len = self.get_db_length()
        self.env_done = True
        self.atoms = None
        self.mean_energy = 0.
        self.std_energy = 1.
        self.initial_molecule_conformations = []

        # Store random subset of molecules DB
        self.get_initial_molecule_conformations(num_initial_conformations)
        self.conformation_idx = 0

        self.action_space = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict()

    def step(self, actions):
        self.atoms.set_positions(self.atoms.get_positions() + actions)

        self.env_steps += 1
        self.env_done = self.env_steps >= self.TL

        obs = self.converter(self.atoms)
        reward = None
        if self.done_on_timelimit:
            done = self.env_done
        else:
            done = False
        info = {'env_done': self.env_done}

        return obs, reward, done, info

    def reset(self):
        # If sample_initial_conformations iterate over all initial conformations sequentially
        if self.sample_initial_conformations:
            db_idx = np.random.randint(len(self.initial_molecule_conformations))
        else:
            db_idx = self.conformation_idx % len(self.initial_molecule_conformations)
            self.conformation_idx += 1
        row = self.initial_molecule_conformations[db_idx]

        # Copy to avoid changing the atoms object inplace
        self.atoms = row.toatoms().copy()

        # Remove hydrogen
        if self.remove_hydrogen:
                del self.atoms[[atom.index for atom in self.atoms if atom.symbol=='H']]

        # Check if row has Smiles
        if hasattr(row, 'smiles'):
            self.smiles = row.smiles
            self.energy = row.data['energy']

        # Inject noise into the initial state
        if self.inject_noise:
            current_positions = self.atoms.get_positions()
            noise = np.random.normal(scale=self.noise_std, size=current_positions.shape)
            self.atoms.set_positions(current_positions + noise)

        self.env_steps = 0
        self.env_done = False
        obs = self.converter(self.atoms)
        return obs

    def update_timelimit(self, new_timelimit):
        self.TL = new_timelimit

    def get_db_length(self):
        with connect(self.db_path) as conn:
            db_len = len(conn)
        return db_len

    def get_env_step(self):
        return self.env_steps

    def get_initial_molecule_conformations(self, num_initial_conformations):
        if num_initial_conformations == -1 or num_initial_conformations == self.db_len:
            indices = np.arange(1, self.db_len + 1)
        else:
            indices = np.random.choice(np.arange(1, self.db_len + 1), min(self.db_len, num_initial_conformations), replace=False)
        self.initial_molecule_conformations = []
        for idx in indices:
            row = self.get_molecule(int(idx))
            self.initial_molecule_conformations.append(row)
    
    # Makes sqllite3 database compatible with NFS storages
    @backoff.on_exception(
        backoff.expo,
        exception=DatabaseError,
        jitter=backoff.full_jitter,
        max_tries=10,
        on_giveup=on_giveup
    )
    def get_molecule(self, idx):
        with connect(self.db_path) as conn:
            return conn.get(idx)

    def get_atoms_num(self):
        return len(self.atoms.get_atomic_numbers())

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        return seed

def env_fn(**kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    converter = AtomsConverter(device=torch.device('cpu'))
    env = MolecularDynamics(converter=converter, **kwargs)
    return env