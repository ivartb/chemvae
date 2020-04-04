from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu

import numpy as np


vae = VAEUtils(directory='../models/zinc_properties')

f = open("moleculs.out", "w")
for line in open("moleculs.in", "r"):
    smiles = mu.canon_smiles(line.strip())
    print("Input SMILES:", smiles, file=f)

    X = vae.smiles_to_hot(smiles, canonize_smiles=True)
    encoded = vae.encode(X)
    print("Encoded representation: shape =", encoded.shape, ", norm =", np.linalg.norm(encoded), file=f)

    props = vae.predict_prop_Z(encoded)
    print('Properties (qed, SAS, logP):', props, file=f)

    noise = 5.0
    print("Searching molecules randomly sampled from", noise, "std (z-distance) from the point", file=f)
    df = vae.z_to_smiles(encoded, decode_attempts=100, noise_norm=noise)
    print("Found", len(set(df['smiles'])), "unique mols, out of" ,sum(df['count']), file=f)
    print('Output SMILES:\n', df.smiles, file=f)
    print("\n==================================================\n", file=f)