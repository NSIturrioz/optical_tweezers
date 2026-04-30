import os
import json
import hashlib
from datetime import datetime
import numpy as np
import h5py

# ---------------------------
# Utility: hash input parameters
# ---------------------------
def generate_input_hash(input_dict):
    # Convert to JSON string (sorted for consistency)
    input_str = json.dumps(input_dict, sort_keys=True)
    return hashlib.md5(input_str.encode()).hexdigest()


# ---------------------------
# Main save function
# ---------------------------
def save_simulation(data_folder, inputs, results):
    
    os.makedirs(data_folder, exist_ok=True)
    
    # Generate unique hash
    sim_hash = generate_input_hash(inputs)
    
    filename = f"sim_{sim_hash}.h5"
    filepath = os.path.join(data_folder, filename)
    
    # Check if already exists
    if os.path.exists(filepath):
        print("Simulation already exists. File not created.")
        return
    
    # Create file
    with h5py.File(filepath, "w") as f:
        
        # ---------------------------
        # Save metadata (inputs)
        # ---------------------------
        input_group = f.create_group("inputs")
        for key, value in inputs.items():
            input_group.attrs[key] = json.dumps(value)
        
        # Save date
        f.attrs["date"] = datetime.now().isoformat()
        f.attrs["hash"] = sim_hash
        
        # ---------------------------
        # Save results
        # ---------------------------
        results_group = f.create_group("results")
        
        for key, value in results.items():
            results_group.create_dataset(key, data=value, compression="gzip")
    
    print(f"Simulation saved: {filename}")