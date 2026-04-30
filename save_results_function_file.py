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
    clean_dict = convert_to_serializable(input_dict)
    input_str = json.dumps(clean_dict, sort_keys=True)
    return hashlib.md5(input_str.encode()).hexdigest()

# ---------------------------
# Main save function
# ---------------------------
def save_simulation(data_folder, inputs, results):
    
    os.makedirs(data_folder, exist_ok=True)
    
    # Generate hash
    sim_hash = generate_input_hash(inputs)
    
    filename = f"sim_{sim_hash}.h5"
    filepath = os.path.join(data_folder, filename)
    
    # Check if already exists
    if os.path.exists(filepath):
        print("Simulation already exists. File not created.")
        return
    
    with h5py.File(filepath, "w") as f:
        
        # --- Save inputs ---
        input_group = f.create_group("inputs")
        for key, value in inputs.items():
            clean_value = convert_to_serializable(value)
            input_group.attrs[key] = json.dumps(clean_value)
        
        # --- Save metadata ---
        f.attrs["date"] = datetime.now().isoformat()
        f.attrs["hash"] = sim_hash
        
        # --- Save results ---
        results_group = f.create_group("results")
        for key, value in results.items():
            results_group.create_dataset(key, data=value, compression="gzip")
    
    print(f"Simulation saved: {filename}")

# ---------------------------
# Convert everything to python native types
# ---------------------------
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj

# ---------------------------
# Search for specific simulation
# ---------------------------
def find_simulations(folder, conditions):
    matches = []
    
    for filename in os.listdir(folder):
        if not filename.endswith(".h5"):
            continue
        
        filepath = os.path.join(folder, filename)
        
        with h5py.File(filepath, "r") as f:
            inputs = {k: json.loads(v) for k, v in f["inputs"].attrs.items()}
            
            match = True
            for key, condition in conditions.items():
                
                if key not in inputs:
                    match = False
                    break
                
                value = inputs[key]
                
                # Condition can be a function
                if callable(condition):
                    if not condition(value):
                        match = False
                        break
                else:
                    if value != condition:
                        match = False
                        break
            
            if match:
                matches.append(filepath)
    
    return matches