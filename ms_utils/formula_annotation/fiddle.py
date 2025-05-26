# ...existing code...

# --- Import FIDDLE Modules ---
# Note: This assumes FIDDLE modules are importable (e.g., FIDDLE_DIR is in sys.path)
try:
    from utils.mol_utils import vector_to_formula, formula_to_vector, formula_to_dict
    from utils.msms_utils import filter_spec, ce2nce, mass_calculator
    from utils.pkl_utils import generate_ms, melt_neutral_precursor
    from utils.refine_utils import formula_refinement
    from model_tcn import MS2FNet_tcn, FDRNet
    # from dataset import MGFDataset # Used for config loading logic if needed
# ...existing code...

# --- Helper Functions ---

def load_config(config_path):
    """Loads the YAML configuration file and prepares necessary keys."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}")
        sys.exit(1)

    # Add atom_dict and atom_list to config for convenience
    # Extracting from 'mona_qtof' as it's in the specified config file.
    # This might need adjustment if using different config files.
    try:
        # Assuming the relevant atom list is under a dataset key like 'mona_qtof'
        dataset_key = 'mona_qtof' # Adjust if your config uses a different key
        if dataset_key not in config:
            # Fallback or error if the expected dataset key isn't present
            print(f"Warning: Dataset key '{dataset_key}' not found in config. Trying to find 'atom_type' at top level.")
            if 'atom_type' in config:
                 mol_elements = config['atom_type']
            else:
                 # Attempt to find atom_type in the first dictionary entry if possible
                 first_dict_key = next((k for k, v in config.items() if isinstance(v, dict) and 'atom_type' in v), None)
                 if first_dict_key:
                     print(f"Using atom_type from '{first_dict_key}' section.")
                     mol_elements = config[first_dict_key]['atom_type']
                 else:
                     print(f"Error: Could not find 'atom_type' list in the configuration file '{config_path}'.")
                     sys.exit(1)
        else:
            mol_elements = config[dataset_key]['atom_type']

        if not isinstance(mol_elements, list):
             print(f"Error: Expected 'atom_type' to be a list, but got {type(mol_elements)}.")
             sys.exit(1)

        config['mol_elements'] = mol_elements # Store the found list
        config['atom_dict'] = formula_to_dict(config['mol_elements'])
        config['atom_list'] = list(config['atom_dict'].keys())
    except NameError:
         print("Error: 'formula_to_dict' function not found. Make sure it's imported correctly from utils.mol_utils.")
         sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key '{e}' while trying to extract atom types from config.")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating atom_dict/atom_list from config: {e}")
        sys.exit(1)

    # Add other potentially missing keys used by model constructors if they aren't present
    # These might have defaults in the model class, but we add them for clarity if needed.
    # config.setdefault('property_dim', config['model']['output_dim']) # Example if needed
    # config.setdefault('env_dim', config['model']['embedding_dim']) # Example if needed

    return config

def load_models(config, model_path, fdr_model_path, device):
    """Loads the pre-trained FIDDLE models using nested config parameters."""
    try:
        # --- Formula Prediction Model ---
        # Map nested config keys to the expected constructor arguments
        # Note: This mapping involves assumptions about the MS2FNet_tcn constructor.
        # If the constructor expects different arguments (e.g., the lists directly),
        # this needs adjustment.
        model_cfg = config['model']
        ms2f_params = {
            'input_dim': model_cfg['input_channels'], # Assuming input_dim corresponds to input_channels
            'hidden_dim': model_cfg['tcn_channels'][-1] if model_cfg.get('tcn_channels') else 128, # Assumption: use last channel size as hidden_dim
            'output_dim': model_cfg['output_dim'],
            'n_layers': len(model_cfg['tcn_channels']) if model_cfg.get('tcn_channels') else 6, # Assumption: derive from channels list length
            'kernel_size': model_cfg['tcn_kernel_sizes'][0] if model_cfg.get('tcn_kernel_sizes') else 45, # Assumption: use first kernel size
            'dropout': model_cfg['tcn_dropout'],
            # Assuming property_dim relates to output_dim or embedding_dim
            'property_dim': model_cfg.get('property_dim', model_cfg['output_dim']),
             # Assuming env_dim relates to embedding_dim or specific embedding dims combined
            'env_dim': model_cfg.get('env_dim', model_cfg['embedding_dim']),
            'pool_method': model_cfg.get('pool_method', 'max') # Provide default if missing
        }
        print(f"Debug: MS2FNet_tcn params: {ms2f_params}") # Debug print
        model = MS2FNet_tcn(**ms2f_params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # --- FDR Model ---
        # Map nested config keys to the expected constructor arguments
        # Similar assumptions as above apply to FDRNet constructor.
        fdr_params = {
            'input_dim': model_cfg['input_channels'], # Assuming input_dim corresponds to input_channels
            'hidden_dim': model_cfg['fdr_decoder_layers'][0] if model_cfg.get('fdr_decoder_layers') else 128, # Assumption: use first decoder layer size
            'output_dim': 1, # Predicts a single score
            'n_layers': len(model_cfg['fdr_decoder_layers']) if model_cfg.get('fdr_decoder_layers') else 5, # Assumption: derive from layers list length
            'kernel_size': model_cfg.get('fdr_kernel_size', 5), # Provide default if missing
            'dropout': model_cfg.get('fdr_dropout', model_cfg['tcn_dropout']), # Provide default if missing (use main dropout?)
            'property_dim': model_cfg['output_dim'], # Takes formula vector as property
            'env_dim': model_cfg.get('env_dim', model_cfg['embedding_dim']), # Assuming env_dim relates to embedding_dim
            'pool_method': model_cfg.get('fdr_pool_method', 'max') # Provide default if missing
        }
        print(f"Debug: FDRNet params: {fdr_params}") # Debug print
        fdr_model = FDRNet(**fdr_params).to(device)
        fdr_model.load_state_dict(torch.load(fdr_model_path, map_location=device))
        fdr_model.eval()

    except KeyError as e:
        print(f"Error: Missing key '{e}' in config['model'] section while loading models.")
        print(f"Model config section: {config.get('model', 'Not Found')}")
        sys.exit(1)
    except TypeError as e:
        print(f"Error: TypeError during model initialization. Check if constructor arguments match config: {e}")
        print(f"MS2FNet params used: {ms2f_params}")
        print(f"FDRNet params used: {fdr_params}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    return model, fdr_model

def preprocess_spectrum(mz_list, intensity_list, precursor_mz, precursor_type, collision_energy, config):
    """Preprocesses raw spectrum data into model input tensors."""
    # Ensure inputs are numpy arrays
    mz_array = np.array(mz_list, dtype=np.float32)
    intensity_array = np.array(intensity_list, dtype=np.float32)

    # 1. Process Spectrum (Normalization, etc.) - Mimics part of MGFDataset loading
    processed_spec = filter_spec(mz_array, intensity_array, precursor_mz)

    # 2. Generate Binned Spectrum Array (`spec`)
    try:
        spec_arr = generate_ms(
            processed_spec,
            max_mz=config['encoding']['max_mz'],
            resolution=config['encoding']['resolution']
        )
    except KeyError as e:
        print(f"Error: Missing key '{e}' in config['encoding'] section during spectrum generation.")
        sys.exit(1)


    # 3. Generate Environment Vector (`env`)
    try:
        nce = ce2nce(collision_energy, precursor_mz, precursor_type) # Calculate NCE
        env_arr = melt_neutral_precursor(
            precursor_mz,
            nce,
            precursor_type,
            config['encoding']['precursor_type'] # Mapping from config
        )
    except KeyError as e:
        print(f"Error: Missing key '{e}' in config['encoding'] section during environment vector generation.")
        sys.exit(1)


    # 4. Convert to Tensors
    spec_tensor = torch.from_numpy(spec_arr).float().unsqueeze(0).to(DEVICE) # Add batch dim
    env_tensor = torch.from_numpy(env_arr).float().unsqueeze(0).to(DEVICE)   # Add batch dim

    return spec_tensor, env_tensor

def predict_formula_for_row(row_data, model, fdr_model, config):
    """Predicts formula for a single row of data."""
    try:
        # --- 1. Extract and Preprocess ---
        spec_tensor, env_tensor = preprocess_spectrum(
            row_data[MZ_LIST_COL],
            row_data[INTENSITY_LIST_COL],
            row_data[PRECURSOR_MZ_COL],
            row_data[PRECURSOR_TYPE_COL],
            row_data[COLLISION_ENERGY_COL],
            config
        )

        # --- 2. Initial Prediction ---
        with torch.no_grad():
            y_pred, mass_pred, atom_num_pred, hc_ratio_pred = model(spec_tensor, env_tensor)

        # Use the top prediction vector
        formula_vector_pred = y_pred[0].cpu().numpy()
        # 'atom_list' should be added by load_config
        initial_formula = vector_to_formula(formula_vector_pred, config['atom_list'])
        if not initial_formula:
             return {"error": "Initial prediction failed"} # Handle cases where no formula is generated

        # --- 3. Refinement ---
        neutral_mass = mass_calculator(row_data[PRECURSOR_MZ_COL], row_data[PRECURSOR_TYPE_COL])
        if neutral_mass is None:
            return {"error": f"Could not calculate neutral mass for type {row_data[PRECURSOR_TYPE_COL]}"}

        try:
            refine_params = {
                'f0_list': [initial_formula], # Start refinement from the top prediction
                'M': neutral_mass,
                'mass_tolerance': config['post_processing']['mass_tolerance'],
                'ppm_mode': config['post_processing']['ppm_mode'],
                'top_k': config['post_processing']['top_k'],
                # 'D': config['post_processing']['maxium_miss_atom_num'], # Map D to maxium_miss_atom_num if appropriate, else remove
                'T': config['post_processing']['time_out'],
                'atom_type': config['post_processing']['refine_atom_type'],
                'atom_num': config['post_processing']['refine_atom_num'],
                'atom_dict': config['atom_dict'] # Added by load_config
                # buddy_path/sirius_path integration omitted for simplicity
            }
            # Remove D if formula_refinement doesn't expect it or if maxium_miss_atom_num isn't the right mapping
            # Check formula_refinement definition for expected parameters. Assuming 'D' is not expected for now.

            refined_results = formula_refinement(**refine_params)

        except KeyError as e:
             print(f"Error: Missing key '{e}' in config['post_processing'] section during refinement.")
             return {"error": f"Missing config key for refinement: {e}"}


        if not refined_results['formula']: # Check if refinement yielded results
             return {"error": "Refinement yielded no formulas"}

        # --- 4. FDR Reranking ---
        # Prepare inputs for FDR model (needs batch dimension)
        num_candidates = len(refined_results['formula'])
        spec_batch = spec_tensor.repeat(num_candidates, 1, 1)
        env_batch = env_tensor.repeat(num_candidates, 1)

        # Convert refined formulas back to vectors for FDR model input
        formula_vectors = []
        valid_indices = [] # Keep track of formulas that could be vectorized
        for idx, f in enumerate(refined_results['formula']):
            # 'atom_dict' added by load_config, 'output_dim' from model config
            vec = formula_to_vector(f, config['atom_dict'], config['model']['output_dim'])
            if vec is not None:
                formula_vectors.append(vec)
                valid_indices.append(idx)
            else:
                print(f"Warning: Could not vectorize refined formula '{f}' for FDR scoring.")

        if not formula_vectors:
             return {"error": "Could not vectorize any refined formulas for FDR"}

        # Filter results to only include valid formulas for FDR
        refined_formulas_valid = [refined_results['formula'][i] for i in valid_indices]
        refined_masses_valid = [refined_results['mass'][i] for i in valid_indices]

        formula_tensor = torch.from_numpy(np.array(formula_vectors)).float().to(DEVICE)

        # Filter spec and env tensors if needed (though they were repeated based on original count)
        if len(valid_indices) < num_candidates:
             spec_batch = spec_batch[valid_indices]
             env_batch = env_batch[valid_indices]

        with torch.no_grad():
            fdr_scores = fdr_model(spec_batch, env_batch, formula_tensor).cpu().numpy().flatten()

        # Combine results and sort by FDR score
        reranked_results = sorted(zip(fdr_scores, refined_formulas_valid, refined_masses_valid), key=lambda x: x[0])

        # Format output (Top K results after reranking)
        # Use a report_top_k if defined, else refine_top_k from post_processing
        top_k = config.get('report_top_k', config['post_processing']['top_k'])
        final_results = {
            "spectrum_id": row_data.get(SPECTRUM_ID_COL, None), # Include ID if present
            "top_k_formulas": [res[1] for res in reranked_results[:top_k]],
            "top_k_masses": [res[2] for res in reranked_results[:top_k]],
            "top_k_fdr": [res[0] for res in reranked_results[:top_k]],
            "error": None
        }

    except KeyError as e:
        print(f"Error processing row {row_data.get(SPECTRUM_ID_COL, 'N/A')}: Missing config key {e}")
        final_results = {
            "spectrum_id": row_data.get(SPECTRUM_ID_COL, None),
            "top_k_formulas": [], "top_k_masses": [], "top_k_fdr": [], "error": f"Missing config key: {e}"
        }
    except Exception as e:
        print(f"Error processing row {row_data.get(SPECTRUM_ID_COL, 'N/A')}: {e}")
        import traceback
        traceback.print_exc()
        final_results = {
            "spectrum_id": row_data.get(SPECTRUM_ID_COL, None),
            "top_k_formulas": [], "top_k_masses": [], "top_k_fdr": [], "error": str(e)
        }

    return final_results

# --- Main Execution ---
# ... (rest of the script remains the same) ...
if __name__ == "__main__":
    # --- Load Input Data ---
    # Replace this with your actual Polars DataFrame loading logic
    # Example: df = pl.read_csv("your_input_data.csv")
    # Ensure your DataFrame has columns matching the constants defined above
    # Example DataFrame structure:
    data = {
        SPECTRUM_ID_COL: ["spec1", "spec2"],
        MZ_LIST_COL: [[85.0296, 129.0196], [100.1, 150.2, 200.3]],
        INTENSITY_LIST_COL: [[100.0, 8.03], [50.0, 100.0, 25.0]],
        PRECURSOR_MZ_COL: [129.01941, 250.1234],
        PRECURSOR_TYPE_COL: ["[M-H]-", "[M+H]+"],
        COLLISION_ENERGY_COL: [50.0, 30.0],
    }
    df = pl.DataFrame(data)
    print("Input DataFrame:")
    print(df)

    # --- Load Config and Models ---
    print("Loading config...")
    config = load_config(CONFIG_PATH)
    # print(f'Loaded config: {config}') # Optional: print loaded config for debugging
    print("Loading models...")
    model, fdr_model = load_models(config, MODEL_WEIGHTS_PATH, FDR_MODEL_WEIGHTS_PATH, DEVICE)
    print(f"Using device: {DEVICE}")

    # --- Process DataFrame ---
    results_list = []
    print("Processing spectra...")
    # Iterate through rows (Polars encourages expression-based operations,
    # but row-by-row is simpler for this complex function application)
    for row_dict in df.to_dicts():
        result = predict_formula_for_row(row_dict, model, fdr_model, config)
        results_list.append(result)

    # --- Combine Results ---
    results_df = pl.DataFrame(results_list)

    print("\nPrediction Results:")
    print(results_df)

    # Optional: Join results back to the original DataFrame
    # Ensure SPECTRUM_ID_COL exists and is unique in both DataFrames
    if SPECTRUM_ID_COL in df.columns and SPECTRUM_ID_COL in results_df.columns and results_df[SPECTRUM_ID_COL].is_not_null().all():
         final_df = df.join(results_df, on=SPECTRUM_ID_COL, how="left")
         print("\nJoined DataFrame:")
         print(final_df)
    else:
         print("\nCould not join results (SPECTRUM_ID_COL missing or has nulls in results).")
         print("Original DF columns:", df.columns)
         print("Results DF columns:", results_df.columns)


    print("\nProcessing complete.")
