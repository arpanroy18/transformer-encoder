"""
Configuration module for the Transformer model.

This module centralizes all hyperparameters and settings for the Transformer
decoder model, facilitating easy management and modification. It includes
parameters related to model architecture (dimensions, layers, heads),
dropout rates, positional encoding, training settings, KV cache usage,
and other miscellaneous constants.

The `get_config()` function provides a convenient way to access all
these parameters as a dictionary.
"""

# Model Dimensions
vocab_size = 50257  # Vocabulary size (e.g., GPT-2's vocab size for broad compatibility)
d_model = 768       # Embedding dimension / Hidden size of the model
n_layers = 12       # Number of decoder layers stacked in the model
n_heads = 12        # Number of attention heads in the multi-head attention mechanism
d_head = d_model // n_heads  # Dimension of each individual attention head (d_model must be divisible by n_heads)
d_ff = d_model * 4  # Dimension of the hidden layer in the feed-forward network (typically 4 * d_model)

# Dropout Rates
# These are applied during training to prevent overfitting.
embed_dropout = 0.1     # Dropout rate for embeddings
attn_dropout = 0.1      # Dropout rate for attention layers
ff_dropout = 0.1        # Dropout rate for the feed-forward network layers
residual_dropout = 0.1  # Dropout rate for residual connections

# Positional Encoding
max_seq_len = 1024  # Maximum sequence length the model can process

# Training Hyperparameters (placeholders, primarily used if full training is implemented)
batch_size = 32         # Number of sequences processed in one training iteration
learning_rate = 3e-4    # Learning rate for the optimizer
num_epochs = 5          # Number of times the entire dataset is processed during training
warmup_steps = 2000     # Number of steps for learning rate warmup (if implemented)
gradient_clip_val = 1.0 # Value for gradient clipping to prevent exploding gradients

# KV Cache
use_kv_cache = True # Flag to enable/disable Key-Value caching during generation

# Other
epsilon = 1e-5  # Small constant for numerical stability in Layer Normalization

# Add unified dropout_rate if not present
dropout_rate = 0.1

def get_config(overrides: dict = None) -> dict:
    """Returns a dictionary containing all defined configuration parameters.

    Allows overriding default parameters by passing a dictionary.

    Parameters
    ----------
    overrides : dict, optional
        A dictionary of parameters to override the defaults.

    Returns
    -------
    dict
        A dictionary where keys are parameter names (strings) and
        values are their corresponding settings, updated with overrides.
    """
    config = {
        # Model Dimensions
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        # Calculate d_head based on possibly overridden d_model/n_heads
        # This requires accessing possibly overridden values carefully
        # Let's calculate it after merging overrides for simplicity
        # "d_head": d_head, # Defer calculation
        "d_ff": d_ff,
        # Dropout Rates - Add a general one if used by modules
        "dropout_rate": dropout_rate,
        "embed_dropout": embed_dropout,
        "attn_dropout": attn_dropout,
        "ff_dropout": ff_dropout,
        "residual_dropout": residual_dropout,
        # Positional Encoding
        "max_seq_len": max_seq_len,
        # Training Hyperparameters
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "gradient_clip_val": gradient_clip_val,
        # KV Cache
        "use_kv_cache": use_kv_cache,
        # Other
        "epsilon": epsilon,
    }

    if overrides:
        config.update(overrides)
        
        # Recalculate dependent parameters like d_head if dimensions were overridden
        if "d_model" in config and "n_heads" in config:
            if config["d_model"] % config["n_heads"] != 0:
                 # Warn or raise error if overridden values are incompatible
                 print(f"Warning: Overridden d_model ({config['d_model']}) is not divisible by n_heads ({config['n_heads']}). d_head calculation may be incorrect or cause errors.")
                 # Assign a default or placeholder, or let it fail later?
                 # For now, let's calculate anyway, downstream code should handle errors.
            config["d_head"] = config["d_model"] // config["n_heads"]
        elif "d_head" not in config: # Ensure d_head exists even if overrides didn't include d_model/n_heads
             config["d_head"] = config["d_model"] // config["n_heads"] # Use original calculation if possible

    # Ensure d_head is present if it wasn't calculated via overrides
    if "d_head" not in config:
        config["d_head"] = config["d_model"] // config["n_heads"]

    return config

if __name__ == "__main__":
    # Validation Test
    default_config = get_config()
    print("Default Transformer Configuration:")
    for key, value in default_config.items():
        print(f"- {key}: {value}")
    assert default_config['d_model'] == d_model
    assert default_config['d_head'] == d_model // n_heads

    # Test overrides
    test_overrides = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "learning_rate": 1e-5,
        "max_seq_len": 128
    }
    overridden_config = get_config(test_overrides)
    print("\nOverridden Configuration:")
    for key, value in overridden_config.items():
        print(f"- {key}: {value}")
        
    assert overridden_config['d_model'] == 64
    assert overridden_config['n_layers'] == 2
    assert overridden_config['n_heads'] == 2
    assert overridden_config['learning_rate'] == 1e-5
    assert overridden_config['max_seq_len'] == 128
    # Check recalculated d_head
    assert overridden_config['d_head'] == 64 // 2, f"Overridden d_head calculation failed. Expected 32, Got {overridden_config['d_head']}"
    # Check a parameter that wasn't overridden
    assert overridden_config['epsilon'] == epsilon

    print("\nConfiguration module with overrides seems to be working correctly.") 