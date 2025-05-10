"""Full Transformer Decoder model implementation.

This module defines the `Decoder` class, which stacks multiple `DecoderBlock`
layers to form a complete Transformer decoder. It also includes a simple `Linear`
layer class used for the final output projection to vocabulary logits.

The `Decoder` handles:
- Input embedding and positional encoding.
- Forward pass through a stack of decoder blocks.
- Final linear projection to logits.
- Optional KV caching for efficient step-by-step generation.
- Collection of parameters and gradients from all sub-modules.
"""
import numpy as np
import config
import embedding
import positional_encoding
import decoder_block
import multi_head_attention

# If Tokenizer is needed for type hinting or other purposes:
# import tokenizer

class Linear:
    """A simple fully connected linear layer.

    Implements `output = input @ W + b`.

    Parameters
    ----------
    d_input : int
        Dimensionality of the input features.
    d_output : int
        Dimensionality of the output features.

    Attributes
    ----------
    d_input : int
        Input dimensionality.
    d_output : int
        Output dimensionality.
    W : np.ndarray
        Weight matrix of shape `(d_input, d_output)`.
    b : np.ndarray
        Bias vector of shape `(1, d_output)`.
    input_cache : np.ndarray or None
        Cache for input `x` from the last forward pass.
    grad_W : np.ndarray or None
        Gradient of the loss with respect to `W`.
    grad_b : np.ndarray or None
        Gradient of the loss with respect to `b`.
    """
    def __init__(self, d_input: int, d_output: int):
        """Initializes the Linear layer.

        Weights are initialized using Glorot uniform initialization.
        Biases are initialized to zeros.

        Parameters
        ----------
        d_input : int
            Dimensionality of the input.
        d_output : int
            Dimensionality of the output.
        """
        self.d_input = d_input
        self.d_output = d_output
        limit = np.sqrt(6. / (d_input + d_output))
        self.W = np.random.uniform(-limit, limit, (d_input, d_output))
        self.b = np.zeros((1, d_output))
        self.input_cache = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass: `output = x @ W + b`.

        Parameters
        ----------
        x : np.ndarray
            Input tensor. Expected shape `(..., d_input)`, where `...` indicates
            any number of leading dimensions (e.g., batch_size, seq_len).

        Returns
        -------
        np.ndarray
            Output tensor of shape `(..., d_output)`.
        """
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def get_parameters(self, prefix: str = "") -> dict:
        """Retrieves the learnable parameters (W, b).

        Parameters
        ----------
        prefix : str, optional
            Prefix for parameter names, by default "".

        Returns
        -------
        dict
            Dictionary mapping parameter names to values.
        """
        return {
            f"{prefix}W": self.W,
            f"{prefix}b": self.b
        }

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Computes gradients for the linear layer.

        Calculates dL/dW, dL/db, and dL/dx.

        Parameters
        ----------
        d_output : np.ndarray
            Gradient of the loss with respect to the output of this layer.
            Shape `(..., d_output)`.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input `x` of this layer.
            Shape `(..., d_input)`.
        """
        original_input_shape = self.input_cache.shape
        if self.input_cache.ndim > 2:
            input_reshaped = self.input_cache.reshape(-1, self.d_input)
            d_output_reshaped = d_output.reshape(-1, self.d_output)
        else:
            input_reshaped = self.input_cache
            d_output_reshaped = d_output

        self.grad_W = np.dot(input_reshaped.T, d_output_reshaped)
        self.grad_b = np.sum(d_output_reshaped, axis=0, keepdims=True)

        d_input_reshaped = np.dot(d_output_reshaped, self.W.T)
        d_input = d_input_reshaped.reshape(original_input_shape[:-1] + (self.d_input,))
        
        return d_input

    def get_gradients(self, prefix: str = "") -> dict:
        """Retrieves computed gradients for W and b.

        Parameters
        ----------
        prefix : str, optional
            Prefix for gradient names, by default "".

        Returns
        -------
        dict
            Dictionary mapping gradient names to values.
            Initializes to zeros if `backward` not called.
        """
        if self.grad_W is None: self.grad_W = np.zeros_like(self.W)
        if self.grad_b is None: self.grad_b = np.zeros_like(self.b)
        return {
            f"{prefix}W": self.grad_W,
            f"{prefix}b": self.grad_b
        }

class Decoder:
    """Full Transformer Decoder model.

    Combines input embedding, positional encoding, a stack of DecoderBlocks,
    and a final linear layer to produce vocabulary logits.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 max_seq_len: int, epsilon: float = 1e-5, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.input_embedding = embedding.InputEmbedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionalEncoding(max_seq_len, d_model)
        self.decoder_blocks = [
            decoder_block.DecoderBlock(d_model, n_heads, d_ff, epsilon, dropout_rate=dropout_rate) for _ in range(n_layers)
        ]
        self.final_linear_layer = Linear(d_model, vocab_size)
        self.attention_weights = [] # To store attention weights from each block
        self.training_mode = False # Add training_mode flag

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Computes softmax numerically stably along the last axis."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, token_ids: np.ndarray, mask: np.ndarray = None, kv_cache_list: list = None, current_token_idx: int = 0, training_mode: bool = False) -> np.ndarray:
        """Performs the forward pass for the entire Decoder."""
        self.training_mode = training_mode # Store training_mode
        single_sequence_input = False
        if token_ids.ndim == 1:
            token_ids = np.expand_dims(token_ids, axis=0)
            single_sequence_input = True

        batch_size, seq_len = token_ids.shape
        
        x = self.input_embedding.forward(token_ids)
        
        pe_offset = current_token_idx if kv_cache_list is not None and seq_len == 1 else 0
        x = self.positional_encoding.forward(x, offset=pe_offset)
        
        self.attention_weights = [] 

        for i, block in enumerate(self.decoder_blocks):
            layer_kv_cache = kv_cache_list[i] if kv_cache_list is not None else None
            block_mask = None if kv_cache_list is not None and seq_len == 1 else mask
            x = block.forward(x, mask=block_mask, layer_kv_cache=layer_kv_cache, training_mode=training_mode)
            if hasattr(block, 'attention_weights') and block.attention_weights is not None:
                 self.attention_weights.append(block.attention_weights)
            else:
                 self.attention_weights.append(np.array([]))

        logits = self.final_linear_layer.forward(x)

        if single_sequence_input:
            logits = logits.squeeze(axis=0)
            if logits.ndim == 1 and self.vocab_size == 1: 
                logits = np.expand_dims(logits, axis=0)

        return logits

    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """Creates a causal mask for self-attention."""
        return multi_head_attention.create_causal_mask(seq_len)

    def get_parameters(self, prefix: str = "decoder") -> dict:
        """Retrieves all learnable parameters."""
        params = {}
        params.update(self.input_embedding.get_parameters(prefix=f"{prefix}_input_embedding"))
        for i, block in enumerate(self.decoder_blocks):
            params.update(block.get_parameters(prefix=f"{prefix}_block{i}"))
        params.update(self.final_linear_layer.get_parameters(prefix=f"{prefix}_final_linear"))
        return params

    def backward(self, d_logits: np.ndarray): # Removed -> None, should return dict or rely on get_gradients
        """Performs backpropagation for the entire Decoder model."""
        if d_logits.ndim == 2 and hasattr(self.input_embedding, 'input_cache') and \
           self.input_embedding.input_cache is not None and self.input_embedding.input_cache.ndim == 3:
             if self.input_embedding.input_cache.shape[0] == 1:
                d_logits = np.expand_dims(d_logits, axis=0) 

        d_x_after_blocks = self.final_linear_layer.backward(d_logits)

        for i in range(self.n_layers - 1, -1, -1):
            block = self.decoder_blocks[i]
            d_x_after_blocks = block.backward(d_x_after_blocks)

        d_embedded = d_x_after_blocks
        if hasattr(self.positional_encoding, 'backward'): # If PE had a backward pass
             d_embedded = self.positional_encoding.backward(d_embedded)
        
        if hasattr(self.input_embedding, 'backward'):
            self.input_embedding.backward(d_embedded)
        
        # Gradients are stored in sub-modules and collected by get_gradients()

    def get_gradients(self, prefix: str = "decoder") -> dict:
        """Retrieves all computed gradients."""
        grads = {}
        if hasattr(self.input_embedding, 'get_gradients'):
            grads.update(self.input_embedding.get_gradients(prefix=f"{prefix}_input_embedding"))
        for i, block in enumerate(self.decoder_blocks):
            if hasattr(block, 'get_gradients'):
                grads.update(block.get_gradients(prefix=f"{prefix}_block{i}"))
        if hasattr(self.final_linear_layer, 'get_gradients'):
            grads.update(self.final_linear_layer.get_gradients(prefix=f"{prefix}_final_linear"))
        return grads

if __name__ == "__main__":
    print("Decoder Validation Block")
    import sys, os
    # Simplified path logic
    current_dir = os.getcwd()
    project_root = current_dir
    # Navigate up if current_dir is 'src' or 'tests' etc. to find project_root
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # These imports are now absolute, assuming src is in sys.path
    # config, embedding, positional_encoding, decoder_block, multi_head_attention already imported at top
    import tokenizer # For the __main__ block

    config_params = config.get_config()
    
    d_model = config_params['d_model']
    n_layers = config_params['n_layers']
    n_heads = config_params['n_heads']
    d_ff = config_params['d_ff']
    max_seq_len = config_params['max_seq_len']
    epsilon = config_params['epsilon']
    # Ensure dropout_rate is in config or provide a default
    dropout_rate = config_params.get('dropout_rate', 0.1)


    sample_corpus = "This is a test. This is only a test."
    test_tokenizer = tokenizer.Tokenizer(sample_corpus)
    vocab_size = test_tokenizer.vocab_size
    config_params['vocab_size'] = vocab_size # Update config if needed by other parts

    print(f"Vocab size: {vocab_size}")

    # Pass dropout_rate to Decoder
    decoder_model = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, epsilon, dropout_rate)

    # Test forward pass (batched)
    print("\nTesting forward pass (batched input)...")
    batch_size = 2
    seq_len_test = 5
    dummy_token_ids_batch = np.random.randint(0, vocab_size, (batch_size, seq_len_test))
    causal_mask_batch = Decoder.create_causal_mask(seq_len_test)
    
    logits_batch = decoder_model.forward(dummy_token_ids_batch, mask=causal_mask_batch)
    print(f"Input token IDs shape: {dummy_token_ids_batch.shape}")
    print(f"Output logits shape: {logits_batch.shape}")
    assert logits_batch.shape == (batch_size, seq_len_test, vocab_size), f"Batched output shape mismatch: {logits_batch.shape}"

    # Test forward pass (single sequence)
    print("\nTesting forward pass (single sequence input)...")
    dummy_token_ids_single = np.random.randint(0, vocab_size, (seq_len_test,))
    # causal_mask_single = Decoder.create_causal_mask(seq_len_test) # Mask is same shape
    logits_single = decoder_model.forward(dummy_token_ids_single, mask=causal_mask_batch) # Can reuse batch mask
    print(f"Input token IDs shape: {dummy_token_ids_single.shape}")
    print(f"Output logits shape: {logits_single.shape}")
    assert logits_single.shape == (seq_len_test, vocab_size), f"Single sequence output shape mismatch: {logits_single.shape}"

    print("\nTesting attention weights storage...")
    # decoder_model.forward was called above, attention_weights should be populated
    num_attention_layers = len(decoder_model.attention_weights)
    print(f"Number of attention weight sets stored: {num_attention_layers} (expected {n_layers})")
    assert num_attention_layers == n_layers, "Mismatch in number of stored attention weight sets."
    if n_layers > 0 and decoder_model.attention_weights[0] is not None and decoder_model.attention_weights[0].size > 0 :
        expected_attn_shape_part = (batch_size, n_heads, seq_len_test, seq_len_test)
        print(f"Shape of first layer attention weights: {decoder_model.attention_weights[0].shape}")
        assert decoder_model.attention_weights[0].shape == expected_attn_shape_part, "Attention weights shape mismatch."
    elif n_layers == 0:
        print("No layers, so no attention weights to check.")
    else:
        print("First layer attention weights are None or empty, or not shaped as expected, skipping detailed shape check.")


    print("\nTesting forward pass with KV Cache (conceptual)...")
    current_token_id_kv = np.array([dummy_token_ids_batch[0, -1]]) 
    current_idx_kv = seq_len_test - 1
    kv_cache_dummy_list = [{} for _ in range(n_layers)]
    
    current_token_id_kv_batch = np.expand_dims(current_token_id_kv, axis=0)
    
    logits_kv = decoder_model.forward(
        current_token_id_kv_batch, 
        mask=None, 
        kv_cache_list=kv_cache_dummy_list, 
        current_token_idx=current_idx_kv
    )
    print(f"Input token ID (KV) shape: {current_token_id_kv_batch.shape}")
    print(f"Output logits (KV) shape: {logits_kv.shape}")
    assert logits_kv.shape == (1, 1, vocab_size), f"KV cache output shape mismatch: {logits_kv.shape}"
    if n_layers > 0:
        # In MHA, cache is stored under default key 'self_attention' if not named
        # Let's assume the DecoderBlock's MHA is named self_mha and it stores cache like self_mha.kv_cache
        # The kv_cache_list passed to Decoder.forward contains dicts, one per layer.
        # Each dict is passed to DecoderBlock.forward as layer_kv_cache.
        # DecoderBlock passes it to its MHA. MHA updates it.
        first_layer_cache = kv_cache_dummy_list[0]
        # The structure of first_layer_cache depends on MHA's implementation
        # Assuming MHA stores K,V directly as keys e.g. first_layer_cache = {'k': K, 'v': V}
        if 'k' in first_layer_cache and 'v' in first_layer_cache:
             print("KV cache appears to be populated by MHA in first block.")
        else:
             print("KV cache does not have 'k', 'v' keys in first block. MHA internal structure might differ.")
             print(f"Cache content: {first_layer_cache}")
        # A more robust check would inspect the shapes of K and V in the cache.


    print("\nDecoder standalone validation completed.")

    # Basic Test for Linear layer backward pass
    print("\nTesting Linear Layer backward pass (basic shapes)...")
    linear_test = Linear(d_input=10, d_output=5)
    dummy_input_linear = np.random.rand(3, 10) 
    output_linear = linear_test.forward(dummy_input_linear)
    dummy_grad_output_linear = np.random.rand(3, 5) 
    grad_input_linear = linear_test.backward(dummy_grad_output_linear)
    print(f"Linear input grad shape: {grad_input_linear.shape} (expected {(3,10)})")
    assert grad_input_linear.shape == (3,10)
    lin_grads = linear_test.get_gradients()
    print(f"Linear W grad shape: {lin_grads['W'].shape} (expected {(10,5)})")
    assert lin_grads['W'].shape == (10,5)
    print(f"Linear b grad shape: {lin_grads['b'].shape} (expected {(1,5)})")
    assert lin_grads['b'].shape == (1,5)
    print("Linear layer backward pass shapes seem okay.")

    # Conceptual test for Decoder.backward()
    print("\nTesting Decoder.backward() (conceptual shape checks)...")
    dummy_d_logits = np.random.rand(*logits_batch.shape) 
    decoder_model.backward(dummy_d_logits) 
    model_grads = decoder_model.get_gradients()

    # Check a few key gradients shapes
    if hasattr(decoder_model.input_embedding, 'W'):
        expected_emb_grad_shape = decoder_model.input_embedding.W.shape
        if 'decoder_input_embedding_W' in model_grads:
            actual_emb_grad_shape = model_grads['decoder_input_embedding_W'].shape
            print(f"Input Embedding W grad shape: {actual_emb_grad_shape} (expected {expected_emb_grad_shape})")
            assert actual_emb_grad_shape == expected_emb_grad_shape
        else:
            print("Input Embedding W gradient not found.")

    if n_layers > 0 and hasattr(decoder_model.decoder_blocks[0].self_mha, 'W_q'):
        expected_block0_mha_Wq_shape = decoder_model.decoder_blocks[0].self_mha.W_q.shape
        if f'decoder_block0_self_mha_W_q' in model_grads:
            actual_block0_mha_Wq_grad_shape = model_grads[f'decoder_block0_self_mha_W_q'].shape
            print(f"Block0 MHA W_q grad shape: {actual_block0_mha_Wq_grad_shape} (expected {expected_block0_mha_Wq_shape})")
            assert actual_block0_mha_Wq_grad_shape == expected_block0_mha_Wq_shape
        else:
            print(f"Block0 MHA W_q gradient not found (expected key: decoder_block0_self_mha_W_q).")
            print(f"Available grad keys: {list(model_grads.keys())}")


    if hasattr(decoder_model.final_linear_layer, 'W'):
        expected_final_linear_W_shape = decoder_model.final_linear_layer.W.shape
        if 'decoder_final_linear_W' in model_grads:
            actual_final_linear_W_grad_shape = model_grads['decoder_final_linear_W'].shape
            print(f"Final Linear W grad shape: {actual_final_linear_W_grad_shape} (expected {expected_final_linear_W_shape})")
            assert actual_final_linear_W_grad_shape == expected_final_linear_W_shape
        else:
            print("Final Linear W gradient not found.")
            
    print("Decoder.backward() and get_gradients() shape checks completed.")
    print("\nFull Decoder module validation completed.")