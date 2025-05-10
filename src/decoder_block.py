"""Transformer Decoder Block module.

This module defines the `DecoderBlock` class, which represents a single layer
of the Transformer decoder stack. Each block typically contains a masked
multi-head self-attention mechanism, followed by an add-and-normalize step,
and then a position-wise feed-forward network, also followed by an
add-and-normalize step.
"""
import numpy as np
# import sys # Not needed for this change
# import os # Not needed for this change

# from .config import get_config # Changed
# from .multi_head_attention import MultiHeadAttention, create_causal_mask # Changed
# from .feed_forward_network import FeedForwardNetwork # Changed
# from .layer_normalization import LayerNormalization # Changed

import config # New
import multi_head_attention # New
import feed_forward_network # New
import layer_normalization # New

class DecoderBlock:
    """Implements a single block of the Transformer Decoder.

    A DecoderBlock consists of two main sub-layers:
    1. Masked Multi-Head Self-Attention, followed by Layer Normalization and a
       residual connection.
    2. Position-wise Feed-Forward Network, followed by Layer Normalization and a
       residual connection.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output embeddings/hidden states.
    n_heads : int
        The number of attention heads for the multi-head attention layer.
    d_ff : int
        The dimensionality of the inner layer of the Feed-Forward Network.
    epsilon : float, optional
        A small value for numerical stability in Layer Normalization, by default 1e-5.
    dropout_rate : float, optional
        Dropout rate for the MultiHeadAttention and FeedForwardNetwork layers, by default 0.1.

    Attributes
    ----------
    d_model, n_heads, d_ff, epsilon, dropout_rate : 
        Stored initialization parameters.
    self_mha : MultiHeadAttention
        The masked multi-head self-attention layer.
    norm1 : LayerNormalization
        The first layer normalization layer (after self-attention).
    ffn : FeedForwardNetwork
        The position-wise feed-forward network.
    norm2 : LayerNormalization
        The second layer normalization layer (after FFN).
    attention_weights : np.ndarray or None
        Attention weights from the self-attention mechanism from the last
        forward pass. Shape `(batch_size, n_heads, seq_len, seq_len)`.
    x_cache, self_mha_output_cache, norm1_output_cache, ffn_output_cache : np.ndarray or None
        Caches for intermediate activations from the forward pass, used for backpropagation.
    training_mode : bool
        Flag indicating whether the block is in training mode.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, epsilon: float = 1e-5, dropout_rate: float = 0.1):
        """Initializes the DecoderBlock.

        Creates instances of MultiHeadAttention, FeedForwardNetwork, and
        LayerNormalization sub-layers.

        Parameters
        ----------
        d_model : int
            Dimensionality of input/output.
        n_heads : int
            Number of attention heads.
        d_ff : int
            Dimensionality of the FFN inner layer.
        epsilon : float, optional
            Epsilon for Layer Normalization, by default 1e-5.
        dropout_rate : float, optional
            Dropout rate for the MultiHeadAttention and FeedForwardNetwork layers, by default 0.1.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.training_mode = False

        self.self_mha = multi_head_attention.MultiHeadAttention(d_model, n_heads, dropout_rate=dropout_rate)
        self.norm1 = layer_normalization.LayerNormalization(d_model, epsilon)
        self.ffn = feed_forward_network.FeedForwardNetwork(d_model, d_ff, dropout_rate=dropout_rate)
        self.norm2 = layer_normalization.LayerNormalization(d_model, epsilon)
        
        self.attention_weights = None
        self.x_cache = None
        self.self_mha_output_cache = None
        self.norm1_output_cache = None
        self.ffn_output_cache = None

    def forward(self, x: np.ndarray, mask: np.ndarray, layer_kv_cache: dict = None, training_mode: bool = False) -> np.ndarray:
        """Performs the forward pass through the DecoderBlock.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape `(batch_size, seq_len, d_model)`.
        mask : np.ndarray
            Mask to be applied in the self-attention layer. Typically a causal mask.
            Shape should be broadcastable to `(batch_size, n_heads, seq_len, seq_len)`.
            If `layer_kv_cache` is active for single token generation, this mask may be
            internally overridden to `None` for the MHA call.
        layer_kv_cache : dict, optional
            A dictionary for KV caching for this block's MHA layer. If provided and
            `x` represents a single token, MHA uses this cache. Defaults to None.
        training_mode : bool, optional
            Flag indicating whether the block is in training mode.

        Returns
        -------
        np.ndarray
            Output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        self.training_mode = training_mode
        self.x_cache = x

        mha_mask_to_use = mask
        # If KV cache is active for single token generation (x.shape[1] == 1), MHA handles masking internally with None.
        if layer_kv_cache is not None and x.shape[-2] == 1: # Check seq_len dim
            mha_mask_to_use = None 
        
        mha_out, self.attention_weights = self.self_mha.forward(
            q_input=x, k_input=x, v_input=x, 
            mask=mha_mask_to_use, 
            kv_cache=layer_kv_cache,
            training_mode=training_mode
        )
        self.self_mha_output_cache = mha_out

        add_norm1_input = x + mha_out
        norm1_out = self.norm1.forward(add_norm1_input)
        self.norm1_output_cache = norm1_out
        
        ffn_out = self.ffn.forward(norm1_out, training_mode=training_mode)
        self.ffn_output_cache = ffn_out
        
        add_norm2_input = norm1_out + ffn_out
        norm2_out = self.norm2.forward(add_norm2_input)
        
        return norm2_out

    def get_parameters(self, prefix: str = "") -> dict:
        """Retrieves all learnable parameters from this block and its sub-layers.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to parameter names, by default "".

        Returns
        -------
        dict
            A dictionary mapping hierarchical parameter names to their NumPy array values.
        """
        params = {}
        params.update(self.self_mha.get_parameters(prefix=f"{prefix}self_mha_"))
        params.update(self.norm1.get_parameters(prefix=f"{prefix}norm1_"))
        params.update(self.ffn.get_parameters(prefix=f"{prefix}ffn_"))
        params.update(self.norm2.get_parameters(prefix=f"{prefix}norm2_"))
        return params

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the DecoderBlock.

        Backpropagates the gradient `d_output` through the FFN, LayerNorms,
        and Self-Attention layers, including residual connections.

        Parameters
        ----------
        d_output : np.ndarray
            Gradient of the loss with respect to the output of this DecoderBlock.
            Shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input `x` of this DecoderBlock.
            Shape `(batch_size, seq_len, d_model)`.
        """
        # Backward through Add & Norm (2)
        d_add_norm2_input = self.norm2.backward(d_output)
        d_norm1_out_residual2 = d_add_norm2_input
        d_ffn_out = d_add_norm2_input

        # Backward through Feed-Forward Network
        d_norm1_out_from_ffn = self.ffn.backward(d_ffn_out)
        d_norm1_out = d_norm1_out_from_ffn + d_norm1_out_residual2

        # Backward through Add & Norm (1)
        d_add_norm1_input = self.norm1.backward(d_norm1_out)
        d_x_residual1 = d_add_norm1_input
        d_mha_out = d_add_norm1_input

        # Backward through Masked Multi-Head Self-Attention
        # self_mha.backward returns (d_q_input, d_k_input, d_v_input)
        d_q_input, d_k_input, d_v_input = self.self_mha.backward(d_mha_out)
        # Since q_input=k_input=v_input=x for self-attention in decoder block:
        d_x_from_mha = d_q_input + d_k_input + d_v_input

        # Total gradient for input x
        d_x = d_x_from_mha + d_x_residual1
        
        return d_x

    def get_gradients(self, prefix: str = "") -> dict:
        """Retrieves all computed gradients from this block and its sub-layers.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to gradient names, by default "".

        Returns
        -------
        dict
            A dictionary mapping hierarchical gradient names to their NumPy array values.
        """
        grads = {}
        grads.update(self.self_mha.get_gradients(prefix=f"{prefix}self_mha_"))
        grads.update(self.norm1.get_gradients(prefix=f"{prefix}norm1_"))
        grads.update(self.ffn.get_gradients(prefix=f"{prefix}ffn_"))
        grads.update(self.norm2.get_gradients(prefix=f"{prefix}norm2_"))
        return grads

if __name__ == "__main__":
    print("Running validation tests for DecoderBlock...")

    # Get configuration
    cfg = get_config()
    d_model_cfg = cfg['d_model']
    n_heads_cfg = cfg['n_heads']
    d_ff_cfg = cfg['d_ff']
    epsilon_cfg = cfg['epsilon']
    max_seq_len_cfg = cfg['max_seq_len'] # For mask creation and input data

    # 1. Test Initialization
    print("\n1. Test Initialization...")
    try:
        decoder_block = DecoderBlock(d_model=d_model_cfg, n_heads=n_heads_cfg, d_ff=d_ff_cfg, epsilon=epsilon_cfg)
        print(f"  Successfully initialized DecoderBlock with d_model={d_model_cfg}, n_heads={n_heads_cfg}, d_ff={d_ff_cfg}")
        assert isinstance(decoder_block.self_mha, MultiHeadAttention)
        assert isinstance(decoder_block.norm1, LayerNormalization)
        assert isinstance(decoder_block.ffn, FeedForwardNetwork)
        assert isinstance(decoder_block.norm2, LayerNormalization)
        print("  Initialization test PASSED.")
    except Exception as e:
        print(f"  Initialization test FAILED: {e}")
        # Stop further tests if initialization fails
        exit()

    # 2. Test Forward Pass with Batched Input
    print("\n2. Test Forward Pass (Batched Input)...")
    batch_size_test = 2
    seq_len_test = 10 # Should be <= max_seq_len_cfg
    
    # Ensure test sequence length is valid
    if seq_len_test > max_seq_len_cfg:
        print(f"  Warning: seq_len_test ({seq_len_test}) > max_seq_len_cfg ({max_seq_len_cfg}). Adjusting seq_len_test.")
        seq_len_test = max_seq_len_cfg

    input_data_batched = np.random.rand(batch_size_test, seq_len_test, d_model_cfg)
    causal_mask_batched = create_causal_mask(seq_len_test) # Shape (1, 1, seq_len, seq_len)

    try:
        output_batched = decoder_block.forward(input_data_batched, causal_mask_batched)
        print(f"  Input shape: {input_data_batched.shape}, Output shape: {output_batched.shape}")
        assert output_batched.shape == input_data_batched.shape, "Output shape mismatch for batched input."
        
        # Check that attention weights were stored and have a plausible shape
        # Expected shape for attention_weights: (batch_size, n_heads, seq_len, seq_len)
        assert hasattr(decoder_block, 'attention_weights'), "Attention weights not stored."
        expected_attn_shape = (batch_size_test, n_heads_cfg, seq_len_test, seq_len_test)
        assert decoder_block.attention_weights.shape == expected_attn_shape, \
            f"Attention weights shape mismatch. Expected {expected_attn_shape}, got {decoder_block.attention_weights.shape}"
        print(f"  Attention weights stored with shape: {decoder_block.attention_weights.shape}")
        print("  Forward pass (batched) PASSED.")
    except Exception as e:
        print(f"  Forward pass (batched) FAILED: {e}")


    # 3. Test Forward Pass with Single Sequence Input
    print("\n3. Test Forward Pass (Single Sequence Input)...")
    # Use a different sequence length for variety, still <= max_seq_len_cfg
    seq_len_single_test = 5
    if seq_len_single_test > max_seq_len_cfg:
        seq_len_single_test = max_seq_len_cfg
        
    input_data_single = np.random.rand(seq_len_single_test, d_model_cfg)
    # The MHA class handles reshaping single sequence input if needed, 
    # but the mask needs to be appropriate.
    # Our current MHA expects batch_size dimension for Q,K,V splitting.
    # Let's reshape input_data_single to (1, seq_len_single_test, d_model_cfg)
    # and adjust mask creation if needed.
    # However, create_causal_mask is independent of batch size for the mask itself.
    causal_mask_single = create_causal_mask(seq_len_single_test)

    # For the forward pass, the MultiHeadAttention expects a batch dimension.
    # We can either modify MHA or reshape here. Let's reshape input for the test.
    input_data_single_batched_for_test = np.expand_dims(input_data_single, axis=0) # (1, seq_len, d_model)
    
    try:
        # Pass the 'batched' single sequence
        output_single_processed = decoder_block.forward(input_data_single_batched_for_test, causal_mask_single)
        print(f"  Input shape (original single): {input_data_single.shape}")
        print(f"  Input shape (passed to forward): {input_data_single_batched_for_test.shape}")
        print(f"  Output shape (from forward): {output_single_processed.shape}")
        
        # Output should correspond to the 'batched' single sequence, so (1, seq_len, d_model)
        assert output_single_processed.shape == input_data_single_batched_for_test.shape, "Output shape mismatch for single sequence (processed as batch)."
        
        # Optionally, squeeze the batch dim if you want to compare with original single input shape
        output_single_squeezed = np.squeeze(output_single_processed, axis=0)
        assert output_single_squeezed.shape == input_data_single.shape, "Output shape (squeezed) does not match original single input shape."
        
        # Check attention weights for the single (batched) case
        # Expected shape for attention_weights: (1, n_heads, seq_len_single_test, seq_len_single_test)
        expected_attn_shape_single = (1, n_heads_cfg, seq_len_single_test, seq_len_single_test)
        assert decoder_block.attention_weights.shape == expected_attn_shape_single, \
            f"Attention weights shape mismatch for single. Expected {expected_attn_shape_single}, got {decoder_block.attention_weights.shape}"
        print(f"  Attention weights stored with shape: {decoder_block.attention_weights.shape}")
        print("  Forward pass (single sequence) PASSED.")
    except Exception as e:
        print(f"  Forward pass (single sequence) FAILED: {e}")

    # 4. Test Error Handling for mismatched d_model in input
    print("\n4. Test Error Handling (mismatched d_model)...")
    wrong_d_model = d_model_cfg + 1
    input_data_wrong_dim = np.random.rand(batch_size_test, seq_len_test, wrong_d_model)
    # Use the same causal_mask_batched as it doesn't depend on d_model
    # For KV cache test, create a dummy cache structure (not strictly necessary for this error test)
    dummy_kv_cache_for_error_test = {'k': None, 'v': None}
    try:
        decoder_block.forward(input_data_wrong_dim, causal_mask_batched, layer_kv_cache=dummy_kv_cache_for_error_test)
        print("  Error handling test FAILED: ValueError not raised for mismatched d_model.")
    except ValueError as ve:
        print(f"  Successfully caught ValueError: {ve}")
        print("  Error handling test PASSED.")
    except Exception as e:
        print(f"  Error handling test FAILED with unexpected error: {e}")
        
    # 5. Test Causal Masking Effect (Qualitative Check)
    print("\n5. Test Causal Masking Effect (Qualitative)...")
    # We'll use a small sequence and check attention weights.
    # For a causal mask, attention_weights[b, h, i, j] should be 0 if j > i.
    seq_len_mask_test = 3
    input_mask_test = np.random.rand(1, seq_len_mask_test, d_model_cfg)
    causal_mask_test = create_causal_mask(seq_len_mask_test)
    
    try:
        _ = decoder_block.forward(input_mask_test, causal_mask_test)
        attn_weights = decoder_block.attention_weights # Shape (1, n_heads, seq_len_mask_test, seq_len_mask_test)
        
        # Check upper triangle (excluding diagonal) for zeros
        # For each head, for each sequence item i, attn_weights[0, head_idx, i, j] for j > i should be zero.
        masked_correctly = True
        for h in range(attn_weights.shape[1]): # Iterate over heads
            for i in range(seq_len_mask_test): # Iterate over query positions
                for j in range(i + 1, seq_len_mask_test): # Iterate over key positions (j > i)
                    if not np.isclose(attn_weights[0, h, i, j], 0.0):
                        masked_correctly = False
                        print(f"  Causality FAILED: Attn[{0},{h},{i},{j}] = {attn_weights[0,h,i,j]} (should be 0)")
                        break
                if not masked_correctly: break
            if not masked_correctly: break
            
        if masked_correctly:
            print("  Causal masking appears to be effective (upper triangle of attention is zero).")
            print("  Causal masking test PASSED.")
        else:
            print("  Causal masking test FAILED.")
            
    except Exception as e:
        print(f"  Causal masking test FAILED with error: {e}")

    # Test DecoderBlock initialization and forward pass
    print("\nTesting DecoderBlock Initialization and Forward Pass...")
    try:
        cfg = get_config()
        decoder_block = DecoderBlock(
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            d_ff=cfg['d_ff'],
            epsilon=cfg['epsilon']
        )
        print("DecoderBlock initialized successfully.")

        test_results = {}

        # Test 1: Batched Input Forward Pass
        print("  1. Testing batched input forward pass...")
        batch_size, seq_len = 2, 10
        x_batch = np.random.rand(batch_size, seq_len, cfg['d_model'])
        mask_batch = create_causal_mask(seq_len)
        try:
            output_batch = decoder_block.forward(x_batch, mask_batch)
            assert output_batch.shape == x_batch.shape
            print("    Batched forward pass: PASSED")
            test_results["Batched Forward"] = "PASSED"
        except Exception as e:
            print(f"    Batched forward pass: FAILED - {e}")
            test_results["Batched Forward"] = "FAILED"

        # Test 2: Single Input Forward Pass (reshaped to batch_size=1)
        print("  2. Testing single input forward pass...")
        seq_len_single = 5
        x_single = np.random.rand(1, seq_len_single, cfg['d_model'])
        mask_single = create_causal_mask(seq_len_single)
        try:
            output_single = decoder_block.forward(x_single, mask_single)
            assert output_single.shape == x_single.shape
            print("    Single forward pass: PASSED")
            test_results["Single Forward"] = "PASSED"
        except Exception as e:
            print(f"    Single forward pass: FAILED - {e}")
            test_results["Single Forward"] = "FAILED"

        # Test 3: KV Cache Forward Pass (Token 1 and Token 2)
        print("  3. Testing KV cache forward pass...")
        seq_len_kv = 1
        x_kv1 = np.random.rand(batch_size, seq_len_kv, cfg['d_model'])
        x_kv2 = np.random.rand(batch_size, seq_len_kv, cfg['d_model'])
        kv_cache = {'k': None, 'v': None}
        try:
            out_kv1 = decoder_block.forward(x_kv1, mask=None, layer_kv_cache=kv_cache)
            assert out_kv1.shape == x_kv1.shape
            assert kv_cache['k'] is not None and kv_cache['v'] is not None
            assert kv_cache['k'].shape == (batch_size, cfg['n_heads'], seq_len_kv, cfg['d_model'] // cfg['n_heads'])
            print("    KV cache token 1: PASSED")
            test_results["KV Cache T1"] = "PASSED"

            out_kv2 = decoder_block.forward(x_kv2, mask=None, layer_kv_cache=kv_cache)
            assert out_kv2.shape == x_kv2.shape
            assert kv_cache['k'].shape == (batch_size, cfg['n_heads'], seq_len_kv * 2, cfg['d_model'] // cfg['n_heads'])
            print("    KV cache token 2: PASSED")
            test_results["KV Cache T2"] = "PASSED"
        except Exception as e:
            print(f"    KV cache: FAILED - {e}")
            test_results["KV Cache T1"] = "FAILED"
            test_results["KV Cache T2"] = "FAILED"

        # Test 4: Causal Masking Effect
        print("  4. Testing causal masking effect...")
        # Using a small sequence to check attention weights
        seq_len_causal = 3
        x_causal = np.random.rand(1, seq_len_causal, cfg['d_model'])
        mask_causal = create_causal_mask(seq_len_causal)
        _ = decoder_block.forward(x_causal, mask_causal) # Populate attention_weights
        attn_weights = decoder_block.attention_weights
        # Expected: attn_weights[batch, head, query_pos, key_pos]
        # For query i, key j > i should have zero weight.
        causal_ok = True
        if attn_weights is not None and attn_weights.shape == (1, cfg['n_heads'], seq_len_causal, seq_len_causal):
            for i in range(seq_len_causal):
                for j in range(i + 1, seq_len_causal):
                    if not np.allclose(attn_weights[0, :, i, j], 0.0, atol=1e-6):
                        print(f"    Causality FAILED: Attn[0,:,{i},{j}] = {attn_weights[0, :, i, j]} (should be 0)")
                        causal_ok = False
                        break
                if not causal_ok: break
            if causal_ok:
                print("    Causal masking: PASSED")
                test_results["Causal Masking"] = "PASSED"
            else:
                test_results["Causal Masking"] = "FAILED"
        else:
            print("    Causal masking: SKIPPED (attn_weights not available or wrong shape)")
            test_results["Causal Masking"] = "SKIPPED"

    except Exception as e:
        print(f"Error during DecoderBlock tests: {e}")
        # test_results remains as is or set all to FAILED

    print("\nDecoderBlock Validation Summary:")
    for test_name, status in test_results.items():
        print(f"  {test_name}: {status}")

    print("\nAll DecoderBlock validation tests completed.") 