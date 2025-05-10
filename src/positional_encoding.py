"""Positional Encoding module for the Transformer model.

This module defines the `PositionalEncoding` class, which generates and applies
sinusoidal positional encodings to input embeddings. These encodings provide the
model with information about the relative or absolute position of tokens in a
sequence, which is crucial as the self-attention mechanism itself is
permutation-invariant.
"""
import numpy as np

class PositionalEncoding:
    """Generates and applies sinusoidal positional encodings.

    The positional encodings are pre-calculated up to `max_seq_len` and added
    to the input embeddings during the forward pass. Supports an `offset`
    parameter for use cases like KV caching during step-by-step generation.

    Parameters
    ----------
    max_seq_len : int
        The maximum sequence length for which positional encodings are pre-computed.
    d_model : int
        The dimensionality of the embeddings and positional encodings.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    d_model : int
        Dimensionality of the model.
    pe : np.ndarray
        The pre-computed positional encoding matrix of shape
        `(max_seq_len, d_model)`.
    """
    def __init__(self, max_seq_len: int, d_model: int):
        """Initializes the PositionalEncoding layer.

        Pre-computes the sinusoidal positional encoding matrix.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length.
        d_model : int
            Dimensionality of the model (embedding dimension).
        """
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = np.zeros((max_seq_len, d_model))

        position = np.arange(0, max_seq_len).reshape(max_seq_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        self.pe[:, 0::2] = np.sin(position * div_term)  # Apply to even indices
        self.pe[:, 1::2] = np.cos(position * div_term)  # Apply to odd indices

    def forward(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """Adds positional encoding to the input tensor.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape `(batch_size, seq_len, d_model)` or
            `(seq_len, d_model)` if batch_size is 1. This is typically
            the token embeddings.
        offset : int, optional
            The starting position offset for selecting the positional encoding slice.
            This is used during step-by-step generation with a KV cache to apply
            the correct positional encoding for the current token. Defaults to 0.

        Returns
        -------
        np.ndarray
            The input tensor `x` with positional encodings added. Shape remains
            the same as `x`.

        Raises
        ------
        ValueError
            If `offset + input_seq_len` exceeds `self.max_seq_len`.
        """
        # input_seq_len is the length of the sequence dimension in x
        # x can be (batch_size, seq_len, d_model) or (seq_len, d_model)
        input_seq_len = x.shape[-2]

        effective_len = offset + input_seq_len
        if effective_len > self.max_seq_len:
            raise ValueError(
                f"Effective sequence length (offset {offset} + input_seq_len {input_seq_len} = {effective_len}) "
                f"exceeds max_seq_len {self.max_seq_len}."
            )

        # Select the appropriate slice of positional encodings
        # self.pe has shape (max_seq_len, d_model)
        # We need a slice of shape (input_seq_len, d_model) starting from `offset`
        positional_encodings_slice = self.pe[offset : effective_len, :]

        # Ensure the slice can be broadcast if x is (batch, seq_len, d_model)
        # If x is (seq_len, d_model), direct addition works.
        # If x is (batch, seq_len, d_model), PE slice (seq_len, d_model) broadcasts.
        return x + positional_encodings_slice

if __name__ == "__main__":
    # Ensure src is in path
    import sys, os
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import config as cfg # New
    # import tokenizer as tkn # Not strictly needed for PE test, but for vocab size in embedding test
    import embedding as emb # New

    config_params = cfg.get_config()
    d_model = config_params['d_model']
    max_seq_len = config_params['max_seq_len']
    print(f"--- Using d_model: {d_model}, max_seq_len: {max_seq_len} ---")

    # --- Test 1: Initialization and PE Matrix Shape ---
    print("\n--- Test 1: Initialization and PE Matrix Shape ---")
    pe_layer = PositionalEncoding(max_seq_len, d_model)
    assert pe_layer.pe.shape == (max_seq_len, d_model), \
        f"PE matrix shape mismatch: Expected {(max_seq_len, d_model)}, Got {pe_layer.pe.shape}"
    print("PE matrix shape is correct.")
    print(f"PE matrix sample (first 5 rows, first 5 cols):\n{pe_layer.pe[:5, :5]}")

    # --- Test 2: Forward Pass Output Shape ---
    print("\n--- Test 2: Forward Pass Output Shape ---")
    batch_size = 2
    test_seq_len = 50
    dummy_input_batch = np.random.rand(batch_size, test_seq_len, d_model)
    output_batch = pe_layer.forward(dummy_input_batch)
    assert output_batch.shape == (batch_size, test_seq_len, d_model), \
        f"Output shape mismatch (batch): Expected {(batch_size, test_seq_len, d_model)}, Got {output_batch.shape}"
    print(f"Output shape for batched input is correct: {output_batch.shape}")

    dummy_input_single = np.random.rand(test_seq_len, d_model)
    output_single = pe_layer.forward(dummy_input_single)
    assert output_single.shape == (test_seq_len, d_model), \
        f"Output shape mismatch (single): Expected {(test_seq_len, d_model)}, Got {output_single.shape}"
    print(f"Output shape for single input is correct: {output_single.shape}")

    # --- Test 3: Correctness of PE Values (Spot Check) ---
    print("\n--- Test 3: Correctness of PE Values (Spot Check) ---")
    pos = 3
    i_dim = 4 # an even dimension
    expected_pe_sin = np.sin(pos / (10000 ** (i_dim / d_model)))
    assert np.isclose(pe_layer.pe[pos, i_dim], expected_pe_sin), \
        f"PE value mismatch at (pos={pos}, 2i={i_dim}). Expected {expected_pe_sin}, Got {pe_layer.pe[pos, i_dim]}"
    print(f"PE value for sin at (pos={pos}, 2i={i_dim}) is correct.")

    i_dim = 5 # an odd dimension
    expected_pe_cos = np.cos(pos / (10000 ** ((i_dim-1) / d_model)))
    assert np.isclose(pe_layer.pe[pos, i_dim], expected_pe_cos), \
        f"PE value mismatch at (pos={pos}, 2i+1={i_dim}). Expected {expected_pe_cos}, Got {pe_layer.pe[pos, i_dim]}"
    print(f"PE value for cos at (pos={pos}, 2i+1={i_dim}) is correct.")

    # --- Test 4: Integration with InputEmbedding ---
    print("\n--- Test 4: Integration with InputEmbedding ---")
    vocab_size = config_params.get('vocab_size', 50257)
    print(f"Using vocab_size for Embedding: {vocab_size}")

    embedding_layer = emb.InputEmbedding(vocab_size, d_model)
    
    # Create dummy token IDs within vocab range
    test_seq_len_integ = 5
    token_ids = np.random.randint(0, vocab_size, size=test_seq_len_integ) 
    print(f"Sample Token IDs: {token_ids}")
    
    input_tokens_np = np.array([token_ids]) # Batch size 1: Shape (1, 5)
    
    embeddings = embedding_layer.forward(input_tokens_np)
    assert embeddings.shape == (1, test_seq_len_integ, d_model), \
        f"Embeddings shape mismatch. Expected {(1, test_seq_len_integ, d_model)}, Got {embeddings.shape}"
    print(f"Embeddings shape: {embeddings.shape}")

    # Now pass the embeddings to the PE layer
    output_with_pe = pe_layer.forward(embeddings)
    assert output_with_pe.shape == embeddings.shape, \
        f"Output shape after PE mismatch. Expected {embeddings.shape}, Got {output_with_pe.shape}"
    print(f"Shape after adding PE: {output_with_pe.shape}")

    # Check that values have changed (i.e., PE was added)
    assert not np.allclose(output_with_pe, embeddings) or np.allclose(pe_layer.pe[:test_seq_len_integ, :], 0), \
        "Positional encoding did not change the embedding values (or PE is all zeros)."
    print("Integration with InputEmbedding successful: PE added to embeddings.")

    # --- Test 5: Sequence Length Exceeding max_seq_len ---
    print("\n--- Test 5: Sequence Length Exceeding max_seq_len ---")
    long_seq_len = max_seq_len + 10
    dummy_input_long = np.random.rand(long_seq_len, d_model)
    try:
        pe_layer.forward(dummy_input_long)
        print("FAIL: ValueError not raised for too long sequence.")
        assert False, "ValueError not raised for too long sequence."
    except ValueError as e:
        print(f"SUCCESS: Correctly raised ValueError for long sequence: {e}")
    
    # --- Test 6: Forward Pass with Offset (for KV Cache Scenario)
    print("\n--- Test 6: Forward Pass with Offset ---")
    # Test with batch_size = 1, seq_len = 1 (typical for generation step)
    test_offset_batch = 1
    test_offset_seq_len = 1
    test_offset_val = 5 # Arbitrary offset

    # Ensure test_offset_val + test_offset_seq_len <= max_seq_len
    if test_offset_val + test_offset_seq_len > max_seq_len:
        print(f"  Skipping offset test as offset ({test_offset_val}) + seq_len ({test_offset_seq_len}) > max_seq_len ({max_seq_len})")
    else:
        dummy_input_offset = np.random.rand(test_offset_batch, test_offset_seq_len, d_model)
        output_offset = pe_layer.forward(dummy_input_offset, offset=test_offset_val)
        assert output_offset.shape == (test_offset_batch, test_offset_seq_len, d_model), \
            f"Output shape mismatch (offset): Expected {(test_offset_batch, test_offset_seq_len, d_model)}, Got {output_offset.shape}"
        
        # Verify correct PE slice was added
        expected_pe_slice = pe_layer.pe[test_offset_val : test_offset_val + test_offset_seq_len, :]
        assert np.allclose(output_offset - dummy_input_offset, expected_pe_slice), \
            "PE slice added with offset does not match expected PE values."
        print(f"Output shape for offset input is correct: {output_offset.shape}")
        print(f"PE for offset={test_offset_val} correctly applied.")

        # Test offset leading to out of bounds
        too_large_offset = max_seq_len 
        try:
            pe_layer.forward(dummy_input_offset, offset=too_large_offset) # offset + input_seq_len = max_seq_len + 1
            print("  FAIL: ValueError not raised for offset + input_seq_len exceeding max_seq_len.")
            assert False, "ValueError not raised for offset + input_seq_len exceeding max_seq_len."
        except ValueError as e:
            print(f"  SUCCESS: Correctly raised ValueError for out-of-bounds offset: {e}")
        
        # Test offset with seq_len > 1
        test_offset_multiseq_len = 3
        dummy_input_offset_multiseq = np.random.rand(test_offset_batch, test_offset_multiseq_len, d_model)
        test_offset_multiseq_val = 2
        if test_offset_multiseq_val + test_offset_multiseq_len <= max_seq_len:
            output_offset_multiseq = pe_layer.forward(dummy_input_offset_multiseq, offset=test_offset_multiseq_val)
            expected_pe_slice_multiseq = pe_layer.pe[test_offset_multiseq_val : test_offset_multiseq_val + test_offset_multiseq_len, :]
            assert np.allclose(output_offset_multiseq - dummy_input_offset_multiseq, expected_pe_slice_multiseq), \
                "PE slice for multi-sequence with offset incorrect."
            print(f"  PE for offset={test_offset_multiseq_val} and seq_len={test_offset_multiseq_len} correctly applied.")
        else:
            print(f"  Skipping multi-sequence offset test as it would exceed max_seq_len.")

    print("\nAll PositionalEncoding tests passed!") 