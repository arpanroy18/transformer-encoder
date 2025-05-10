"""Input embedding layer for the Transformer model.

This module defines the `InputEmbedding` class, which is responsible for
converting sequences of discrete token IDs into continuous vector representations
(embeddings). It includes initialization of the embedding matrix and the forward
pass logic to look up embeddings for given token IDs.
It also includes basic backward pass logic for gradient computation with respect
to the embedding matrix.
"""
import numpy as np
# from .config import get_config # Changed
# from .tokenizer import Tokenizer # Changed

import config as cfg # New
import tokenizer as tkn # New

class InputEmbedding:
    """Converts token IDs to dense vector embeddings.

    This layer maintains an embedding matrix and looks up the corresponding
    vector for each input token ID.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary (number of unique tokens).
    d_model : int
        The dimensionality of the embedding vectors.

    Attributes
    ----------
    vocab_size : int
        Size of the vocabulary.
    d_model : int
        Dimensionality of the embedding vectors.
    embedding_matrix : np.ndarray
        The learnable embedding matrix of shape `(vocab_size, d_model)`,
        initialized using Glorot uniform initialization.
    input_token_ids_cache : np.ndarray or None
        Cache for input token IDs from the last forward pass, used in backward pass.
    grad_embedding_matrix : np.ndarray or None
        Gradient of the loss with respect to the embedding matrix.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """Initializes the InputEmbedding layer.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary (number of unique tokens).
        d_model : int
            The dimensionality of the embedding vectors.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding matrix using Glorot uniform initialization
        limit = np.sqrt(6. / (vocab_size + d_model))
        self.embedding_matrix = np.random.uniform(-limit, limit, (vocab_size, d_model))

        self.input_token_ids_cache = None
        self.grad_embedding_matrix = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the embedding layer.

        Looks up the embeddings for the given token IDs from the embedding matrix.

        Parameters
        ----------
        token_ids : np.ndarray
            A NumPy array of token IDs (integers). Can be 1D (single sequence)
            or 2D (batch of sequences, shape `(batch_size, sequence_length)`).

        Returns
        -------
        np.ndarray
            A NumPy array containing the embeddings for the input token IDs.
            Shape will be `(sequence_length, d_model)` if input is 1D, or
            `(batch_size, sequence_length, d_model)` if input is 2D.

        Raises
        -------
        TypeError
            If `token_ids` are not integers.
        ValueError
            If any `token_ids` are out of the vocabulary range [0, vocab_size-1].
        """
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise TypeError("Token IDs must be integers.")

        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            # Find specific out-of-bounds values for better error message
            min_val, max_val = np.min(token_ids), np.max(token_ids)
            raise ValueError(f"Token IDs out of bounds [0, {self.vocab_size -1}]. Min found: {min_val}, Max found: {max_val}")

        self.input_token_ids_cache = token_ids
        return self.embedding_matrix[token_ids]

    def get_parameters(self, prefix: str = "") -> dict:
        """Retrieves the learnable parameters of the layer.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to the parameter names, by default "".

        Returns
        -------
        dict
            A dictionary mapping parameter names to their NumPy array values.
            Contains `embedding_matrix`.
        """
        return {
            f"{prefix}embedding_matrix": self.embedding_matrix
        }

    def backward(self, d_output: np.ndarray) -> None:
        """Computes gradients for the InputEmbedding layer.

        Calculates the gradient of the loss with respect to the embedding matrix
        and stores it in `self.grad_embedding_matrix`.

        The gradient with respect to the input token IDs is not computed as they
        are discrete.

        Parameters
        ----------
        d_output : np.ndarray
            Gradient of the loss with respect to the output of this layer.
            Shape should match the output of the forward pass, e.g.,
            `(batch_size, seq_len, d_model)` or `(seq_len, d_model)`.

        Raises
        -------
        ValueError
            If the shape of `input_token_ids_cache` is not 1D or 2D.
        """
        if self.grad_embedding_matrix is None:
            self.grad_embedding_matrix = np.zeros_like(self.embedding_matrix)
        else:
            self.grad_embedding_matrix.fill(0) # Reset for current backward calculation

        original_shape = self.input_token_ids_cache.shape
        if self.input_token_ids_cache.ndim == 2: # (batch_size, seq_len)
            batch_size, seq_len = original_shape
            token_ids_flat = self.input_token_ids_cache.reshape(-1)
            d_output_flat = d_output.reshape(batch_size * seq_len, self.d_model)
            np.add.at(self.grad_embedding_matrix, token_ids_flat, d_output_flat)
        elif self.input_token_ids_cache.ndim == 1: # (seq_len,)
            np.add.at(self.grad_embedding_matrix, self.input_token_ids_cache, d_output)
        else:
            raise ValueError(f"Unsupported shape for token_ids: {original_shape}")

        return None

    def get_gradients(self, prefix: str = "") -> dict:
        """Retrieves the computed gradients for the layer's parameters.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to the gradient names, by default "".

        Returns
        -------
        dict
            A dictionary mapping gradient names (corresponding to parameter names)
            to their NumPy array values. Contains gradient for `embedding_matrix`.
            If `backward` has not been called, gradients are initialized to zeros.
        """
        if self.grad_embedding_matrix is None:
            self.grad_embedding_matrix = np.zeros_like(self.embedding_matrix)
        return {
            f"{prefix}embedding_matrix": self.grad_embedding_matrix
        }

if __name__ == "__main__":
    print("--- Validating InputEmbedding Layer (Integrated Test) ---")
    # Ensure src is in path if running this file directly
    import sys, os
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # cfg and tkn are already imported at the top level of the module

    print("Loading model configuration...")
    config_params = cfg.get_config() # Use aliased import
    d_model = config_params['d_model']
    print(f"Using d_model: {d_model} from config.py")

    print("\nInitializing tokenizer...")
    sample_corpus = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow."
    tokenizer_instance = tkn.Tokenizer(sample_corpus) # Use aliased import
    vocab_size = tokenizer_instance.vocab_size
    print(f"Using vocab_size: {vocab_size} from tokenizer (corpus: '{sample_corpus[:20]}...')")

    print(f"\nInitializing InputEmbedding with vocab_size={vocab_size}, d_model={d_model}")
    embedding_layer = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
    print("InputEmbedding layer initialized.")
    print(f"Embedding matrix shape: {embedding_layer.embedding_matrix.shape}")

    test_sentence = "quick fox"
    if not test_sentence:
        print("\nERROR: Test sentence for encoding is empty. Please provide a valid sentence.")
        exit()

    print(f"\nEncoding test sentence: '{test_sentence}'")
    sample_token_ids = tokenizer_instance.encode(test_sentence) # Use aliased import

    if not sample_token_ids:
        # This might happen if no characters in test_sentence are in the corpus
        print(f"Warning: Encoding of '{test_sentence}' resulted in empty token list.")
        print("This could be because no characters in the test sentence are part of the tokenizer's corpus.")
        print("Skipping forward pass and subsequent tests for this scenario.")
        # Create a dummy token_id to allow some tests to proceed if needed for structure, but highlight the issue.
        # However, for a real scenario, we'd likely error out or use a more representative sentence.
        # For now, we'll try a single valid token if possible, or just report.
        if vocab_size > 0:
            sample_token_ids = [0] # Use the first token in vocab if available
            print(f"Using a fallback token ID: {sample_token_ids} for minimal testing.")
        else:
            print("Vocabulary is empty. Cannot proceed with embedding tests.")
            exit()
            
    sample_token_ids = np.array(sample_token_ids)
    num_tokens_to_embed = len(sample_token_ids)
    print(f"Generated sample token IDs: {sample_token_ids} (Count: {num_tokens_to_embed})")


    # 5. Perform the forward pass
    print("\nPerforming forward pass...")
    try:
        embeddings = embedding_layer.forward(sample_token_ids)
        print("Forward pass successful.")
        print(f"Input token IDs shape: {sample_token_ids.shape}")
        print(f"Output embeddings shape: {embeddings.shape}")

        # 6. Validate output shape
        expected_shape = (num_tokens_to_embed, d_model)
        if embeddings.shape == expected_shape:
            print(f"Output shape validation PASSED. Expected {expected_shape}, Got {embeddings.shape}")
        else:
            print(f"Output shape validation FAILED. Expected {expected_shape}, Got {embeddings.shape}")
            exit() # Critical failure

        # 7. Validate content (basic check)
        manual_embeddings = embedding_layer.embedding_matrix[sample_token_ids]
        if np.array_equal(embeddings, manual_embeddings):
            print("Content validation PASSED. Embeddings match direct lookup.")
        else:
            print("Content validation FAILED. Embeddings do not match direct lookup.")
            exit() # Critical failure

    except ValueError as e:
        print(f"Error during forward pass: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()

    # 8. Test with out-of-bounds token ID
    print("\n--- Testing with out-of-bounds token ID ---")
    # Create an invalid token ID that is definitely out of bounds
    invalid_token_id_value = vocab_size 
    invalid_token_ids = np.array([sample_token_ids[0] if num_tokens_to_embed > 0 else 0, invalid_token_id_value])
    print(f"Testing with token IDs: {invalid_token_ids} (one is out of bounds: {invalid_token_id_value})")
    try:
        embedding_layer.forward(invalid_token_ids)
        print("FAILED: Expected ValueError for out-of-bounds token.")
    except ValueError as e:
        print(f"PASSED: Correctly caught ValueError: {e}")

    # 9. Test with non-integer token ID
    print("\n--- Testing with non-integer token ID ---")
    non_integer_token_ids = np.array([sample_token_ids[0] if num_tokens_to_embed > 0 else 0, 1.5, 2.0]) # Add a float
    # Ensure the integer parts are valid if they were actual tokens
    # For this test, we mainly care about the float causing an error.
    print(f"Testing with token IDs: {non_integer_token_ids} (contains non-integer)")
    try:
        embedding_layer.forward(non_integer_token_ids)
        print("FAILED: Expected ValueError for non-integer token.")
    except ValueError as e:
        print(f"PASSED: Correctly caught ValueError: {e}")
    except TypeError as e: # Numpy might raise TypeError for mixed types in some operations
        print(f"PASSED: Correctly caught TypeError (or similar) for mixed types: {e}")


    print("\n--- Validation Complete ---")

    # Test InputEmbedding initialization and forward pass
    print("\nTesting InputEmbedding Initialization and Forward Pass...")
    try:
        embedding_layer = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
        print(f"  Successfully initialized InputEmbedding with vocab_size={vocab_size}, d_model={d_model}")
        assert embedding_layer.embedding_matrix.shape == (vocab_size, d_model), "Embedding matrix shape mismatch"
        print("  Initialization test PASSED (parameter shapes).")

        # Test forward pass
        batch_size_test = 2
        seq_len_test = 5
        dummy_token_ids_batched = np.random.randint(0, vocab_size, size=(batch_size_test, seq_len_test))
        dummy_token_ids_single = np.random.randint(0, vocab_size, size=(seq_len_test,))

        output_batched = embedding_layer.forward(dummy_token_ids_batched)
        expected_shape_batched = (batch_size_test, seq_len_test, d_model)
        assert output_batched.shape == expected_shape_batched, f"Batched output shape mismatch. Expected {expected_shape_batched}, got {output_batched.shape}"
        print(f"  Forward pass (batched) PASSED. Output shape: {output_batched.shape}")

        output_single = embedding_layer.forward(dummy_token_ids_single)
        expected_shape_single = (seq_len_test, d_model)
        assert output_single.shape == expected_shape_single, f"Single sequence output shape mismatch. Expected {expected_shape_single}, got {output_single.shape}"
        print(f"  Forward pass (single sequence) PASSED. Output shape: {output_single.shape}")
        
        # Verify content of embeddings (matches direct lookup)
        sample_token_id = dummy_token_ids_batched[0, 0]
        assert np.array_equal(output_batched[0, 0, :], embedding_layer.embedding_matrix[sample_token_id, :]), \
            "Embedding content mismatch for batched input."
        sample_token_id_single = dummy_token_ids_single[0]
        assert np.array_equal(output_single[0, :], embedding_layer.embedding_matrix[sample_token_id_single, :]), \
            "Embedding content mismatch for single input."
        print("  Embedding content verification PASSED.")

        # Test error handling for out-of-bounds and non-integer token IDs
        try:
            invalid_ids_oor = np.array([[0, vocab_size]]) # vocab_size is out of bounds
            embedding_layer.forward(invalid_ids_oor)
            assert False, "Error for out-of-bounds token ID was not raised."
        except ValueError as e:
            print(f"  Error handling (out-of-bounds ID) PASSED. Received: {e}")
        
        try:
            invalid_ids_neg = np.array([[-1, 0]]) # -1 is out of bounds
            embedding_layer.forward(invalid_ids_neg)
            assert False, "Error for negative token ID was not raised."
        except ValueError as e:
            print(f"  Error handling (negative ID) PASSED. Received: {e}")

        try:
            invalid_ids_type = np.array([[0.5, 1.0]])
            embedding_layer.forward(invalid_ids_type)
            assert False, "Error for non-integer token ID was not raised."
        except TypeError as e:
            print(f"  Error handling (non-integer ID) PASSED. Received: {e}")

        # Test get_parameters
        print("\nTesting InputEmbedding get_parameters()...")
        params = embedding_layer.get_parameters("test_emb.")
        assert isinstance(params, dict), "get_parameters should return a dictionary."
        assert "test_emb.embedding_matrix" in params, "Missing embedding_matrix in parameters."
        assert params["test_emb.embedding_matrix"].shape == embedding_layer.embedding_matrix.shape
        print(f"InputEmbedding get_parameters() test passed. Collected: {list(params.keys())}")

        # Test backward pass & get_gradients
        print("\nTesting InputEmbedding backward() & get_gradients()...")
        dummy_d_output_batched = np.random.randn(*output_batched.shape).astype(np.float32)
        
        _ = embedding_layer.forward(dummy_token_ids_batched) # Call forward to set cache
        embedding_layer.backward(dummy_d_output_batched)
        
        grads = embedding_layer.get_gradients("test_emb.")
        assert isinstance(grads, dict), "get_gradients should return a dict."
        assert "test_emb.embedding_matrix" in grads, "Missing embedding_matrix in gradients."
        grad_matrix = grads["test_emb.embedding_matrix"]
        assert grad_matrix.shape == embedding_layer.embedding_matrix.shape, "Gradient matrix shape mismatch"
        print(f"InputEmbedding get_gradients collected. Grad matrix shape: {grad_matrix.shape} (Correct)")

        # Check if gradients are non-zero for used tokens and zero for unused (simple check)
        # Re-initialize grad_embedding_matrix for a clean test
        embedding_layer.grad_embedding_matrix = np.zeros_like(embedding_layer.embedding_matrix)
        test_ids = np.array([[0, 1], [0, 2]]) # token 0 used twice, 1 once, 2 once
        test_d_output = np.ones((2, 2, d_model)) # grad of 1 for all components
        _ = embedding_layer.forward(test_ids)
        embedding_layer.backward(test_d_output)
        grad_matrix_test = embedding_layer.get_gradients()["embedding_matrix"]
        
        assert np.all(grad_matrix_test[0] == 2*np.ones(d_model)), "Gradient for token 0 incorrect."
        assert np.all(grad_matrix_test[1] == 1*np.ones(d_model)), "Gradient for token 1 incorrect."
        assert np.all(grad_matrix_test[2] == 1*np.ones(d_model)), "Gradient for token 2 incorrect."
        if vocab_size > 3:
            assert np.all(grad_matrix_test[3] == 0), "Gradient for unused token 3 not zero."
        print("InputEmbedding backward pass accumulates gradients correctly (checked specific token grads)." )

    except Exception as e:
        print(f"Error during InputEmbedding tests: {e}")
        import traceback
        traceback.print_exc()
