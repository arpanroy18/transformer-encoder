"""Training loop and associated components for the Transformer model.

This module includes:
- `CrossEntropyLoss`: A class for calculating cross-entropy loss and its gradient.
- `AdamOptimizer`: A basic implementation of the Adam optimization algorithm.
- `prepare_batch`: A function to create input and target batches from a tokenized corpus.
- `train`: The main function to orchestrate the training process, including data batching,
  forward pass, loss calculation, backward pass, and parameter updates.
"""
import numpy as np
# import sys # Not needed for this change
# import os # Not needed for this change

# from .config import get_config # Changed
# from .tokenizer import Tokenizer # Changed
# from .decoder import Decoder # Changed

import config as cfg # New
import tokenizer as tkn # New
import decoder as dec # New

class CrossEntropyLoss:
    """Computes Cross-Entropy Loss and its gradient.

    Used for classification tasks, particularly for next-token prediction in
    language models.

    Attributes
    ----------
    probs : np.ndarray or None
        Cached softmax probabilities from the last forward pass.
    target : np.ndarray or None
        Cached target token IDs from the last forward pass.
    """
    def __init__(self):
        """Initializes the CrossEntropyLoss calculator."""
        self.probs = None
        self.target = None

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Computes softmax probabilities numerically stably along the last axis.

        Parameters
        ----------
        logits : np.ndarray
            Input array of logits.

        Returns
        -------
        np.ndarray
            Array of probabilities after applying softmax.
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def forward(self, logits: np.ndarray, target_token_ids: np.ndarray) -> float:
        """Computes the average cross-entropy loss.

        Parameters
        ----------
        logits : np.ndarray
            Output logits from the model, shape `(batch_size, seq_len, vocab_size)`.
        target_token_ids : np.ndarray
            Target token IDs, shape `(batch_size, seq_len)`.

        Returns
        -------
        float
            The average cross-entropy loss over all tokens in the batch.
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_reshaped = logits.reshape(batch_size * seq_len, vocab_size)
        target_reshaped = target_token_ids.reshape(batch_size * seq_len)

        self.probs = self._softmax(logits_reshaped)
        self.target = target_reshaped

        epsilon = 1e-9 # To prevent log(0)
        # Select probabilities of the target tokens
        correct_log_probs = np.log(self.probs[np.arange(len(target_reshaped)), target_reshaped] + epsilon)
        loss = -np.sum(correct_log_probs) / len(target_reshaped)
        return loss

    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to the input logits.

        The gradient is `(probabilities - one_hot_target) / N`, where N is the
        number of elements over which the loss was averaged.

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. logits, shape `(batch_size * seq_len, vocab_size)`.

        Raises
        ------
        ValueError
            If `forward()` has not been called before `backward()`.
        """
        if self.probs is None or self.target is None:
            raise ValueError("Forward pass must be called before backward pass.")

        grad_logits = self.probs.copy()
        grad_logits[np.arange(len(self.target)), self.target] -= 1
        grad_logits /= len(self.target) # Normalize by the number of elements
        return grad_logits

def prepare_batch(corpus_tokens: list[int], batch_size: int, seq_len: int, current_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepares a batch of input and target sequences from a tokenized corpus.

    Input sequences are `x_i, x_i+1, ..., x_i+seq_len-1`.
    Target sequences are `x_i+1, x_i+2, ..., x_i+seq_len`.
    Uses a sliding window approach with modulo arithmetic to cycle through the corpus.

    Parameters
    ----------
    corpus_tokens : list[int]
        The entire tokenized corpus as a list of integer token IDs.
    batch_size : int
        The number of sequences in a batch.
    seq_len : int
        The length of each input/target sequence.
    current_idx : int
        The starting index in `corpus_tokens` for creating the first sequence
        of the current batch. This index is used in a modulo fashion.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - inputs : np.ndarray of shape `(batch_size, seq_len)` containing input token IDs.
        - targets : np.ndarray of shape `(batch_size, seq_len)` containing target token IDs.

    Notes
    -----
    Requires `len(corpus_tokens)` to be at least `seq_len + 1` to form a valid pair.
    The `current_idx` allows cycling through the dataset across multiple calls.
    """
    inputs = []
    targets = []
    if len(corpus_tokens) < seq_len + 1:
        raise ValueError(f"Corpus too short ({len(corpus_tokens)} tokens) for sequence length ({seq_len}). Needs at least {seq_len + 1} tokens.")
    max_start_idx = len(corpus_tokens) - seq_len - 1
    if max_start_idx < 0: max_start_idx = 0
    for i in range(batch_size):
        start_idx = (current_idx + i * seq_len) % (max_start_idx + 1) if max_start_idx >=0 else 0
        end_idx = start_idx + seq_len
        input_seq = corpus_tokens[start_idx:end_idx]
        target_seq = corpus_tokens[start_idx+1 : end_idx+1]
        inputs.append(input_seq)
        targets.append(target_seq)
    return np.array(inputs), np.array(targets)

class AdamOptimizer:
    """Implements the Adam optimization algorithm.

    Adam is an adaptive learning rate optimization algorithm that computes
    individual learning rates for different parameters from estimates of
    first and second moments of the gradients.

    Parameters
    ----------
    parameters : dict
        A dictionary where keys are parameter names (str) and values are the
        NumPy arrays of the parameters to be optimized.
    learning_rate : float, optional
        The learning rate (step size), by default 0.001.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates, by default 0.9.
    beta2 : float, optional
        Exponential decay rate for the second-moment estimates, by default 0.999.
    epsilon : float, optional
        A small constant for numerical stability, by default 1e-8.

    Attributes
    ----------
    m : dict
        First moment vector (moving average of gradients) for each parameter.
    v : dict
        Second moment vector (moving average of squared gradients) for each parameter.
    t : int
        Timestep counter.
    """
    def __init__(self, parameters: dict, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Initializes the AdamOptimizer.

        Parameters are passed as a dictionary and are updated in-place by `update()`.
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {name: np.zeros_like(param) for name, param in parameters.items()}
        self.v = {name: np.zeros_like(param) for name, param in parameters.items()}
        self.t = 0

    def update(self, grads: dict) -> None:
        """Updates the parameters based on the computed gradients.

        Parameters
        ----------
        grads : dict
            A dictionary where keys are parameter names (must match those in
            `self.parameters`) and values are the gradient NumPy arrays.
        """
        self.t += 1
        for name, grad_value in grads.items():
            if name not in self.parameters:
                print(f"Warning: Gradient for '{name}' received but no corresponding parameter found in optimizer. Skipping update for this parameter.")
                continue
            if grad_value is None:
                print(f"Warning: Gradient for '{name}' is None. Skipping update for this parameter.")
                continue

            param_to_update = self.parameters[name]
            
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad_value
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad_value ** 2)
            
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update the parameter in-place
            param_to_update -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def train(model: dec.Decoder, tokenizer_instance: tkn.Tokenizer, corpus: str, 
          epochs: int, batch_size: int, learning_rate: float, 
          seq_len: int, print_every: int = 100):
    """Main training loop for the Transformer Decoder model.

    Handles data preparation, batching, forward/backward passes, loss calculation,
    and parameter updates using Adam optimizer.

    Parameters
    ----------
    model : Decoder
        The Transformer Decoder model instance to be trained.
    tokenizer : Tokenizer
        The tokenizer instance for processing the corpus.
    corpus : str
        The training corpus as a single string.
    epochs : int
        The number of training epochs.
    batch_size : int
        The number of sequences per batch.
    learning_rate : float
        The learning rate for the Adam optimizer.
    seq_len : int
        The length of token sequences to be used for training.
    print_every : int, optional
        Log training progress every `print_every` batches, by default 100.

    Raises
    ------
    ValueError
        If the corpus is too short for the given `seq_len` and `batch_size`.
    """
    print(f"Starting training with: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, seq_len={seq_len}")
    
    loss_fn = CrossEntropyLoss()
    model_parameters = model.get_parameters() # Get all learnable parameters from the model
    optimizer = AdamOptimizer(model_parameters, learning_rate=learning_rate)

    corpus_tokens = tokenizer_instance.encode(corpus)
    corpus_len = len(corpus_tokens)

    if corpus_len < seq_len + 1:
        raise ValueError(f"Corpus is too short ({corpus_len} tokens) for seq_len ({seq_len}). Needs at least {seq_len + 1} tokens.")

    # Calculate number of possible starting positions for sequences
    num_possible_starts = corpus_len - seq_len
    if num_possible_starts <= 0:
        raise ValueError(f"Corpus length ({corpus_len}) minus seq_len ({seq_len}) results in non-positive possible starts. Adjust seq_len or corpus.")

    # num_batches based on unique starting positions available, ensuring full coverage if possible
    # This defines how many unique batches can be drawn before cycling through start indices significantly.
    # For `prepare_batch` cycling logic, an epoch can be defined as a certain number of batch steps.
    num_batches_per_epoch = max(1, num_possible_starts // batch_size)
    # Or, simply iterate through all possible start indices for an epoch if data is small relative to batch size
    # For larger datasets, this might be too many batches. A common approach is num_tokens / (batch_size * seq_len)
    # Let's use a definition that ensures data is cycled through.
    # Number of batches to roughly go through the data once, considering sequence length and batch size.
    # This is approximate due to the sliding window of prepare_batch.
    num_batches_per_epoch = (corpus_len - seq_len) // (batch_size) # Number of full batches of starting positions
    if num_batches_per_epoch == 0: num_batches_per_epoch = 1

    print(f"  Corpus length: {corpus_len} tokens")
    print(f"  Vocab size: {tokenizer_instance.vocab_size}")
    print(f"  Effective number of batches per epoch: {num_batches_per_epoch}")

    current_batch_start_idx_in_corpus = 0

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        
        for batch_num in range(num_batches_per_epoch):
            # Prepare batch data
            # The current_batch_start_idx_in_corpus helps `prepare_batch` to cycle through the dataset
            input_batch, target_batch = prepare_batch(corpus_tokens, batch_size, seq_len, current_batch_start_idx_in_corpus)
            
            # Update the starting index for the next call to prepare_batch, ensuring it cycles through the corpus.
            # Each call to prepare_batch effectively processes `batch_size` sequences.
            # We want to advance by `batch_size` potential starting positions.
            current_batch_start_idx_in_corpus = (current_batch_start_idx_in_corpus + batch_size) % num_possible_starts

            # Create causal mask for the current batch
            # Mask shape (1, 1, seq_len, seq_len), then repeat for batch_size
            causal_mask_template = dec.Decoder.create_causal_mask(seq_len)
            # No need to repeat, MHA can broadcast (batch_size, n_heads, seq_len, seq_len) with (1,1,seq_len,seq_len)
            # batch_causal_mask = np.repeat(causal_mask_template, batch_size, axis=0) 

            # Forward pass
            # `kv_cache_list` is None during training, `current_token_idx` is 0 for full sequence processing.
            logits = model.forward(input_batch, mask=causal_mask_template, kv_cache_list=None, current_token_idx=0, training_mode=True)
            
            # Calculate loss
            loss = loss_fn.forward(logits, target_batch)
            epoch_total_loss += loss
            
            # Backward pass
            grad_logits_flat = loss_fn.backward() # Shape (batch_size * seq_len, vocab_size)
            # Reshape gradient to match logits output shape for model.backward()
            grad_logits = grad_logits_flat.reshape(logits.shape) 
            
            model.backward(grad_logits) # This stores gradients in each model sub-component
            
            # Update parameters using optimizer
            # The optimizer needs all gradients from the model.
            all_model_grads = model.get_gradients() # Collect all gradients
            optimizer.update(all_model_grads)

            if (batch_num + 1) % print_every == 0 or batch_num == num_batches_per_epoch - 1:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_num+1}/{num_batches_per_epoch}, Avg Batch Loss: {loss:.4f}")
        
        avg_epoch_loss = epoch_total_loss / num_batches_per_epoch
        print(f"Epoch {epoch+1} completed. Average Epoch Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    print("Starting Training Loop Module Validation...")
    import sys, os # Keep sys, os for path manipulation here
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Imports cfg, tkn, dec are already at the top of the module level

    config_params = cfg.get_config({
        'd_model': 32, 'n_layers': 1, 'n_heads': 1, 'd_ff': 64, 
        'max_seq_len': 20, 'vocab_size': 0, 'dropout_rate': 0.1 # Test with dropout
    })
    sample_corpus_text = "this is a simple test corpus. it repeats to provide enough data for testing the loop."
    sample_corpus_text = sample_corpus_text * 10 
    
    tokenizer_instance = tkn.Tokenizer(sample_corpus_text)
    config_params['vocab_size'] = tokenizer_instance.vocab_size

    decoder_model = dec.Decoder(
        vocab_size=config_params['vocab_size'], d_model=config_params['d_model'],
        n_layers=config_params['n_layers'], n_heads=config_params['n_heads'],
        d_ff=config_params['d_ff'], max_seq_len=config_params['max_seq_len'],
        dropout_rate=config_params['dropout_rate'] 
    )
    print(f"  Model initialized with vocab_size: {config_params['vocab_size']}, dropout: {config_params['dropout_rate']}")

    # ... (Rest of __main__ tests for CrossEntropyLoss, prepare_batch, AdamOptimizer remain as they were) ...
    print("\n  Testing CrossEntropyLoss...")
    loss_fn_test = CrossEntropyLoss()
    test_logits = np.array([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]]).astype(np.float32)
    test_targets = np.array([[2, 0]]).astype(np.int32)
    expected_loss_manual = 0.87315 
    calculated_loss = loss_fn_test.forward(test_logits, test_targets)
    print(f"    Calculated Loss: {calculated_loss:.7f}, Expected (approx): {expected_loss_manual:.7f}")
    assert np.isclose(calculated_loss, expected_loss_manual, atol=1e-4), "CrossEntropyLoss forward pass failed."
    grad_l = loss_fn_test.backward()
    assert grad_l.shape == (1*2, 3), "CrossEntropyLoss backward pass shape failed."
    print("    CrossEntropyLoss test PASSED.")

    print("\n  Testing prepare_batch...")
    test_corpus_tokens = tokenizer_instance.encode("abcdefghij") 
    pb_batch_size = 2
    pb_seq_len = 3
    pb_current_idx = 0
    inputs_pb, targets_pb = prepare_batch(test_corpus_tokens, pb_batch_size, pb_seq_len, pb_current_idx)
    assert inputs_pb.shape == (pb_batch_size, pb_seq_len), "prepare_batch input shape mismatch."
    assert targets_pb.shape == (pb_batch_size, pb_seq_len), "prepare_batch target shape mismatch."
    assert np.array_equal(inputs_pb[0], test_corpus_tokens[0:3]), "prepare_batch input content mismatch."
    assert np.array_equal(targets_pb[0], test_corpus_tokens[1:4]), "prepare_batch target content mismatch."
    print("    prepare_batch test PASSED.")

    print("\n  Testing AdamOptimizer...")
    dummy_params = {"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([0.1, 0.2])}
    dummy_grads = {"W": np.array([[0.1, 0.1], [0.1, 0.1]]), "b": np.array([0.01, 0.01])}
    optimizer_test = AdamOptimizer(dummy_params, learning_rate=0.1)
    initial_W = dummy_params["W"].copy()
    optimizer_test.update(dummy_grads)
    assert not np.array_equal(dummy_params["W"], initial_W), "AdamOptimizer did not update parameters."
    print("    AdamOptimizer test PASSED (parameters updated).")

    print("\n  Testing train() function (1 epoch, few batches)...")
    train_epochs = 1
    train_batch_size = 2 # Min batch size for prepare_batch with current logic
    train_seq_len = 5
    train_lr = 0.001
    train_print_every = 1

    try:
        train(decoder_model, tokenizer_instance, sample_corpus_text, 
              train_epochs, train_batch_size, train_lr, train_seq_len, train_print_every)
        print("    train() function executed without crashing - PASSED (qualitative).")
    except Exception as e:
        print(f"    train() function FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\nTraining Loop Module Validation Completed.") 