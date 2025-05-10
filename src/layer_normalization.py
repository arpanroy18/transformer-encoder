"""Layer Normalization module for the Transformer model.

This module defines the `LayerNormalization` class, which implements layer
normalization as described in the paper "Layer Normalization" by Ba, Kiros, Hinton.
It normalizes the inputs across the features for each data sample independently.
"""
import numpy as np
# import sys # Not needed for this change
# import os # Not needed for this change

# from . import config # Changed: This was an unusual relative import, make it absolute
import config as cfg # New

class LayerNormalization:
    """Implements Layer Normalization.

    Normalizes the activations of a layer across the feature dimension (last axis).
    It helps stabilize training by keeping the mean of activations close to 0 and
    standard deviation close to 1 for each sample independently.
    The formula is: `y = gamma * (x - mean(x)) / sqrt(variance(x) + epsilon) + beta`,
    where `gamma` and `beta` are learnable parameters.

    Parameters
    ----------
    d_model : int
        The dimension of the input features (last dimension of the input tensor).
    epsilon : float, optional
        A small value added to the variance for numerical stability, by default 1e-5.

    Attributes
    ----------
    d_model : int
        Dimensionality of the input features.
    epsilon : float
        Value for numerical stability.
    gamma : np.ndarray
        Learnable scale parameter of shape `(d_model,)`, initialized to ones.
    beta : np.ndarray
        Learnable shift parameter of shape `(d_model,)`, initialized to zeros.
    input_cache : np.ndarray or None
        Cache for input `x` from the last forward pass.
    mean_cache : np.ndarray or None
        Cache for the computed mean from the last forward pass.
    variance_cache : np.ndarray or None
        Cache for the computed variance from the last forward pass.
    normalized_input_cache : np.ndarray or None
        Cache for the input normalized (before scaling by gamma and shifting by beta).
    grad_gamma : np.ndarray or None
        Gradient of the loss with respect to `gamma`.
    grad_beta : np.ndarray or None
        Gradient of the loss with respect to `beta`.
    """
    def __init__(self, d_model: int, epsilon: float = 1e-5):
        """Initializes the LayerNormalization layer.

        Parameters
        ----------
        d_model : int
            The dimension of the input features (last dimension).
        epsilon : float, optional
            A small value for numerical stability, by default 1e-5.
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # Learnable parameters: scale (gamma) and shift (beta)
        # Initialized to 1 and 0 respectively, so initially the layer only normalizes.
        self.gamma = np.ones((d_model,))  # Scale parameter
        self.beta = np.zeros((d_model,))   # Shift parameter

        # Caches for backward pass
        self.input_cache = None
        self.mean_cache = None
        self.variance_cache = None
        self.normalized_input_cache = None # x_normalized before scaling by gamma

        # Gradients for parameters
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies Layer Normalization to the input tensor.

        Normalizes along the last dimension (features).

        Parameters
        ----------
        x : np.ndarray
            Input tensor. Expected shape `(..., d_model)`, where `...` indicates
            any number of leading dimensions (e.g., batch_size, seq_len).

        Returns
        -------
        np.ndarray
            The normalized tensor, with the same shape as the input `x`.

        Raises
        ------
        ValueError
            If the last dimension of `x` does not match `self.d_model`.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input features (last dimension) to be {self.d_model}, but got {x.shape[-1]}")

        self.input_cache = x
        # Calculate mean and variance along the last dimension (features)
        # keepdims=True ensures that mean and variance have the same number of dimensions
        # as x, allowing for broadcasting.
        self.mean_cache = np.mean(x, axis=-1, keepdims=True)
        self.variance_cache = np.var(x, axis=-1, keepdims=True)

        # Normalize the input
        # x_normalized = (x - mean) / sqrt(variance + epsilon)
        self.normalized_input_cache = (x - self.mean_cache) / np.sqrt(self.variance_cache + self.epsilon)

        # Apply scale (gamma) and shift (beta)
        # output = gamma * x_normalized + beta
        output = self.gamma * self.normalized_input_cache + self.beta

        return output

    def get_parameters(self, prefix: str = "") -> dict:
        """Retrieves the learnable parameters of the layer.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to parameter names, by default "".

        Returns
        -------
        dict
            A dictionary mapping parameter names (`gamma`, `beta`)
            to their NumPy array values.
        """
        return {
            f"{prefix}gamma": self.gamma,
            f"{prefix}beta": self.beta
        }

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Computes gradients for the LayerNormalization layer.

        Backpropagates the gradient `d_output` (dL/d(LN_output)) through
        the normalization formula. Stores gradients for `gamma` and `beta`.
        The derivation for dL/dx follows common implementations and lecture notes
        (e.g., Stanford CS231n, Andrej Karpathy's makemore).

        Parameters
        ----------
        d_output : np.ndarray
            Gradient of the loss with respect to the output of this LN layer.
            Shape must match the output of the forward pass (e.g., `(..., d_model)`).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input `x` of the LN layer.
            Shape matches the input `x` (e.g., `(..., d_model)`).
        """
        x = self.input_cache
        x_norm = self.normalized_input_cache # x_hat in some notations
        # mean = self.mean_cache # Not directly used in this d_x derivation but cached
        var = self.variance_cache # Variance for each sample, shape (..., 1)
        N = x.shape[-1] # d_model, number of features being normalized

        # Gradient w.r.t. learnable parameters gamma and beta
        # Sum over all dimensions except the feature dimension (last one)
        sum_axes = tuple(range(d_output.ndim - 1))
        self.grad_gamma = np.sum(d_output * x_norm, axis=sum_axes, keepdims=False)
        self.grad_beta = np.sum(d_output, axis=sum_axes, keepdims=False)

        # Gradient w.r.t. x_norm (normalized input before scale/shift)
        # dL/dx_norm = dL/dOutput * dOutput/dx_norm = d_output * gamma
        d_x_norm = d_output * self.gamma # Shape (..., d_model)

        # Standard deviation (used multiple times)
        std_inv = 1.0 / np.sqrt(var + self.epsilon) # Shape (..., 1)

        # Gradient w.r.t. input x (dL/dx)
        # This formula is derived from the full LayerNorm backward pass.
        # See, e.g., https://arxiv.org/pdf/1803.08494.pdf (Appendix A for Batch Norm, Layer Norm similar)
        # or notes like Karpathy's lectures on Backpropagation.
        # dL/dx_i = (1/std) * [ dL/dx_norm_i - mean(dL/dx_norm) - x_norm_i * mean(dL/dx_norm * x_norm) ]
        # where mean is over the feature dimension (last axis).
        
        # term1: dL/dx_norm_i (already computed as d_x_norm)
        # term2: mean(dL/dx_norm) across features
        mean_d_x_norm = np.mean(d_x_norm, axis=-1, keepdims=True) # Shape (..., 1)
        # term3_intermediate: dL/dx_norm * x_norm
        d_x_norm_times_x_norm = d_x_norm * x_norm # Shape (..., d_model)
        # term3: x_norm_i * mean(dL/dx_norm * x_norm) across features
        mean_d_x_norm_times_x_norm = np.mean(d_x_norm_times_x_norm, axis=-1, keepdims=True) # Shape (..., 1)
        term3_final = x_norm * mean_d_x_norm_times_x_norm # Shape (..., d_model)

        d_x = std_inv * (d_x_norm - mean_d_x_norm - term3_final)
        
        return d_x

    def get_gradients(self, prefix: str = "") -> dict:
        """Retrieves the computed gradients for the layer's parameters.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to gradient names, by default "".

        Returns
        -------
        dict
            A dictionary mapping gradient names (`gamma`, `beta`)
            to their NumPy array values. Initializes gradients to zeros if
            `backward` has not been called or resulted in None.
        """
        if self.grad_gamma is None: self.grad_gamma = np.zeros_like(self.gamma)
        if self.grad_beta is None: self.grad_beta = np.zeros_like(self.beta)
            
        return {
            f"{prefix}gamma": self.grad_gamma,
            f"{prefix}beta": self.grad_beta
        }

if __name__ == "__main__":
    print("Running validation tests for LayerNormalization...")
    # Ensure src is in path
    import sys, os
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # cfg is already imported at top of module

    config_params = cfg.get_config()
    d_model_cfg = config_params['d_model']
    epsilon_cfg = config_params['epsilon']

    # 1. Test Initialization
    print("\n1. Test Initialization...")
    try:
        ln_layer = LayerNormalization(d_model=d_model_cfg, epsilon=epsilon_cfg)
        print(f"  Successfully initialized LayerNormalization with d_model={d_model_cfg}, epsilon={epsilon_cfg}")
        assert ln_layer.gamma.shape == (d_model_cfg,), f"Gamma shape mismatch: {ln_layer.gamma.shape}"
        assert ln_layer.beta.shape == (d_model_cfg,), f"Beta shape mismatch: {ln_layer.beta.shape}"
        assert np.all(ln_layer.gamma == 1), "Gamma not initialized to ones."
        assert np.all(ln_layer.beta == 0), "Beta not initialized to zeros."
        print("  Initialization test PASSED.")
    except Exception as e:
        print(f"  Initialization test FAILED: {e}")

    # 2. Test Forward Pass with Batched Input
    print("\n2. Test Forward Pass (Batched Input)...")
    batch_size_test = 2
    seq_len_test = 10
    input_data_batched = np.random.rand(batch_size_test, seq_len_test, d_model_cfg) * 10 + 5 # Random data with some mean and std

    try:
        normalized_output_batched = ln_layer.forward(input_data_batched)
        print(f"  Input shape: {input_data_batched.shape}, Output shape: {normalized_output_batched.shape}")
        assert normalized_output_batched.shape == input_data_batched.shape, "Output shape mismatch for batched input."

        # Check mean and variance of the normalized output (per instance in batch and sequence)
        # Before gamma/beta, mean should be ~0 and std ~1.
        # With default gamma=1, beta=0, this should hold for the output.
        mean_output_batched = np.mean(normalized_output_batched, axis=-1)
        std_output_batched = np.std(normalized_output_batched, axis=-1)

        print(f"  Mean of output (sample, axis=-1): {mean_output_batched[0,0]:.4f} (expected near 0)")
        print(f"  Std of output (sample, axis=-1): {std_output_batched[0,0]:.4f} (expected near 1)")
        assert np.allclose(mean_output_batched, 0, atol=1e-6), "Mean of normalized output (batched) is not close to 0."
        assert np.allclose(std_output_batched, 1, atol=1e-6), "Std dev of normalized output (batched) is not close to 1."
        print("  Forward pass (batched) PASSED.")
    except Exception as e:
        print(f"  Forward pass (batched) FAILED: {e}")


    # 3. Test Forward Pass with Single Sequence Input
    print("\n3. Test Forward Pass (Single Sequence Input)...")
    input_data_single = np.random.rand(seq_len_test, d_model_cfg) * 5 - 2 # Different random data

    try:
        normalized_output_single = ln_layer.forward(input_data_single)
        print(f"  Input shape: {input_data_single.shape}, Output shape: {normalized_output_single.shape}")
        assert normalized_output_single.shape == input_data_single.shape, "Output shape mismatch for single sequence."

        mean_output_single = np.mean(normalized_output_single, axis=-1)
        std_output_single = np.std(normalized_output_single, axis=-1)
        
        print(f"  Mean of output (sample, axis=-1): {mean_output_single[0]:.4f} (expected near 0)")
        print(f"  Std of output (sample, axis=-1): {std_output_single[0]:.4f} (expected near 1)")
        assert np.allclose(mean_output_single, 0, atol=1e-6), "Mean of normalized output (single) is not close to 0."
        assert np.allclose(std_output_single, 1, atol=1e-6), "Std dev of normalized output (single) is not close to 1."
        print("  Forward pass (single sequence) PASSED.")
    except Exception as e:
        print(f"  Forward pass (single sequence) FAILED: {e}")

    # 4. Test with non-default gamma and beta
    print("\n4. Test with non-default gamma and beta...")
    ln_layer_custom = LayerNormalization(d_model=d_model_cfg, epsilon=epsilon_cfg)
    custom_gamma = np.random.rand(d_model_cfg) * 2 + 0.5 # Random gamma between 0.5 and 2.5
    custom_beta = np.random.rand(d_model_cfg) * 3 - 1.5   # Random beta between -1.5 and 1.5
    ln_layer_custom.gamma = custom_gamma
    ln_layer_custom.beta = custom_beta
    
    input_data_custom = np.random.rand(batch_size_test, seq_len_test, d_model_cfg)

    try:
        output_custom = ln_layer_custom.forward(input_data_custom)
        # The mean of (gamma * x_norm + beta) should be beta (since mean of x_norm is 0)
        # The std of (gamma * x_norm + beta) should be gamma (since std of x_norm is 1, and beta is just a shift)
        # This is an approximation as gamma and beta are vectors, so we check the mean of the means and stds.
        
        # Calculate (x - mean) / sqrt(variance + epsilon) part first
        _mean = np.mean(input_data_custom, axis=-1, keepdims=True)
        _var = np.var(input_data_custom, axis=-1, keepdims=True)
        x_norm_part = (input_data_custom - _mean) / np.sqrt(_var + ln_layer_custom.epsilon)
        
        expected_output_custom = x_norm_part * custom_gamma + custom_beta # Element-wise broadcasting
        
        assert np.allclose(output_custom, expected_output_custom, atol=1e-6), "Output with custom gamma/beta does not match expected."

        # More direct check: mean of output should be around mean of beta, std of output around mean of gamma
        # This is a rough check because gamma and beta are vectors applied element-wise.
        # A more precise check is that for each feature j, output[:,:,j] has mean beta[j] and std gamma[j] IF x_norm was N(0,1)
        # For now, checking element-wise correctness as above is better.
        print(f"  Output with custom gamma/beta matches direct calculation.")
        print("  Custom gamma/beta test PASSED.")
    except Exception as e:
        print(f"  Custom gamma/beta test FAILED: {e}")

    # 5. Test Error Handling for mismatched d_model
    print("\n5. Test Error Handling (mismatched d_model)...")
    wrong_d_model = d_model_cfg + 1
    input_data_wrong_dim = np.random.rand(seq_len_test, wrong_d_model)
    try:
        ln_layer.forward(input_data_wrong_dim)
        print("  Error handling test FAILED: ValueError not raised for mismatched d_model.")
    except ValueError as ve:
        print(f"  Successfully caught ValueError: {ve}")
        print("  Error handling test PASSED.")
    except Exception as e:
        print(f"  Error handling test FAILED with unexpected error: {e}")

    # Testing LayerNormalization Initialization and Forward Pass...
    print("\nTesting LayerNormalization (get_parameters, backward, get_gradients)...")
    try:
        ln_layer_test = LayerNormalization(d_model=d_model_cfg)
        print(f"  Successfully initialized LayerNormalization with d_model={d_model_cfg}")
        assert ln_layer_test.gamma.shape == (d_model_cfg,) and np.all(ln_layer_test.gamma == 1)
        assert ln_layer_test.beta.shape == (d_model_cfg,) and np.all(ln_layer_test.beta == 0)
        print("  Initialization check PASSED.")

        # Prepare dummy inputs/outputs for backward test
        input_batched = np.random.rand(batch_size_test, seq_len_test, d_model_cfg) * 10 - 5
        output_batched = ln_layer_test.forward(input_batched) # Run forward to populate caches
        assert output_batched.shape == input_batched.shape, "Batched output shape mismatch."
        print("  Forward pass completed for backward test setup.")

        # Test get_parameters
        params = ln_layer_test.get_parameters(prefix="test_ln.")
        assert list(params.keys()) == ['test_ln.gamma', 'test_ln.beta'], "get_parameters keys mismatch"
        assert params['test_ln.gamma'].shape == (d_model_cfg,) and params['test_ln.beta'].shape == (d_model_cfg,)
        print("  get_parameters test PASSED.")

        # Test backward and get_gradients (basic check for shapes and existence)
        dummy_d_output = np.random.rand(*output_batched.shape)
        d_input_calc = ln_layer_test.backward(dummy_d_output)
        assert d_input_calc.shape == input_batched.shape, "backward d_input shape mismatch"
        
        grads = ln_layer_test.get_gradients(prefix="test_ln.")
        assert list(grads.keys()) == ['test_ln.gamma', 'test_ln.beta'], "get_gradients keys mismatch"
        assert grads['test_ln.gamma'].shape == (d_model_cfg,) and grads['test_ln.beta'].shape == (d_model_cfg,)
        assert ln_layer_test.grad_gamma is not None and ln_layer_test.grad_beta is not None, "Gradients not computed or stored."
        print("  backward and get_gradients tests PASSED (shapes)." )

    except Exception as e:
        print(f"Error during LayerNormalization tests: {e}")
        import traceback
        traceback.print_exc()

    print("\nAll LayerNormalization validation tests completed.") 