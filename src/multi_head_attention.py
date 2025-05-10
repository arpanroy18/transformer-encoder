"""Multi-Head Attention mechanism for the Transformer model.

This module implements the `MultiHeadAttention` class, a core component of
the Transformer architecture. It allows the model to jointly attend to
information from different representation subspaces at different positions.
The implementation includes:
- Linear projections for queries, keys, and values.
- Splitting inputs into multiple attention heads.
- Scaled dot-product attention mechanism for each head.
- Combining outputs from all heads.
- Optional causal masking for autoregressive tasks (e.g., decoders).
- Support for KV caching to optimize sequential generation.
- Backward pass for gradient computation.

A utility function `create_causal_mask` is also provided.
"""
import numpy as np

class MultiHeadAttention:
    """Implements the Multi-Head Attention mechanism.

    Performs multi-head attention as described in "Attention Is All You Need".
    It projects the queries, keys, and values multiple times with different,
    learned linear projections, allowing the model to attend to information
    from different representation subspaces.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output embeddings.
    n_heads : int
        The number of attention heads. `d_model` must be divisible by `n_heads`.
    dropout_rate : float, optional
        The dropout rate to apply to the attention weights. Defaults to 0.0.

    Attributes
    ----------
    d_model : int
        Dimensionality of the model.
    n_heads : int
        Number of attention heads.
    d_k : int
        Dimensionality of each attention head (`d_model / n_heads`).
    W_q : np.ndarray
        Weight matrix for query projection, shape `(d_model, d_model)`.
    W_k : np.ndarray
        Weight matrix for key projection, shape `(d_model, d_model)`.
    W_v : np.ndarray
        Weight matrix for value projection, shape `(d_model, d_model)`.
    W_o : np.ndarray
        Weight matrix for output projection, shape `(d_model, d_model)`.
    attention_weights : np.ndarray or None
        The attention weights (scores) from the last call to
        `scaled_dot_product_attention`, shape
        `(batch_size, n_heads, seq_len_q, seq_len_k)`.
    q_input_cache, k_input_cache, v_input_cache : np.ndarray or None
        Caches for raw inputs to the forward pass.
    Q_proj_cache, K_proj_cache, V_proj_cache : np.ndarray or None
        Caches for projected Q, K, V (after W_q, W_k, W_v).
    Q_split_cache, K_split_cache, V_split_cache : np.ndarray or None
        Caches for Q, K, V after splitting into heads.
    scaled_dot_product_attention_output_cache : np.ndarray or None
        Cache for the context vector output of SDPA.
    sdpa_Q_cache, sdpa_K_cache, sdpa_V_cache : np.ndarray or None
        Caches for Q, K, V inputs passed to the SDPA function.
    sdpa_softmax_output_cache : np.ndarray or None
        Cache for the softmax output (attention probabilities) within SDPA.
    grad_W_q, grad_W_k, grad_W_v, grad_W_o : np.ndarray or None
        Gradients for the respective weight matrices.
    dropout_rate : float
        The dropout rate to apply to the attention weights.
    training_mode : bool
        Indicates whether the layer is in training mode.

    Raises
    ------
    ValueError
        If `d_model` is not divisible by `n_heads`.
    """
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.0):
        """Initializes the MultiHeadAttention layer.

        Sets up dimensions and initializes weight matrices for Q, K, V projections
        and the final output projection using Glorot uniform initialization.

        Parameters
        ----------
        d_model : int
            Dimensionality of the input and output.
        n_heads : int
            Number of attention heads.
        dropout_rate : float, optional
            The dropout rate to apply to the attention weights. Defaults to 0.0.

        Raises
        ------
        ValueError
            If `d_model` is not divisible by `n_heads`.
        """
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_rate = dropout_rate

        # Weight matrices for Q, K, V, and output projection
        # Initialize with Xavier/Glorot uniform
        limit_qkv = np.sqrt(6. / (d_model + d_model)) # d_model for input, d_model for output of W_q, etc.
        self.W_q = np.random.uniform(-limit_qkv, limit_qkv, (d_model, d_model))
        self.W_k = np.random.uniform(-limit_qkv, limit_qkv, (d_model, d_model))
        self.W_v = np.random.uniform(-limit_qkv, limit_qkv, (d_model, d_model))
        
        limit_o = np.sqrt(6. / (d_model + d_model)) # d_model for input (concatenated heads), d_model for output
        self.W_o = np.random.uniform(-limit_o, limit_o, (d_model, d_model))
        
        self.attention_weights = None # Store attention weights (scores) from scaled_dot_product

        # Caches for backward pass
        self.q_input_cache = None
        self.k_input_cache = None
        self.v_input_cache = None

        self.Q_proj_cache = None # After W_q
        self.K_proj_cache = None # After W_k
        self.V_proj_cache = None # After W_v

        self.Q_split_cache = None # After splitting heads
        self.K_split_cache = None # After splitting heads
        self.V_split_cache = None # After splitting heads

        self.scaled_dot_product_attention_output_cache = None # Output of SDPA (context vector part)
        self.sdpa_Q_cache = None # Q passed to SDPA (already split)
        self.sdpa_K_cache = None # K passed to SDPA (already split)
        self.sdpa_V_cache = None # V passed to SDPA (already split)
        self.sdpa_softmax_output_cache = None # Softmax output within SDPA

        # Gradients for parameters
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None

        self.training_mode = False

    def split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Splits the last dimension of a tensor into multiple heads.

        Reshapes the input tensor `x` from `(batch_size, seq_len, d_model)`
        to `(batch_size, n_heads, seq_len, d_k)` to allow parallel processing
        across heads.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape `(batch_size, seq_len, d_model)`.
        batch_size : int
            The batch size of the input tensor.

        Returns
        -------
        np.ndarray
            Reshaped tensor of shape `(batch_size, n_heads, seq_len, d_k)`.
        """
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        # Transpose to (batch_size, n_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Combines results from multiple attention heads.

        Reshapes the input tensor `x` from `(batch_size, n_heads, seq_len, d_k)`
        back to `(batch_size, seq_len, d_model)` by concatenating head outputs.
        This is the inverse operation of `split_heads`.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape `(batch_size, n_heads, seq_len, d_k)`.
        batch_size : int
            The batch size of the input tensor.

        Returns
        -------
        np.ndarray
            Reshaped tensor of shape `(batch_size, seq_len, d_model)`.
        """
        # x shape: (batch_size, n_heads, seq_len, d_k)
        seq_len = x.shape[2]
        x = x.transpose(0, 2, 1, 3) # (batch_size, seq_len, n_heads, d_k)
        # Reshape to (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None):
        """Computes the scaled dot-product attention.

        Implements the attention mechanism: `output = softmax((Q @ K.T) / sqrt(d_k)) @ V`.
        An optional mask can be applied to prevent attention to certain positions.

        Parameters
        ----------
        Q : np.ndarray
            Query tensor of shape `(batch_size, n_heads, seq_len_q, d_k)`.
        K : np.ndarray
            Key tensor of shape `(batch_size, n_heads, seq_len_k, d_k)`.
        V : np.ndarray
            Value tensor of shape `(batch_size, n_heads, seq_len_v, d_k)`, where
            `seq_len_v` must be equal to `seq_len_k`.
        mask : np.ndarray, optional
            Boolean mask of shape `(..., seq_len_q, seq_len_k)`. Positions where
            `mask` is True will be filled with a very small number before softmax,
            effectively preventing attention to these positions. Defaults to None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - output : np.ndarray
                The context vector (attention output) of shape
                `(batch_size, n_heads, seq_len_q, d_k)`.
            - attention_weights : np.ndarray
                The attention weights (after softmax) of shape
                `(batch_size, n_heads, seq_len_q, seq_len_k)`.
        """
        # Q, K, V shapes: (batch_size, n_heads, seq_len_q, d_k)
        # K.T (transpose last two dims): (batch_size, n_heads, d_k, seq_len_k)
        matmul_qk = np.matmul(Q, K.transpose(0, 1, 3, 2))  # (batch_size, n_heads, seq_len_q, seq_len_k)
        
        # Scale matmul_qk
        scaled_attention_logits = matmul_qk / np.sqrt(self.d_k)

        # Apply mask (if provided)
        if mask is not None:
            fill_value = np.finfo(scaled_attention_logits.dtype).min
            scaled_attention_logits = np.where(mask, fill_value, scaled_attention_logits)

        # SoftMax activation
        # Apply softmax along the last axis (seq_len_k)
        exp_logits = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
        attention_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        self.sdpa_softmax_output_cache = attention_probs # Save for backward pass

        # Apply dropout to attention_probs if dropout_rate > 0.0
        if self.dropout_rate > 0.0 and self.training_mode: # Assume self.training_mode is set appropriately
            # Note: Proper dropout requires scaling during inference or inverse dropout during training.
            # Here, simple dropout for training, assuming scaling is handled elsewhere or inference has dropout_rate=0.
            dropout_mask = np.random.binomial(1, 1.0 - self.dropout_rate, size=attention_probs.shape)
            attention_probs_dropped = attention_probs * dropout_mask / (1.0 - self.dropout_rate) # Inverted dropout
        else:
            attention_probs_dropped = attention_probs

        self.attention_weights = attention_probs # Store original probabilities for inspection if needed
        # Use attention_probs_dropped for the matmul with V
        output = np.matmul(attention_probs_dropped, V)
        return output, self.attention_weights # Return original weights for inspection, output is from (potentially) dropped weights

    def forward(self, q_input: np.ndarray, k_input: np.ndarray, v_input: np.ndarray, mask: np.ndarray = None, kv_cache: dict = None, training_mode: bool = False):
        """Performs the forward pass of the MultiHeadAttention layer.

        Steps:
        1. Linearly project `q_input`, `k_input`, `v_input` using `W_q`, `W_k`, `W_v`.
        2. Split the projected Q, K, V into multiple heads.
        3. If KV caching is enabled and cache is provided:
           - Concatenate current K, V with cached K, V.
           - Update the cache.
           - Use combined K, V for attention. `sdpa_mask` is set to `None`.
        4. Apply `scaled_dot_product_attention` to each head.
        5. Combine the outputs (context vectors) from each head.
        6. Apply the final linear projection `W_o`.

        Parameters
        ----------
        q_input : np.ndarray
            Query input tensor of shape `(batch_size, seq_len_q, d_model)`.
        k_input : np.ndarray
            Key input tensor of shape `(batch_size, seq_len_k, d_model)`.
        v_input : np.ndarray
            Value input tensor of shape `(batch_size, seq_len_v, d_model)`.
            `seq_len_k` and `seq_len_v` must be the same.
        mask : np.ndarray, optional
            Mask to be applied in scaled dot-product attention. Its shape should
            be broadcastable to `(batch_size, n_heads, seq_len_q, seq_len_k)`.
            Typically a causal mask for self-attention. Defaults to None.
        kv_cache : dict, optional
            A dictionary for KV caching. If provided, it's expected to contain
            keys 'k' and 'v' storing past key and value tensors for each head.
            The cache is updated in-place. Defaults to None.
        training_mode : bool, optional
            Indicates whether the layer is in training mode. Defaults to False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - output : np.ndarray
                The final output of the multi-head attention layer, of shape
                `(batch_size, seq_len_q, d_model)`.
            - attention_weights : np.ndarray
                The attention weights from the scaled dot-product attention,
                shape `(batch_size, n_heads, seq_len_q, seq_len_k_effective)`,
                where `seq_len_k_effective` includes cached length if used.
        """
        self.training_mode = training_mode # Store training_mode for dropout
        self.q_input_cache, self.k_input_cache, self.v_input_cache = q_input, k_input, v_input # Caching for backward pass (original inputs)
        batch_size = q_input.shape[0]
        seq_len_q = q_input.shape[1]

        # 1. Linear projections for current Q, K, V
        Q_proj = np.dot(q_input, self.W_q)  # (batch_size, seq_len_q, d_model)
        K_proj = np.dot(k_input, self.W_k)  # (batch_size, seq_len_k_current, d_model)
        V_proj = np.dot(v_input, self.W_v)  # (batch_size, seq_len_v_current, d_model)
        
        # Cache projections for backward pass (based on current inputs only)
        self.Q_proj_cache, self.K_proj_cache, self.V_proj_cache = Q_proj, K_proj, V_proj

        # 2. Split heads for current Q, K, V
        Q_split = self.split_heads(Q_proj, batch_size)  # (batch_size, n_heads, seq_len_q, d_k)
        K_curr_split = self.split_heads(K_proj, batch_size)  # (batch_size, n_heads, seq_len_k_current, d_k)
        V_curr_split = self.split_heads(V_proj, batch_size)  # (batch_size, n_heads, seq_len_v_current, d_k)

        # Cache split Q for backward pass
        self.Q_split_cache = Q_split 
        # K_split_cache and V_split_cache for backward pass will be based on the K/V actually used in SDPA
        # So, they are set after KV cache logic.

        sdpa_mask = mask # Default mask

        if kv_cache is not None:
            # KV Caching is active (typically seq_len_q will be 1)
            if 'k' in kv_cache and kv_cache['k'] is not None:
                # Concatenate past K, V with current K, V
                past_K = kv_cache['k'] # Shape: (batch_size, n_heads, past_seq_len, d_k)
                past_V = kv_cache['v'] # Shape: (batch_size, n_heads, past_seq_len, d_k)
                
                K_combined = np.concatenate((past_K, K_curr_split), axis=2)
                V_combined = np.concatenate((past_V, V_curr_split), axis=2)
            else:
                # First token with KV cache, or cache was empty
                K_combined = K_curr_split
                V_combined = V_curr_split
            
            # Update cache with the new combined K, V
            kv_cache['k'] = K_combined
            kv_cache['v'] = V_combined

            K_to_use_in_sdpa = K_combined
            V_to_use_in_sdpa = V_combined
            
            # When using KV cache with a single query token, Q attends to all K in cache + current K.
            # The causal nature is handled by sequential generation. So, no additional mask within SDPA for this Q.
            # Mask applies to QK^T. If Q is current, K is all past+current, we want Q to see all of K.
            sdpa_mask = None 
        else:
            # No KV Caching / Training mode
            K_to_use_in_sdpa = K_curr_split # K_curr_split is from the full k_input if no cache
            V_to_use_in_sdpa = V_curr_split # V_curr_split is from the full v_input if no cache

        # Update K_split_cache and V_split_cache for backward pass to reflect what was used in SDPA
        self.K_split_cache = K_to_use_in_sdpa
        self.V_split_cache = V_to_use_in_sdpa
        
        # For SDPA backward pass: Q, K, V that went into SDPA
        self.sdpa_Q_cache, self.sdpa_K_cache, self.sdpa_V_cache = Q_split, K_to_use_in_sdpa, V_to_use_in_sdpa

        # 3. Scaled dot-product attention
        context_vector, current_attention_weights = self.scaled_dot_product_attention(
            Q_split, K_to_use_in_sdpa, V_to_use_in_sdpa, sdpa_mask
        )
        self.attention_weights = current_attention_weights # Store for inspection
        self.scaled_dot_product_attention_output_cache = context_vector
        
        # 4. Combine heads
        # context_vector shape: (batch_size, n_heads, seq_len_q, d_k)
        combined_context = self.combine_heads(context_vector, batch_size)  # (batch_size, seq_len_q, d_model)
        
        # 5. Final linear projection
        output = np.dot(combined_context, self.W_o)  # (batch_size, seq_len_q, d_model)
        
        return output, self.attention_weights # Return attention_weights as well

    def get_parameters(self, prefix="") -> dict:
        """Retrieves the learnable parameters of the layer.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to parameter names, by default "".

        Returns
        -------
        dict
            A dictionary mapping parameter names (W_q, W_k, W_v, W_o)
            to their NumPy array values.
        """
        return {
            f"{prefix}W_q": self.W_q,
            f"{prefix}W_k": self.W_k,
            f"{prefix}W_v": self.W_v,
            f"{prefix}W_o": self.W_o,
        }

    def _scaled_dot_product_attention_backward(self, d_context_vector: np.ndarray):
        """Computes gradients for the scaled dot-product attention part.

        This is a helper method for the main `backward` pass. It computes
        dL/dQ_sdpa, dL/dK_sdpa, dL/dV_sdpa, where Q_sdpa, K_sdpa, V_sdpa are
        the inputs to the `scaled_dot_product_attention` function during forward.

        Parameters
        ----------
        d_context_vector : np.ndarray
            Gradient of the loss with respect to the output of the
            `scaled_dot_product_attention` (i.e., dL/d(context_vector)).
            Shape `(batch_size, n_heads, seq_len_q, d_k)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - d_Q_sdpa : Gradient w.r.t. Q_sdpa, shape `(batch, n_heads, seq_len_q, d_k)`.
            - d_K_sdpa : Gradient w.r.t. K_sdpa, shape `(batch, n_heads, seq_len_k, d_k)`.
            - d_V_sdpa : Gradient w.r.t. V_sdpa, shape `(batch, n_heads, seq_len_v, d_k)`.
        """
        # d_context_vector shape: (batch, n_heads, seq_len_q, d_k)
        # self.sdpa_softmax_output_cache (attention_weights) shape: (batch, n_heads, seq_len_q, seq_len_k)
        # self.sdpa_V_cache shape: (batch, n_heads, seq_len_v=seq_len_k, d_k)
        # self.sdpa_K_cache shape: (batch, n_heads, seq_len_k, d_k)
        # self.sdpa_Q_cache shape: (batch, n_heads, seq_len_q, d_k)

        # Grad w.r.t. V in SDPA: dL/dV = Softmax(QK^T/sqrt(dk)).T @ dL/dContext
        # (att_weights.T @ d_context_vector)
        d_V_sdpa = np.matmul(self.sdpa_softmax_output_cache.transpose(0,1,3,2), d_context_vector)

        # Grad w.r.t. attention_weights (softmax output): dL/dAtt = dL/dContext @ V.T
        d_attention_weights = np.matmul(d_context_vector, self.sdpa_V_cache.transpose(0,1,3,2))

        # Grad w.r.t. scaled_attention_logits (input to softmax)
        # If S = softmax(L), dL/dL_i = S_i * (dL/dS_i - sum_j(dL/dS_j * S_j))
        # More simply, for dL/dL_i = dL/dS_i * S_i - S_i * sum_j(dL/dS_j * S_j)  <-- this is tricky
        # A common way: dL/dL = (dL/dS - sum(dL/dS * S)) * S ... where sum and * are element-wise then sum
        # Or, for each row S_r of softmax output, and grad dS_r: dL_r = (S_r * dS_r) - S_r * np.dot(S_r, dS_r)
        # This derivative is dL/d(logits) = (P - y) if loss is CE. Here we have dL/d(Probabilities)
        # dL/d_logits = P * (dL/dP - sum(dL/dP_j * P_j)) where P is softmax output
        # Let P = self.sdpa_softmax_output_cache, dP = d_attention_weights
        # d_scaled_logits_j = P_j * (dP_j - sum_k(dP_k * P_k)) (sum over k, for fixed j)
        
        P = self.sdpa_softmax_output_cache # (batch, n_heads, seq_len_q, seq_len_k)
        dP = d_attention_weights          # (batch, n_heads, seq_len_q, seq_len_k)
        
        # For each row of P (last dim), compute S_i * (dL/dS_i - sum_j dL/dS_j * S_j)
        # S_dot_dLdS = np.sum(P * dP, axis=-1, keepdims=True) # Sum over key_len for each query
        # d_scaled_attention_logits = P * (dP - S_dot_dLdS)
        # This one works for dL/dLogits where L is input to softmax: dL/dLogits = P * dL/dP - P * sum(P * dL/dP, axis=-1, keepdims=True)
        sum_val = np.sum(P * dP, axis=-1, keepdims=True)
        d_scaled_attention_logits = P * (dP - sum_val)

        # Apply mask if it was applied in forward. Gradient does not flow to masked positions.
        # This should be handled naturally if mask made logits very small, so P near zero, so grad near zero.
        # Or if mask was additive, d_mask_term = d_scaled_attention_logits. Mask grad is zero.

        # Grad w.r.t. (Q @ K.T / sqrt(d_k))
        # dL/d(QK^T/sqrt(dk)) = d_scaled_attention_logits
        d_matmul_qk_scaled = d_scaled_attention_logits / np.sqrt(self.d_k)

        # Grad w.r.t. Q in SDPA: dL/dQ = dL/d(QK^T scaled) @ K
        d_Q_sdpa = np.matmul(d_matmul_qk_scaled, self.sdpa_K_cache)

        # Grad w.r.t. K in SDPA: dL/dK = (dL/d(QK^T scaled)).T @ Q
        d_K_sdpa = np.matmul(d_matmul_qk_scaled.transpose(0,1,3,2), self.sdpa_Q_cache)
        
        return d_Q_sdpa, d_K_sdpa, d_V_sdpa

    def backward(self, d_output: np.ndarray):
        """Computes gradients for the MultiHeadAttention layer.

        This involves backpropagating the gradient `d_output` (dL/d(final_output))
        through the final projection (W_o), head combination, scaled dot-product
        attention, head splitting, and initial Q, K, V projections (W_q, W_k, W_v).

        KV cache logic is not part of backpropagation as it's used during inference.
        Gradients for W_q, W_k, W_v, W_o are stored in `self.grad_W_q`, etc.

        Parameters
        ----------
        d_output : np.ndarray
            Gradient of the loss with respect to the output of this MHA layer.
            Shape `(batch_size, seq_len_q, d_model)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - d_q_input : Gradient w.r.t. the original `q_input` to `forward()`.
            - d_k_input : Gradient w.r.t. the original `k_input` to `forward()`.
            - d_v_input : Gradient w.r.t. the original `v_input` to `forward()`.
              All have shape `(batch_size, seq_len, d_model)` corresponding to inputs.
        """
        batch_size = d_output.shape[0]

        # 5. Backward through final linear projection (W_o)
        # d_output is dL/d(final_output). combined_context was input to W_o.
        # combined_context shape: (batch, seq_len_q, d_model)
        # W_o shape: (d_model, d_model)
        # d_output shape: (batch, seq_len_q, d_model)
        # grad_W_o: (d_model, d_model)
        # Need to use combined_context from forward, which is output of combine_heads(context_vector)
        # And context_vector is self.scaled_dot_product_attention_output_cache
        combined_context = self.combine_heads(self.scaled_dot_product_attention_output_cache, batch_size)
        
        # Reshape for dot product to get grad_W_o (sum over batch and seq_len)
        # combined_context_reshaped (batch*seq, d_model), d_output_reshaped (batch*seq, d_model)
        if combined_context.ndim == 3:
            bs, sl, dm = combined_context.shape
            combined_context_reshaped = combined_context.reshape(bs * sl, dm)
            d_output_reshaped = d_output.reshape(bs * sl, dm)
            self.grad_W_o = np.dot(combined_context_reshaped.T, d_output_reshaped)
        else: # Should always be 3D due to batching
            self.grad_W_o = np.dot(combined_context.T, d_output)

        # Gradient w.r.t. combined_context (dL/d(combined_context))
        d_combined_context = np.dot(d_output, self.W_o.T) # (batch, seq_len_q, d_model)

        # 4. Backward through combine_heads
        # d_combined_context is grad for output of combine_heads.
        # Input to combine_heads was context_vector from SDPA.
        # combine_heads is just a reshape and transpose, so backward is inverse ops.
        # d_context_vector shape: (batch, n_heads, seq_len_q, d_k)
        d_context_vector = self.split_heads(d_combined_context, batch_size) # Inverse of combine is split

        # 3. Backward through scaled_dot_product_attention
        # d_context_vector is dL/d(SDPA_output)
        d_Q_sdpa, d_K_sdpa, d_V_sdpa = self._scaled_dot_product_attention_backward(d_context_vector)
        # d_Q_sdpa, d_K_sdpa, d_V_sdpa shapes: (batch, n_heads, seq_len, d_k)

        # 2. Backward through split_heads for Q, K, V
        # d_Q_proj = combine_heads(d_Q_sdpa), etc.
        d_Q_proj = self.combine_heads(d_Q_sdpa, batch_size) # (batch, seq_len_q, d_model)
        d_K_proj = self.combine_heads(d_K_sdpa, batch_size) # (batch, seq_len_k, d_model)
        d_V_proj = self.combine_heads(d_V_sdpa, batch_size) # (batch, seq_len_v, d_model)

        # 1. Backward through linear projections (W_q, W_k, W_v)
        # d_Q_proj is dL/d(Q_proj). q_input_cache was input to W_q.
        # grad_W_q shape (d_model, d_model)
        # q_input_cache (batch, seq, d_model), d_Q_proj (batch, seq, d_model)
        if self.q_input_cache.ndim == 3:
            bs_q, sl_q, dm_q = self.q_input_cache.shape
            q_input_reshaped = self.q_input_cache.reshape(bs_q * sl_q, dm_q)
            d_Q_proj_reshaped = d_Q_proj.reshape(bs_q * sl_q, dm_q)
            self.grad_W_q = np.dot(q_input_reshaped.T, d_Q_proj_reshaped)
        else:
            self.grad_W_q = np.dot(self.q_input_cache.T, d_Q_proj)
        d_q_input = np.dot(d_Q_proj, self.W_q.T)

        if self.k_input_cache.ndim == 3:
            bs_k, sl_k, dm_k = self.k_input_cache.shape
            k_input_reshaped = self.k_input_cache.reshape(bs_k * sl_k, dm_k)
            d_K_proj_reshaped = d_K_proj.reshape(bs_k * sl_k, dm_k)
            self.grad_W_k = np.dot(k_input_reshaped.T, d_K_proj_reshaped)
        else:
            self.grad_W_k = np.dot(self.k_input_cache.T, d_K_proj)
        d_k_input = np.dot(d_K_proj, self.W_k.T)

        if self.v_input_cache.ndim == 3:
            bs_v, sl_v, dm_v = self.v_input_cache.shape
            v_input_reshaped = self.v_input_cache.reshape(bs_v * sl_v, dm_v)
            d_V_proj_reshaped = d_V_proj.reshape(bs_v * sl_v, dm_v)
            self.grad_W_v = np.dot(v_input_reshaped.T, d_V_proj_reshaped)
        else:
            self.grad_W_v = np.dot(self.v_input_cache.T, d_V_proj)
        d_v_input = np.dot(d_V_proj, self.W_v.T)
        
        return d_q_input, d_k_input, d_v_input

    def get_gradients(self, prefix="") -> dict:
        """Retrieves the computed gradients for the layer's parameters.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to gradient names, by default "".

        Returns
        -------
        dict
            A dictionary mapping gradient names (for W_q, W_k, W_v, W_o)
            to their NumPy array values. If `backward` has not been called or
            resulted in zero gradients, they might be None or zeros.
        """
        # Ensure grads are initialized if backward didn't run or d_output was zero
        if self.grad_W_q is None: self.grad_W_q = np.zeros_like(self.W_q)
        if self.grad_W_k is None: self.grad_W_k = np.zeros_like(self.W_k)
        if self.grad_W_v is None: self.grad_W_v = np.zeros_like(self.W_v)
        if self.grad_W_o is None: self.grad_W_o = np.zeros_like(self.W_o)
        return {
            f"{prefix}W_q": self.grad_W_q,
            f"{prefix}W_k": self.grad_W_k,
            f"{prefix}W_v": self.grad_W_v,
            f"{prefix}W_o": self.grad_W_o,
        }

def create_causal_mask(seq_len: int) -> np.ndarray:
    """Creates a causal mask for self-attention mechanisms.

    The mask prevents positions from attending to subsequent positions.
    It's used in decoders to ensure autoregressive generation.

    The mask is shaped `(1, 1, seq_len, seq_len)` to be broadcastable
    with attention logits of shape `(batch_size, n_heads, seq_len, seq_len)`.
    Positions where the mask is `True` should be masked out (e.g., by setting
    logits to a very small number before softmax).

    Parameters
    ----------
    seq_len : int
        The length of the sequence for which the mask is to be created.

    Returns
    -------
    np.ndarray
        A boolean causal mask of shape `(1, 1, seq_len, seq_len)`.
        `True` indicates a position that should be masked (future position).
        `False` indicates a position that can be attended to (current or past).
    """
    mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=1).astype(bool)
    return mask

if __name__ == "__main__":
    # Add import for config if it was missing from the original file's __main__
    import sys, os
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import config # Make sure config is imported for __main__

    cfg = config.get_config()
    d_model_cfg = cfg['d_model']
    n_heads_cfg = cfg['n_heads']
    dropout_rate_cfg = cfg.get('dropout_rate', 0.0) # Get dropout rate

    print("\n1. Test Initialization...")
    try:
        # Pass dropout_rate_cfg
        mha = MultiHeadAttention(d_model=d_model_cfg, n_heads=n_heads_cfg, dropout_rate=dropout_rate_cfg)
        print(f"  Successfully initialized MultiHeadAttention with d_model={d_model_cfg}, n_heads={n_heads_cfg}, dropout_rate={dropout_rate_cfg}")
        # ... (rest of __main__ initialization assertions)
    except Exception as e:
        print(f"FAIL: Initialization failed: {e}")
        raise

    # ... (rest of __main__) 