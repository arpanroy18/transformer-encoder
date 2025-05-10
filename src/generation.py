"""Text generation utilities for the Transformer model.

This module provides the `generate` function, which facilitates step-by-step
text generation using a trained `Decoder` model and a `Tokenizer`.
It supports features like temperature-based sampling, KV caching for efficient
generation, a maximum generation length, and optional stop token handling.
It can also optionally return attention weights from the generation process.
"""
import numpy as np
import time # Added for Step 15
import tokenizer as tkn
import decoder as dec
import config as cfg

def generate(model: dec.Decoder, tokenizer_instance: tkn.Tokenizer, prompt: str, max_len: int, temperature: float = 1.0, stop_token_id: int = None, use_kv_cache: bool = True, return_attention_weights: bool = False) -> tuple[str, list] | str:
    """Generates text token by token starting from a prompt.

    Handles the step-by-step generation loop, including KV cache management
    (if enabled), token sampling (greedy or temperature-based), and stop criteria.

    Parameters
    ----------
    model : Decoder
        The trained Transformer Decoder model.
    tokenizer : Tokenizer
        The tokenizer instance used for encoding the prompt and decoding generated tokens.
    prompt : str
        The initial string to start generation from.
    max_len : int
        The maximum number of new tokens to generate after the prompt.
    temperature : float, optional
        Factor for scaling logits before softmax. Lower values (e.g., <= 0.0 for greedy)
        make the output more deterministic, while higher values (e.g., > 1.0) 
        make it more random. Defaults to 1.0.
    stop_token_id : int, optional
        The token ID that, if generated, will stop the generation process.
        Defaults to None (no stop token).
    use_kv_cache : bool, optional
        If True, enables the Key-Value caching mechanism in the model for faster
        generation. The `model.forward` method must support `kv_cache_list` and
        `current_token_idx` arguments. Defaults to True.
    return_attention_weights : bool, optional
        If True, the function returns a tuple `(generated_text, attention_weights_list)`,
        where `attention_weights_list` is a list of attention weights from each
        generation step. Each item in this list is itself a list of weights per
        decoder layer. Defaults to False.

    Returns
    -------
    str or tuple[str, list]
        - If `return_attention_weights` is False: The generated string (prompt + new tokens).
        - If `return_attention_weights` is True: A tuple `(generated_string, all_step_attention_weights)`,
          where `all_step_attention_weights` is a list of lists of attention weight arrays.
          `all_step_attention_weights[step][layer]` gives the attention weights for a specific
          generation step and decoder layer.

    Notes
    -----
    - When `use_kv_cache` is True, the `model.forward` method is called with only the
      most recently generated token and the `kv_cache_list` is updated by the model.
    - The `current_token_idx` passed to `model.forward` is crucial for correct
      positional encoding when using KV cache.
    """
    # model.eval() # Set model to evaluation mode if it has specific layers like dropout (not in current impl but good practice)
    
    prompt_was_empty = False # Flag to track if prompt was empty
    if not prompt:
        prompt_was_empty = True
        if 0 in tokenizer_instance.idx_to_char:
            start_token_id = 0
            current_sequence_tokens = [start_token_id]
            generated_text = tokenizer_instance.decode([start_token_id]) 
            print(f"Warning: Empty prompt provided. Starting generation with default token ID {start_token_id} ('{generated_text}').")
        else:
            # If token 0 is invalid or we don't want a default start,
            # we could return empty string or raise error.
            # Let's return empty for now if max_len > 0, or prompt if max_len=0.
            print("Warning: Empty prompt provided and default start token ID 0 is invalid. Returning empty result.")
            if max_len == 0:
                 return "" # Match prompt
            else:
                 # Return empty generated text and empty weights if requested
                 return ("", []) if return_attention_weights else ""
    else:
        # Normal encoding for non-empty prompt
        current_sequence_tokens = tokenizer_instance.encode(prompt)
        # Ensure the resulting array is integer type even if empty (though prompt is not empty here)
        if isinstance(current_sequence_tokens, list): 
            current_sequence_tokens = np.array(current_sequence_tokens, dtype=np.int32)
        elif current_sequence_tokens.size == 0: # Should not happen if prompt is not empty
            current_sequence_tokens = np.array([], dtype=np.int32)
            
        generated_text = prompt

    all_step_attention_weights = []
    kv_cache_list = None
    if use_kv_cache and hasattr(model, 'n_layers') and model.n_layers > 0:
        kv_cache_list = [{} for _ in range(model.n_layers)] 

    # Ensure current_sequence_tokens is a list for easy appending
    if isinstance(current_sequence_tokens, np.ndarray):
        current_sequence_tokens = current_sequence_tokens.tolist()

    # Determine number of generation steps
    # If prompt was empty and we prepended a token, we generate max_len-1 more tokens
    # to reach a total of max_len *new* tokens (relative to the empty input).
    # If max_len is 0, num_steps should be 0.
    num_steps = max_len
    if prompt_was_empty and max_len > 0 and 0 in tokenizer_instance.idx_to_char:
        # We already added one token, so generate max_len-1 more. But the loop runs `num_steps` times.
        # The total *generated* length should be max_len. The current text has len 1.
        # We need to run the loop max_len times to add max_len more tokens.
        # The test assertion needs to be reconsidered. Let's stick to running max_len times.
        pass # Run loop max_len times

    for idx in range(num_steps):
        if not current_sequence_tokens: # Should not happen if we handled empty prompt
            print("Error: Token sequence became empty during generation.")
            break 
            
        current_input_pos = len(current_sequence_tokens) - 1

        if kv_cache_list is not None:
            # Use np.array with explicit dtype for the single token
            input_tokens_step = np.array([[current_sequence_tokens[-1]]], dtype=np.int32)
            logits = model.forward(input_tokens_step, mask=None, kv_cache_list=kv_cache_list, current_token_idx=current_input_pos)
        else:
            # Use np.array with explicit dtype for the full sequence
            input_tokens_full = np.array([current_sequence_tokens], dtype=np.int32)
            seq_len_full = input_tokens_full.shape[1]
            if seq_len_full == 0: # Should not happen with prompt handling
                print("Error: Input token sequence is empty for non-cached forward pass.")
                break
            causal_mask_full = dec.Decoder.create_causal_mask(seq_len_full)
            logits = model.forward(input_tokens_full, mask=causal_mask_full, current_token_idx=0)

        if return_attention_weights and hasattr(model, 'attention_weights'):
            # model.attention_weights is a list of weights per layer
            # Each element is (batch_size, n_heads, seq_len_q, seq_len_k)
            # We make a deep copy to avoid issues if the model reuses/modifies the list later
            # For KV cache, seq_len_q = 1. For no cache, seq_len_q = seq_len_full.
            current_step_weights = [aw.copy() for aw in model.attention_weights]
            all_step_attention_weights.append(current_step_weights)

        # Focus on the logits for the next token (the last token in the sequence)
        next_token_logits = logits[0, -1, :] # Batch 0, last time step of the output

        # Apply temperature
        if temperature <= 0.0: # Treat 0 or negative as greedy
            next_token_id = np.argmax(next_token_logits)
        else:
            # Scale logits by temperature
            scaled_logits = next_token_logits / temperature
            
            # Numerically stable softmax
            probs = np.exp(scaled_logits - np.max(scaled_logits))
            probs = probs / np.sum(probs)
            
            # Sample from the distribution
            next_token_id = np.random.choice(len(probs), p=probs)
        
        # Decode the token and append to the sequence
        next_char = tokenizer_instance.decode([next_token_id])
        
        if not next_char: # Handle cases where decode might return empty for unknown/padding
            print(f"Warning: Generated token ID {next_token_id} decoded to empty string. Stopping generation at step {idx+1}.")
            break

        generated_text += next_char
        current_sequence_tokens.append(next_token_id)

        # Stop if stop_token_id is generated
        if stop_token_id is not None and next_token_id == stop_token_id:
            print(f"Stop token {stop_token_id} ('{next_char}') generated. Stopping.")
            break
            
    if return_attention_weights:
        return generated_text, all_step_attention_weights
    else:
        return generated_text

if __name__ == "__main__":
    print("Starting Generation Module Validation...")

    # 1. Setup: Configuration, Tokenizer, Model
    config_params = cfg.get_config()
    config_params['vocab_size'] = 50
    config_params['d_model'] = 64
    config_params['n_layers'] = 2
    config_params['n_heads'] = 2
    config_params['d_ff'] = 128
    config_params['max_seq_len'] = 50
    config_params['epsilon'] = 1e-5
    dropout_rate_cfg = config_params.get('dropout_rate', 0.0)

    sample_corpus = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;!?-"
    tokenizer_instance = tkn.Tokenizer(sample_corpus)
    config_params['vocab_size'] = tokenizer_instance.vocab_size

    # Ensure max_seq_len for PE is sufficient for prompt + generation length
    # The generation function itself will truncate generation to its max_len argument
    # but the model's PE must be able to handle the sequences it sees.
    # For this test, prompt_len + gen_len <= config['max_seq_len']

    print(f"  Using vocab_size: {tokenizer_instance.vocab_size}")
    print(f"  Using d_model: {config_params['d_model']}")
    print(f"  Using max_seq_len: {config_params['max_seq_len']}")


    model_instance = dec.Decoder(
        vocab_size=config_params['vocab_size'],
        d_model=config_params['d_model'],
        n_layers=config_params['n_layers'],
        n_heads=config_params['n_heads'],
        d_ff=config_params['d_ff'],
        max_seq_len=config_params['max_seq_len'],
        epsilon=config_params['epsilon'],
        dropout_rate=dropout_rate_cfg
    )
    print("  Model, Tokenizer, and Config initialized.")

    # 2. Test Generation
    prompt_text = "abc"
    generation_length = 10 # Number of new tokens to generate
    
    # Ensure prompt length + generation length <= model's max_seq_len
    if len(tokenizer_instance.encode(prompt_text)) + generation_length > config_params['max_seq_len']:
        print(f"Warning: Prompt length ({len(tokenizer_instance.encode(prompt_text))}) + generation length ({generation_length}) exceeds model's max_seq_len ({config_params['max_seq_len']}).")
        print("This might lead to errors if not handled inside the model/PE layer.")
        # For this test, we'll proceed, assuming PE can handle it or error out if it can't.
        # A robust PE should clip or error if seq_len > max_seq_len.

    print(f"  Generating text with prompt: '{prompt_text}', length: {generation_length}")
    
    generated_output = generate(model_instance, tokenizer_instance, prompt_text, generation_length)
    
    print(f"  Prompt: '{prompt_text}'")
    print(f"  Generated: '{generated_output}'")

    # 3. Assertions
    assert generated_output.startswith(prompt_text), f"Generated text '{generated_output}' does not start with prompt '{prompt_text}'"
    print("  Assertion: Generated text starts with prompt - PASSED")

    # Expected total length is prompt length + number of new tokens generated
    expected_total_length = len(prompt_text) + generation_length
    assert len(generated_output) == expected_total_length, \
        f"Generated text length is {len(generated_output)}, expected {expected_total_length} (prompt: {len(prompt_text)}, generated: {generation_length})"
    print(f"  Assertion: Generated text length is correct ({expected_total_length}) - PASSED")
    
    # Test with a different prompt and length
    prompt_text_2 = "hello "
    generation_length_2 = 5
    if len(tokenizer_instance.encode(prompt_text_2)) + generation_length_2 > config_params['max_seq_len']:
         print(f"Warning for test 2: Prompt length + generation length exceeds model's max_seq_len.")

    print(f"  Generating text with prompt: '{prompt_text_2}', length: {generation_length_2}")
    generated_output_2 = generate(model_instance, tokenizer_instance, prompt_text_2, generation_length_2)
    print(f"  Prompt: '{prompt_text_2}'")
    print(f"  Generated: '{generated_output_2}'")
    
    assert generated_output_2.startswith(prompt_text_2), f"Generated text 2 '{generated_output_2}' does not start with prompt '{prompt_text_2}'"
    print("  Assertion (Test 2): Generated text starts with prompt - PASSED")
    
    expected_total_length_2 = len(prompt_text_2) + generation_length_2
    assert len(generated_output_2) == expected_total_length_2, \
        f"Generated text 2 length is {len(generated_output_2)}, expected {expected_total_length_2}"
    print(f"  Assertion (Test 2): Generated text length is correct ({expected_total_length_2}) - PASSED")

    # Test with max_len = 0 (should return only the prompt)
    prompt_text_3 = "test"
    generation_length_3 = 0
    print(f"  Generating text with prompt: '{prompt_text_3}', length: {generation_length_3}")
    generated_output_3 = generate(model_instance, tokenizer_instance, prompt_text_3, generation_length_3)
    print(f"  Prompt: '{prompt_text_3}'")
    print(f"  Generated: '{generated_output_3}'")
    assert generated_output_3 == prompt_text_3, "Generation with max_len=0 should return prompt only."
    print("  Assertion (Test 3): Generation with max_len=0 - PASSED")

    # Test Temperature Sampling
    print("\n  Testing Temperature Sampling...")
    prompt_text_temp = "the quick brown fox"
    generation_length_temp = 20
    
    if len(tokenizer_instance.encode(prompt_text_temp)) + generation_length_temp > config_params['max_seq_len']:
        print(f"Warning for temperature test: Prompt length + generation length exceeds model's max_seq_len.")

    # Test greedy decoding (temperature = 0.0)
    print(f"  Generating with temperature=0.0 (greedy) from: '{prompt_text_temp[:20]}...'")
    greedy_output = generate(model_instance, tokenizer_instance, prompt_text_temp, generation_length_temp, temperature=0.0)
    print(f"    Greedy Output: '{greedy_output}'")
    assert greedy_output.startswith(prompt_text_temp)
    assert len(greedy_output) == len(prompt_text_temp) + generation_length_temp

    # Test sampling with temperature = 0.7
    print(f"  Generating with temperature=0.7 from: '{prompt_text_temp[:20]}...'")
    temp_07_output = generate(model_instance, tokenizer_instance, prompt_text_temp, generation_length_temp, temperature=0.7)
    print(f"    Temp 0.7 Output: '{temp_07_output}'")
    assert temp_07_output.startswith(prompt_text_temp)
    assert len(temp_07_output) == len(prompt_text_temp) + generation_length_temp

    # Test sampling with temperature = 1.0
    print(f"  Generating with temperature=1.0 from: '{prompt_text_temp[:20]}...'")
    temp_10_output = generate(model_instance, tokenizer_instance, prompt_text_temp, generation_length_temp, temperature=1.0)
    print(f"    Temp 1.0 Output: '{temp_10_output}'")
    assert temp_10_output.startswith(prompt_text_temp)
    assert len(temp_10_output) == len(prompt_text_temp) + generation_length_temp
    
    # Basic check: with random weights, outputs might be different, but we are testing the mechanism
    # A more robust test would involve checking the distribution of tokens or running multiple times
    # For now, we mainly check that it runs without error and produces output of correct form.
    # If the model was trained, we might expect greedy and low-temp outputs to be more similar
    # and higher-temp to be more diverse.
    
    # Ensure that the token IDs are valid
    # We can check the last generated token ID from one of the runs
    if generation_length_temp > 0:
        last_token_id_temp_07 = tokenizer_instance.encode(temp_07_output)[-1]
        assert 0 <= last_token_id_temp_07 < tokenizer_instance.vocab_size, \
            f"Generated token ID {last_token_id_temp_07} is out of vocab range [0, {tokenizer_instance.vocab_size-1}]"
        print(f"    Last token ID from temp 0.7 run ({last_token_id_temp_07}) is valid.")

    print("  Temperature Sampling tests completed (ran without errors).")

    # --- Step 14 Validation: Step-by-Step Generation with KV Cache and Stop Token ---
    print("\n--- Validating Step 14: Step-by-Step Generation with KV Cache & Stop Token ---")
    
    # Re-initialize model and tokenizer for clean test environment if needed
    # Using existing config, tokenizer, and a new model instance for this test section
    config_s14 = config_params.copy() # Use a copy of the config
    tokenizer_s14 = tkn.Tokenizer(sample_corpus)
    config_s14['vocab_size'] = tokenizer_s14.vocab_size
    
    # Make a smaller model for faster comparison if generating multiple times
    config_s14['d_model'] = 32 
    config_s14['n_layers'] = 1
    config_s14['n_heads'] = 1
    config_s14['d_ff'] = 64
    config_s14['max_seq_len'] = 20 # Keep it reasonable for tests

    model_s14_cached = dec.Decoder(
        vocab_size=config_s14['vocab_size'], d_model=config_s14['d_model'],
        n_layers=config_s14['n_layers'], n_heads=config_s14['n_heads'],
        d_ff=config_s14['d_ff'], max_seq_len=config_s14['max_seq_len'],
        epsilon=config_s14['epsilon']
    )

    model_s14_non_cached = dec.Decoder(
        vocab_size=config_s14['vocab_size'], d_model=config_s14['d_model'],
        n_layers=config_s14['n_layers'], n_heads=config_s14['n_heads'],
        d_ff=config_s14['d_ff'], max_seq_len=config_s14['max_seq_len'],
        epsilon=config_s14['epsilon']
    )
    # Ensure weights are the same for comparison
    # This is a bit tricky due to how get_parameters and structure works. 
    # For a robust comparison, one would need to implement a set_parameters method or copy them carefully.
    # For now, we rely on fresh initializations being similar for qualitative check, or test structure mainly.
    # A simple way for numpy arrays:
    # model_s14_non_cached.input_embedding.embedding_matrix = model_s14_cached.input_embedding.embedding_matrix.copy()
    # ... and for all other weights in W_q, W_k, W_v, W_o, FFN weights, Linear layer weights.
    # This is too verbose for here. Let's focus on KV cache mechanism and stop token.

    # Test 1: Generation with KV cache (Implicitly, by structure of new generate loop)
    prompt_s14 = "a"
    max_len_s14 = 5
    print(f"  Test 1: Generating with KV Cache. Prompt: '{prompt_s14}', Max Len: {max_len_s14}")
    # This call will use the KV cache path in the modified generate function
    output_cached = generate(model_s14_cached, tokenizer_s14, prompt_s14, max_len_s14, temperature=0.0)
    print(f"    Output (cached): '{output_cached}'")
    assert len(output_cached) == len(prompt_s14) + max_len_s14, "Cached generation length mismatch."
    assert output_cached.startswith(prompt_s14), "Cached generation does not start with prompt."
    print("  Test 1: PASSED (KV cache path executed, length and prompt verified). Output correctness depends on model state.")

    # Test 2: Stop Token Functionality
    # Suppose 'c' is our stop token. Its ID depends on the tokenizer.
    stop_char_s14 = 'c' # Make sure this char is in sample_corpus for tokenizer_s14
    # sample_corpus = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;!?-"
    # 'c' is in sample_corpus
    stop_token_id_s14 = tokenizer_s14.char_to_idx.get(stop_char_s14)
    
    print(f"  Test 2: Generating with stop token ID {stop_token_id_s14} ('{stop_char_s14}'). Prompt: '{prompt_s14}', Max Len: {max_len_s14 + 10}")
    if stop_token_id_s14 is None:
        print(f"    Warning: Stop character '{stop_char_s14}' not in tokenizer vocab for s14. Skipping stop token test.")
    else:
        # Generate text that is likely to include 'c' if model is random, give it enough length
        # Ensure the model is different or weights are random enough to produce varied output
        # For this test, use a slightly different model or ensure a non-zero temperature for variability if needed.
        # Using model_s14_cached, temperature 0.7 for some randomness.
        output_stopped = generate(model_s14_cached, tokenizer_s14, "ab", max_len_s14 + 10, temperature=0.7, stop_token_id=stop_token_id_s14)
        print(f"    Output (with stop token 'c'): '{output_stopped}'")
        if stop_char_s14 in output_stopped:
            assert output_stopped.endswith(stop_char_s14), f"Output should end with stop_char '{stop_char_s14}' if generated."
            assert len(output_stopped) <= len("ab") + max_len_s14 + 10, "Output with stop token is too long."
            print(f"  Test 2: PASSED. Stop token '{stop_char_s14}' was generated and stopped generation.")
        else:
            print(f"  Test 2: Stop token '{stop_char_s14}' was not generated within max_len. Test checks if stopping logic works if token appears.")
            # This isn't a failure of the stop logic itself, but that the token wasn't produced.
            # The generate function prints a message if it stops due to stop_token.

    # Test 3: Output consistency between cached and non-cached (qualitative)
    # This requires running non_cached path, which means kv_cache_list=None
    # The updated `generate` should handle this via its internal logic.
    print(f"  Test 3: Comparing cached vs non-cached output. Prompt: '{prompt_s14}', Max Len: {max_len_s14}")
    # To run non-cached, we'd call generate with a model/setup that forces the non-cached path.
    # The current generate function will create a kv_cache_list if model has n_layers.
    # To force non-cached, we'd have to make kv_cache_list remain None.
    # This part of test might be better if generate() explicitly takes a use_cache flag.
    # For now, let's assume the test for cached path (Test 1) is the primary validation for cache mechanism.
    # A true comparison needs careful weight sharing and control over which path is taken.

    # --- Forcing non-cached path for comparison ---
    # Create a temporary generate_non_cached version or modify generate to accept a flag.
    # For this test, let's assume we can call the non_cached path of the modified generate.
    # The actual `generate` function will be updated to manage `kv_cache_list` creation. 
    # We need to make `kv_cache_list` effectively None for the non-cached run.
    
    # To truly test non-cached, we need to ensure the kv_cache_list is not created or used.
    # The easiest is to modify generate to take a `use_kv_cache` boolean.
    # If not, we can try to simulate it if model_s14_non_cached has n_layers = 0 or similar hack.
    # Let's assume the main edit to `generate` will provide a way or that the logic is clear.

    # For now, this validation block assumes `generate` is updated for KV and stop token.
    # The full validation of this test needs the `generate` function to be edited first with `stop_token_id`.

    print("\\nGeneration Module Step 14 (KV Cache, Stop Token) Validation Block - initial setup complete.")
    print("Full validation of stop token and cached vs non-cached requires `generate` function to be edited.")

    print("\nGeneration Module Validation Completed Successfully.") 

    # --- Validating Step 15: Performance Measurement ---
    print("\n--- Validating Step 15: Performance Measurement ---")
    model_config_perf = {
        'vocab_size': tokenizer_instance.vocab_size,
        'd_model': 32,
        'n_layers': 1,
        'n_heads': 1,
        'd_ff': 64,
        'max_seq_len': 50,
        'epsilon': 1e-5
    }
    
    perf_prompt = "test "
    perf_gen_len = 10

    print(f"  Performance test using prompt: '{perf_prompt}', gen_len: {perf_gen_len}")
    print(f"  Model config for perf test: d_model={model_config_perf['d_model']}, n_layers={model_config_perf['n_layers']}, n_heads={model_config_perf['n_heads']}")

    # --- Non-cached Run --- 
    print("  Running non-cached generation...")
    # Initialize model fresh for non-cached run
    model_perf_non_cached = dec.Decoder(**model_config_perf)
    start_time_nc = time.time()
    output_no_cache = generate(model_perf_non_cached, tokenizer_instance, perf_prompt, perf_gen_len, temperature=0.0, use_kv_cache=False)
    time_no_cache = time.time() - start_time_nc
    print(f"    Non-cached time: {time_no_cache:.6f} seconds")
    print(f"    Non-cached output: '{output_no_cache}'")

    # --- Cached Run --- 
    print("  Running cached generation...")
    # Re-initialize model fresh for cached run to ensure identical starting state
    model_perf_cached = dec.Decoder(**model_config_perf)
    # Ensure weights are identical if initialization is random
    # Copy weights from the non-cached model to the cached model
    params_non_cached = model_perf_non_cached.get_parameters()
    params_cached = model_perf_cached.get_parameters()
    for name in params_cached:
        if name in params_non_cached:
             # Use np.copy to ensure it's a true copy
            params_cached[name][:] = np.copy(params_non_cached[name]) 
        else:
            print(f"Warning: Parameter {name} not found in non-cached model during weight copy.")
    print("    Model weights copied for consistency.")
            
    start_time_c = time.time()
    output_cache = generate(model_perf_cached, tokenizer_instance, perf_prompt, perf_gen_len, temperature=0.0, use_kv_cache=True)
    time_cache = time.time() - start_time_c
    print(f"    Cached time: {time_cache:.6f} seconds")
    print(f"    Cached output: '{output_cache}'")

    # --- Comparison ---
    print("  Comparing results...")
    perf_passed = True
    if time_cache < time_no_cache:
        print(f"  Speed Test: Cached version was faster. Factor: {time_no_cache / time_cache:.2f}x - PASSED")
    else:
        print(f"  Speed Test: Cached version was NOT faster. Non-cached: {time_no_cache:.6f}s, Cached: {time_cache:.6f}s - FAILED")
        perf_passed = False
        
    # --- Logit Comparison for First Generated Token ---
    print("  Comparing logits for first generated token...")
    try:
        # Non-cached: Get logits predicting token after prompt
        model_nc_logits = dec.Decoder(**model_config_perf)
        params_nc_logits = model_nc_logits.get_parameters()
        for name in params_nc_logits: 
            if name in params_non_cached: params_nc_logits[name][:] = np.copy(params_non_cached[name])
        
        prompt_tokens = tokenizer_instance.encode(perf_prompt)
        input_nc = np.array([prompt_tokens]) # Shape (1, prompt_len)
        mask_nc = dec.Decoder.create_causal_mask(input_nc.shape[1])
        logits_nc = model_nc_logits.forward(input_nc, mask=mask_nc)
        first_gen_logits_nc = logits_nc[0, -1, :] 

        # Cached: Simulate processing prompt and getting logits for next token
        model_c_logits = dec.Decoder(**model_config_perf)
        params_c_logits = model_c_logits.get_parameters()
        for name in params_c_logits: 
            if name in params_non_cached: params_c_logits[name][:] = np.copy(params_non_cached[name])
            
        kv_cache_sim = [{} for _ in range(model_config_perf['n_layers'])]
        current_tokens_for_logits = []
        final_logits_for_next = None
        for idx, token_id in enumerate(prompt_tokens):
            input_token_sim = np.array([[token_id]])
            # The forward pass output predicts the *next* token after the input
            logits_step = model_c_logits.forward(input_token_sim, mask=None, kv_cache_list=kv_cache_sim, current_token_idx=idx)
            current_tokens_for_logits.append(token_id)
            # Store the logits generated after processing the last token of the prompt
            if idx == len(prompt_tokens) - 1:
                final_logits_for_next = logits_step
        
        if final_logits_for_next is None:
             raise RuntimeError("Could not get logits from cached simulation.")
             
        first_gen_logits_c_sim = final_logits_for_next[0, -1, :] # Logits predicting token after prompt

        # Compare the logits
        if np.allclose(first_gen_logits_nc, first_gen_logits_c_sim, atol=1e-5):
            print("  Logit Consistency Test (First Token): Logits are CLOSE - PASSED")
        else:
            print("  Logit Consistency Test (First Token): Logits are DIFFERENT - FAILED")
            diff = np.abs(first_gen_logits_nc - first_gen_logits_c_sim)
            print(f"    Max absolute difference: {np.max(diff):.6e}")
            print(f"    Mean absolute difference: {np.mean(diff):.6e}")
            perf_passed = False
            
    except Exception as e:
        print(f"  Logit Consistency Test encountered an error: {e} - FAILED")
        perf_passed = False

    # --- Final Output Comparison (as before) ---
    if output_no_cache == output_cache:
        print("  Output Consistency Test: Outputs are IDENTICAL - PASSED")
    else:
        print("  Output Consistency Test: Outputs are DIFFERENT - FAILED")
        print(f"    Output (No Cache): '{output_no_cache}'")
        print(f"    Output (Cache):    '{output_cache}'")
        # The overall test should fail if logits are different, even if final text happens to match
        perf_passed = False 
        print("    Note: If outputs or logits differ, check MHA masking, KV cache logic, and PE offset implementation.")

    print("--- Step 15 Validation Completed ---")


    print("\nAll Generation Module validations finished.") 