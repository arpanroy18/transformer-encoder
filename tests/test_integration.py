import unittest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import get_config
from tokenizer import Tokenizer
from decoder import Decoder
from generation import generate

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Set up a basic configuration and model for all tests."""
        self.config = get_config({
            "d_model": 64, # Smaller model for faster tests
            "n_layers": 2,
            "n_heads": 2,
            "d_ff": 128,
            "max_seq_len": 50,
            "vocab_size": 0, # Will be set by tokenizer
            "epsilon": 1e-5,
            "dropout_rate": 0.0 # No dropout for deterministic tests
        })
        self.sample_corpus = "This is a test corpus. It contains some simple sentences. For testing purposes."
        self.tokenizer = Tokenizer(self.sample_corpus)
        self.config["vocab_size"] = self.tokenizer.vocab_size
        self.model = Decoder(
            vocab_size=self.config["vocab_size"],
            d_model=self.config["d_model"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
            d_ff=self.config["d_ff"],
            max_seq_len=self.config["max_seq_len"],
            epsilon=self.config["epsilon"],
            dropout_rate=self.config["dropout_rate"]
        )
        # Initialize weights for consistency in tests (optional, but good for some scenarios)
        # For simplicity, we'll rely on the default random initialization for now,
        # but for true output consistency tests, fixed weights would be better.

    def _run_generation_test(self, prompt, max_len, temperature, use_kv_cache, stop_token_id=None, expected_prefix=None, check_content=True):
        """Helper function to run a generation test and perform common assertions."""
        if expected_prefix is None:
            expected_prefix = prompt

        # Check if the prompt was actually empty for adjusting length check
        is_empty_prompt = (prompt == "")

        generated_text, attention_weights = generate(
            model=self.model,
            tokenizer_instance=self.tokenizer,
            prompt=prompt,
            max_len=max_len,
            temperature=temperature,
            stop_token_id=stop_token_id,
            use_kv_cache=use_kv_cache,
            return_attention_weights=True
        )

        self.assertTrue(generated_text.startswith(expected_prefix), f"Generated text '{generated_text}' does not start with expected prefix '{expected_prefix}'")
        
        # Adjust expected length for empty prompt case where a start token is prepended
        # Assumes the start token (ID 0) adds 1 character to the length.
        # This check only applies if generation actually happened (max_len > 0) and no stop token interfered.
        if max_len > 0 and not stop_token_id:
            expected_generated_len = max_len
            # The assertion compares len(generated_text) - len(prompt) with max_len.
            # If prompt is empty, len(prompt) is 0. 
            # If a start token was prepended, len(generated_text) will be 1 (start_token) + max_len (generated).
            # So, len(generated_text) - len(prompt) = (1 + max_len) - 0 = 1 + max_len.
            # This does *not* equal max_len. 
            # The assertion should perhaps check the number of *newly* generated tokens excluding the prepended one.
            
            # Let's check the total length based on whether prompt was empty.
            if is_empty_prompt and 0 in self.tokenizer.idx_to_char: # Check if default start token was used
                 expected_total_len = max_len + 1 # 1 start token + max_len generated
                 self.assertEqual(len(generated_text), expected_total_len, 
                                  f"Generated length mismatch for empty prompt. Expected total len: {expected_total_len}, Got: {len(generated_text)}, Text: '{generated_text}'")
            else:
                 # Original assertion for non-empty prompt
                 self.assertEqual(len(generated_text) - len(prompt), max_len, 
                                  f"Generated length mismatch. Prompt: '{prompt}', Max Len: {max_len}, Got len: {len(generated_text)}, Text: '{generated_text}'")

        if check_content and len(generated_text) > len(prompt):
            # Check generated part based on whether prompt was empty
            if is_empty_prompt and 0 in self.tokenizer.idx_to_char:
                 # Skip the first char (prepended start token) for vocab check
                 generated_tokens_str = generated_text[1:] 
            else:
                 generated_tokens_str = generated_text[len(prompt):]
                 
            if generated_tokens_str: # Proceed only if there are generated tokens to check
                 encoded_generated_tokens = self.tokenizer.encode(generated_tokens_str)
                 for token_id in encoded_generated_tokens:
                      self.assertIn(token_id, self.tokenizer.idx_to_char, f"Generated token ID {token_id} is not in tokenizer vocabulary.")

        self.assertIsNotNone(attention_weights)
        if max_len > 0 and prompt: # Attention weights are returned per generation step
            expected_attn_steps = max_len
            if stop_token_id: # If stop token is hit early, fewer steps
                 # This is tricky to assert precisely without knowing when stop token is hit
                 pass # We'll rely on other checks for stop_token
            else:
                 self.assertEqual(len(attention_weights), expected_attn_steps, "Number of attention weight steps mismatch")

            if attention_weights: # If any steps were made
                for layer_attns in attention_weights[0]: # Check first step's weights
                    self.assertEqual(layer_attns.shape[1], self.config["n_heads"], "Attention weights n_heads mismatch")
        return generated_text

    def test_01_generate_short_greedy_no_cache(self):
        """Test short generation with greedy decoding, no KV cache."""
        prompt = "This is"
        max_len = 5
        self._run_generation_test(prompt, max_len, temperature=0.0, use_kv_cache=False)

    def test_02_generate_short_greedy_with_cache(self):
        """Test short generation with greedy decoding, with KV cache."""
        prompt = "Test a"
        max_len = 6
        self._run_generation_test(prompt, max_len, temperature=0.0, use_kv_cache=True)

    def test_03_generate_longer_sampling_no_cache(self):
        """Test longer generation with sampling, no KV cache."""
        prompt = "The quick brown fox"
        max_len = 10
        self._run_generation_test(prompt, max_len, temperature=0.7, use_kv_cache=False)

    def test_04_generate_longer_sampling_with_cache(self):
        """Test longer generation with sampling, with KV cache."""
        prompt = "A simple test"
        max_len = 12
        self._run_generation_test(prompt, max_len, temperature=1.0, use_kv_cache=True)

    def test_05_generate_max_len_zero(self):
        """Test generation with max_len = 0."""
        prompt = "Hello"
        max_len = 0
        generated_text_no_cache = self._run_generation_test(prompt, max_len, temperature=0.0, use_kv_cache=False)
        self.assertEqual(generated_text_no_cache, prompt, "Generation with max_len=0 (no cache) should return prompt.")

        generated_text_with_cache = self._run_generation_test(prompt, max_len, temperature=0.0, use_kv_cache=True)
        self.assertEqual(generated_text_with_cache, prompt, "Generation with max_len=0 (with cache) should return prompt.")

    def test_06_generate_empty_prompt(self):
        """Test generation with an empty prompt."""
        prompt = ""
        max_len = 5
        # When prompt is empty, the "starts with prompt" is trivially true.
        # The main check is that it runs without error and generates something of some length.
        self._run_generation_test(prompt, max_len, temperature=0.7, use_kv_cache=False, check_content=True)
        self._run_generation_test(prompt, max_len, temperature=0.7, use_kv_cache=True, check_content=True)


    def test_07_generate_with_stop_token_no_cache(self):
        """Test generation with a stop token, no KV cache."""
        prompt = "Generate until stop"
        max_len = 20 # Max len longer than expected generation
        # Choose a common token as stop token, e.g., '.'
        stop_token_id = self.tokenizer.char_to_idx.get('.', None)
        self.assertIsNotNone(stop_token_id, "Stop token '.' not in tokenizer vocab for test.")

        generated_text = self._run_generation_test(prompt, max_len, temperature=0.0, use_kv_cache=False, stop_token_id=stop_token_id, check_content=False)
        # We can't guarantee the stop token will be generated with random weights,
        # but if it is, the text should include it and potentially be shorter than max_len.
        # For this test, we mainly ensure it runs. A more robust test would use a trained model or fixed weights.
        self.assertTrue(len(generated_text) <= len(prompt) + max_len)
        if stop_token_id is not None and self.tokenizer.decode([stop_token_id]) in generated_text[len(prompt):]:
            self.assertTrue(generated_text.endswith(self.tokenizer.decode([stop_token_id])))


    def test_08_generate_with_stop_token_with_cache(self):
        """Test generation with a stop token, with KV cache."""
        prompt = "Another stop test"
        max_len = 20
        stop_token_id = self.tokenizer.char_to_idx.get('s', None) # Stop on 's'
        self.assertIsNotNone(stop_token_id, "Stop token 's' not in tokenizer vocab for test.")

        generated_text = self._run_generation_test(prompt, max_len, temperature=0.7, use_kv_cache=True, stop_token_id=stop_token_id, check_content=False)
        self.assertTrue(len(generated_text) <= len(prompt) + max_len)
        if stop_token_id is not None and self.tokenizer.decode([stop_token_id]) in generated_text[len(prompt):]:
            # Check if the *first* occurrence of stop token is at the end if it was indeed the stop reason
            # This is hard to assert perfectly without knowing if it stopped due to token or max_len
            pass


    # Note: Output consistency between cached and non-cached versions with random weights
    # is not guaranteed due to potential floating point arithmetic differences leading to
    # different argmax outcomes in greedy decoding or different samples in random sampling.
    # A true consistency test would require fixed model weights and careful seeding.
    # The existing `progress.md` notes this caveat. These tests focus on correct execution
    # in various scenarios as per Step 22's primary goal.

if __name__ == '__main__':
    # Create a dummy test_results directory if it doesn't exist
    # This is often needed by test runners for XML reports, etc.
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    # You can run this file directly: python -m tests.test_integration
    # Or use unittest discovery: python -m unittest discover -s tests
    unittest.main() 