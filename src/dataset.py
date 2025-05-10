"""Dataset loading and preprocessing utilities.

This module provides functions to load text data from files and preprocess it
using a given tokenizer. These are basic utilities intended for preparing data
for a character-level Transformer model.
"""
import numpy as np
from .tokenizer import Tokenizer

def load_text_dataset(file_path: str) -> str:
    """Loads a text dataset from a specified file path.

    Parameters
    ----------
    file_path : str
        The path to the text file to be loaded.

    Returns
    -------
    str
        The content of the file as a single string.

    Raises
    ------
    FileNotFoundError
        If the file at `file_path` does not exist.
    Exception
        For other I/O errors during file loading.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        return text_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except Exception as e:
        # It's often good practice to wrap the original exception or log it.
        raise Exception(f"Error loading dataset from {file_path}: {e}")

def preprocess_dataset(text_data: str, tokenizer_instance: Tokenizer) -> list[int]:
    """Preprocesses raw text data by encoding it with a tokenizer.

    Parameters
    ----------
    text_data : str
        The raw text data string.
    tokenizer_instance : Tokenizer
        An initialized `Tokenizer` instance to use for encoding the text.

    Returns
    -------
    list[int]
        A list of integer token IDs representing the encoded text.

    Raises
    -------
    TypeError
        If `tokenizer_instance` is not an instance of the `Tokenizer` class.
    """
    if not isinstance(tokenizer_instance, Tokenizer):
        raise TypeError("tokenizer must be an instance of Tokenizer")
    return tokenizer_instance.encode(text_data)

if __name__ == "__main__":
    import os
    # Ensure src is in path
    import sys
    current_dir = os.getcwd()
    project_root = current_dir
    if os.path.basename(current_dir) in ['src', 'tests', 'notebooks']:
        project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # The import 'from .tokenizer import Tokenizer' at the top level should work
    # when this script is run directly, provided Python treats 'src' as a package
    # or the path manipulation correctly allows it.
    # If running this __main__ block causes issues with the relative import,
    # it might need its own explicit 'from tokenizer import Tokenizer' here
    # *after* sys.path modification, for testing purposes of this script only.
    # However, for module usage, the top-level relative import is preferred.

    # Create a dummy tokenizer for testing
    sample_corpus_for_tokenizer = "abcdefghijklmnopqrstuvwxyz 0123456789"
    # For the __main__ block, if the relative import fails when running as a script,
    # we might need a local, absolute import here after path setup.
    # Let's assume for now the relative import is resolved or adjust if testing fails.
    try:
        # Attempt to use the top-level imported Tokenizer
        test_tokenizer_instance = Tokenizer(sample_corpus_for_tokenizer)
    except NameError: # Fallback for script execution if relative import isn't found directly
        from tokenizer import Tokenizer as ScriptTokenizer # Specific for __main__
        test_tokenizer_instance = ScriptTokenizer(sample_corpus_for_tokenizer)

    # Test 1: load_text_dataset
    print("--- Testing load_text_dataset ---")
    dummy_file_path = "test_dataset.txt"
    dummy_content = "This is a test dataset.\nIt has multiple lines.\nAnd some special characters like !@#$%^&*()."
    
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        f.write(dummy_content)

    try:
        loaded_content = load_text_dataset(dummy_file_path)
        assert loaded_content == dummy_content, f"Test 1.1 Failed: Content mismatch. Expected:\n{dummy_content}\nGot:\n{loaded_content}"
        print("Test 1.1 Passed: load_text_dataset loaded content correctly.")
        
        # Test with a non-existent file
        try:
            load_text_dataset("non_existent_file.txt")
            print("Test 1.2 Failed: FileNotFoundError not raised for non-existent file.")
        except FileNotFoundError:
            print("Test 1.2 Passed: FileNotFoundError raised correctly for non-existent file.")

    except Exception as e:
        print(f"Error during load_text_dataset tests: {e}")
    finally:
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    # Test 2: preprocess_dataset
    print("\n--- Testing preprocess_dataset ---")
    sample_text = "hello world 123"
    expected_tokens = [test_tokenizer_instance.char_to_idx.get(c) for c in sample_text if c in test_tokenizer_instance.char_to_idx]
    
    try:
        processed_tokens = preprocess_dataset(sample_text, test_tokenizer_instance)
        assert processed_tokens == expected_tokens, f"Test 2.1 Failed: Token mismatch. Expected: {expected_tokens}, Got: {processed_tokens}"
        print(f"Test 2.1 Passed: preprocess_dataset tokenized correctly. Tokens: {processed_tokens}")

        # Test with empty string
        empty_text = ""
        expected_empty_tokens = []
        processed_empty_tokens = preprocess_dataset(empty_text, test_tokenizer_instance)
        assert processed_empty_tokens == expected_empty_tokens, "Test 2.2 Failed: Empty string not processed correctly."
        print(f"Test 2.2 Passed: preprocess_dataset handled empty string. Tokens: {processed_empty_tokens}")

        # Test with text containing characters not in tokenizer's vocab
        text_with_unknown = "hello X Y Z world" # X, Y, Z are not in sample_corpus_for_tokenizer
        # Expected tokens should only contain known characters
        expected_unknown_tokens = [test_tokenizer_instance.char_to_idx.get(c) for c in "hello  world" if c in test_tokenizer_instance.char_to_idx] # two spaces
        # Need to manually construct based on tokenizer's known chars
        known_chars_in_text_with_unknown = "".join([c for c in text_with_unknown if c in test_tokenizer_instance.chars])
        expected_unknown_tokens_from_tokenizer = test_tokenizer_instance.encode(known_chars_in_text_with_unknown)

        processed_unknown_tokens = preprocess_dataset(text_with_unknown, test_tokenizer_instance)
        assert processed_unknown_tokens == expected_unknown_tokens_from_tokenizer, \
            f"Test 2.3 Failed: Text with unknown characters not processed correctly. Expected: {expected_unknown_tokens_from_tokenizer}, Got: {processed_unknown_tokens}"
        print(f"Test 2.3 Passed: preprocess_dataset handled unknown characters. Tokens: {processed_unknown_tokens}")

        # Test with invalid tokenizer type
        try:
            preprocess_dataset(sample_text, "not_a_tokenizer")
            print("Test 2.4 Failed: TypeError not raised for invalid tokenizer type.")
        except TypeError:
            print("Test 2.4 Passed: TypeError raised correctly for invalid tokenizer type.")

    except Exception as e:
        print(f"Error during preprocess_dataset tests: {e}")

    print("\nAll dataset.py tests completed.") 