"""Character-level tokenizer for text processing.

This module provides a simple character-level tokenizer that builds a vocabulary
from a given corpus and can encode text into sequences of integer token IDs
and decode token ID sequences back into text.
"""

class Tokenizer:
    """A character-level tokenizer.

    Builds a vocabulary from an initial corpus and provides methods to convert
    text to token IDs and vice-versa. Characters not found in the initial
    corpus vocabulary are ignored during encoding and decoding.

    Attributes
    ----------
    chars : list of str
        Sorted list of unique characters forming the vocabulary.
    char_to_idx : dict
        Mapping from characters to their corresponding integer token IDs.
    idx_to_char : dict
        Mapping from integer token IDs to their corresponding characters.

    Parameters
    ----------
    corpus : str
        The text corpus from which to build the vocabulary.
    """
    def __init__(self, corpus: str):
        """Initializes the Tokenizer with a given corpus.

        Parameters
        ----------
        corpus : str
            The text corpus to build the vocabulary from.
        """
        self.chars = sorted(list(set(corpus)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Encodes a text string into a list of integer token IDs.

        Characters in the input text that are not present in the tokenizer's
        vocabulary (built from the initial corpus) are ignored.

        Parameters
        ----------
        text : str
            The text string to encode.

        Returns
        -------
        list of int
            A list of integer token IDs representing the input text.
        """
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of integer token IDs back into a text string.

        Token IDs in the input list that are not present in the tokenizer's
        vocabulary are ignored.

        Parameters
        ----------
        tokens : list of int
            The list of integer token IDs to decode.

        Returns
        -------
        str
            The decoded text string.
        """
        return "".join([self.idx_to_char[idx] for idx in tokens if idx in self.idx_to_char])

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary.

        The vocabulary size is the number of unique characters found in the
        initial corpus.

        Returns
        -------
        int
            The number of unique characters in the vocabulary.
        """
        return len(self.chars)

if __name__ == "__main__":
    # Sample corpus
    sample_corpus = "hello world! 123 hello? testing."

    # Initialize tokenizer
    tokenizer = Tokenizer(sample_corpus)

    # Test sentence
    test_sentence = "hello?"

    # Encode the sentence
    encoded_sentence = tokenizer.encode(test_sentence)

    # Decode the sentence
    decoded_sentence = tokenizer.decode(encoded_sentence)

    print(f"Sample Corpus: '{sample_corpus}'")
    print(f"Vocabulary: {''.join(tokenizer.chars)}")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Char to Idx mapping: {tokenizer.char_to_idx}")
    print(f"Idx to Char mapping: {tokenizer.idx_to_char}")
    print(f"Original sentence: '{test_sentence}'")
    print(f"Encoded sentence: {encoded_sentence}")
    print(f"Decoded sentence: '{decoded_sentence}'")

    # Validation
    if decoded_sentence == test_sentence:
        print("\nValidation successful: Decoded sentence matches the original.")
    else:
        print("\nValidation failed: Decoded sentence does not match the original.")

    # Test with characters not in the original corpus (should be ignored by encode/decode)
    test_sentence_unknown_chars = "hello there general kenobi"
    encoded_unknown = tokenizer.encode(test_sentence_unknown_chars)
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"\nOriginal sentence with unknown chars: '{test_sentence_unknown_chars}'")
    print(f"Encoded sentence (unknown chars ignored): {encoded_unknown}")
    print(f"Decoded sentence (unknown chars ignored): '{decoded_unknown}'")
    # The expected string should be the original string with unknown characters removed.
    # For corpus "hello world! 123 hello? testing."
    # and input "hello there general kenobi"
    # known characters are h,e,l,l,o, ,t,h,e,r,e, ,g,e,n,e,r,l, ,k,e,n,o,b,i
    # from input:        h,e,l,l,o, ,t,h,e,r,e, ,g,e,n,e,r,l, , ,e,n,o, ,i
    # (k, b are not in corpus, so they are removed)
    expected_unknown_decode = "".join([char for char in test_sentence_unknown_chars if char in tokenizer.chars])
    if decoded_unknown == expected_unknown_decode :
         print("Validation successful for unknown characters: Handled as expected.")
    else:
        print(f"Validation failed for unknown characters. Expected '{expected_unknown_decode}', got '{decoded_unknown}'") 