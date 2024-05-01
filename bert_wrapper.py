from typing import List, Union
from transformers import BertTokenizer

class SequenceTokenizer:
    def __init__(self, pretrained_model_name: str = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def tokenize(self, text: Union[str, List[str]]) -> List[List[int]]:
        """
        Tokenize a single text or a list of texts.
        Args:
            text (Union[str, List[str]]): The text or list of texts to be tokenized.
        Returns:
            List[List[int]]: A list of token ID sequences.
        """
        if isinstance(text, str):
            text = [text]

        token_id_sequences = []
        for t in text:
            tokens = self.tokenizer.tokenize(t)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_id_sequences.append(token_ids)

        return token_id_sequences

    def decode(self, token_id_sequence: List[int]) -> str:
        """
        Decode a sequence of token IDs back to a string.
        Args:
            token_id_sequence (List[int]): The sequence of token IDs to be decoded.
        Returns:
            str: The decoded string.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(token_id_sequence)
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text