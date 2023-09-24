from transformers import PreTrainedTokenizer
from typing import List, Tuple

def analyze_token_counts(dataset: List[Tuple[str, str]], tokenizer: PreTrainedTokenizer, min_tokens: int = 256) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Analyze token counts in the dataset.

    Args:
        dataset (List[Tuple[str, str]]): A list of pairs where the first element is a question and the second element is an answer.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text.
        min_tokens (int, optional): The minimum number of tokens required. Defaults to 256.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: A tuple containing two lists - problematic_rows and clean_rows.
            - problematic_rows: Pairs with either the question or answer having fewer than min_tokens tokens.
            - clean_rows: Pairs with both the question and answer having at least min_tokens tokens.
    """
    problematic_rows = [(question, answer) for question, answer in dataset if len(tokenizer.encode(question)) < min_tokens or len(tokenizer.encode(answer)) < min_tokens]
    clean_rows = [(question, answer) for question, answer in dataset if len(tokenizer.encode(question)) >= min_tokens and len(tokenizer.encode(answer)) >= min_tokens]
    
    return problematic_rows, clean_rows