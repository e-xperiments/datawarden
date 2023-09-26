from typing import List, Tuple
from transformers import PreTrainedTokenizer

def analyze_token_counts(dataset: List[Tuple[List[str], List[str]]], tokenizer: PreTrainedTokenizer, min_tokens_question: int = 256, min_tokens_answer: int = 256) -> Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], List[int], List[int]]:
    """
    Analyze token counts in the dataset.

    Args:
        dataset (List[Tuple[List[str], List[str]]]): A list of pairs where the first element is a list of tokens for the question,
            and the second element is a list of tokens for the answer.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text.
        min_tokens_question (int, optional): The minimum number of tokens required for questions. Defaults to 256.
        min_tokens_answer (int, optional): The minimum number of tokens required for answers. Defaults to 256.

    Returns:
        Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], List[int], List[int]]: A tuple containing four lists -
            - problematic_rows: Pairs with either the question or answer having fewer than the specified min_tokens.
            - clean_rows: Pairs with both the question and answer having at least the specified min_tokens.
            - problematic_indexes: Indexes of problematic rows in the input dataset.
            - clean_indexes: Indexes of clean rows in the input dataset.
    """
    problematic_rows = [(question, answer) for question, answer in dataset if sum(len(tokenizer.encode(tokens)) for tokens in question) < min_tokens_question or sum(len(tokenizer.encode(tokens)) for tokens in answer) < min_tokens_answer]
    clean_rows = [(question, answer) for question, answer in dataset if sum(len(tokenizer.encode(tokens)) for tokens in question) >= min_tokens_question and sum(len(tokenizer.encode(tokens)) for tokens in answer) >= min_tokens_answer]
    
    problematic_indexes = [i for i, (question, answer) in enumerate(dataset) if (question, answer) in problematic_rows]
    clean_indexes = [i for i, (question, answer) in enumerate(dataset) if (question, answer) in clean_rows]
    
    return problematic_rows, clean_rows, problematic_indexes, clean_indexes
