from typing import List, Tuple, Union
from transformers import PreTrainedTokenizer

# typehints
TokenList = List[str]
QuestionAnswerPair = Tuple[TokenList, TokenList]
Dataset = List[QuestionAnswerPair]


class AnalysisResult:
    def __init__(self):
        self.problematic_rows: Dataset = []
        self.clean_rows: Dataset = []
        self.problematic_indexes: List[int] = []
        self.clean_indexes: List[int] = []


def analyze_token_counts(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    min_tokens_question: int = 256,
    min_tokens_answer: int = 256,
) -> AnalysisResult:
    """
    Analyze token counts in the dataset.

    Args:
        dataset (Dataset): A list of pairs where the first element is a list of tokens for the question,
            and the second element is a list of tokens for the answer.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text.
        min_tokens_question (int, optional): The minimum number of tokens required for questions. Defaults to 256.
        min_tokens_answer (int, optional): The minimum number of tokens required for answers. Defaults to 256.

    Returns:
        AnalysisResult: An object encapsulating the results of the analysis.
    """
    result = AnalysisResult()

    for idx, (question, answer) in enumerate(dataset):
        question_tokens_count = sum(
            len(tokenizer.encode(tokens)) for tokens in question
        )
        answer_tokens_count = sum(len(tokenizer.encode(tokens)) for tokens in answer)

        if (
            question_tokens_count < min_tokens_question
            or answer_tokens_count < min_tokens_answer
        ):
            result.problematic_rows.append((question, answer))
            result.problematic_indexes.append(idx)
        else:
            result.clean_rows.append((question, answer))
            result.clean_indexes.append(idx)

    return result
