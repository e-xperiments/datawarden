from typing import List, Tuple
from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer
from datawarden.utils import analyze_token_counts, AnalysisResult

# Constants
TRAIN = "train"
TEST = "test"
INSTRUCTION = "instruction"
CONVERSATIONS = "conversations"
TEXT = "text"

class DataSource:
    HUMAN = "human"
    GPT = "gpt"

class DataWardenDataset:
    """
    A custom dataset class for processing various types of input data.
    """

    def __init__(self, dataset: DatasetDict) -> None:
        """
        Initialize the dataset.

        Args:
            dataset (DatasetDict): A dataset containing 'train' and 'test' subsets.
        """
        self.dataset = dataset

        self.train_data = self._process_dataset(self.dataset[TRAIN])
        self.test_data = self._process_dataset(self.dataset[TEST])

    def _process_dataset(self, dataset_subset: Dataset) -> List[Tuple[List[str], List[str]]]:
        """
        Process a dataset subset and extract questions and answers.

        Args:
            dataset_subset: A subset of the dataset to be processed.

        Returns:
            List of tuples containing questions and answers.
        """
        questions, answers = [], []

        if INSTRUCTION in dataset_subset.features:
            for row in dataset_subset:
                questions.append([f"{row[INSTRUCTION]} {row['input']}"])
                answers.append([row["output"]])

        elif CONVERSATIONS in dataset_subset.features:
            for row in dataset_subset:
                conversation = row[CONVERSATIONS]
                questions.append([msg["value"] for msg in conversation if msg["from"] == DataSource.HUMAN])
                answers.append([msg["value"] for msg in conversation if msg["from"] == DataSource.GPT])

        elif TEXT in dataset_subset.features:
            for row in dataset_subset:
                text_segments = row[TEXT].split("### ")
                questions.append([seg.split("Human: ")[-1].strip() for seg in text_segments if "Human:" in seg])
                answers.append([seg.split("Assistant: ")[-1] for seg in text_segments if "Assistant:" in seg])

        return list(zip(questions, answers))

    def get_token_counts(
        self,
        tokenizer: PreTrainedTokenizer,
        min_tokens_question: int = 256,
        min_tokens_answer: int = 256,
    ) -> Tuple[AnalysisResult, AnalysisResult]:
        """
        Analyze token counts in the train and test subsets of the dataset.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text.
            min_tokens_question (int, optional): The minimum number of tokens required for questions.
            min_tokens_answer (int, optional): The minimum number of tokens required for answers.

        Returns:
            Tuple[AnalysisResult, AnalysisResult]: Analysis results for the train and test data.
        """
        analysis_result_train = analyze_token_counts(
            self.train_data, tokenizer, min_tokens_question, min_tokens_answer
        )
        analysis_result_test = analyze_token_counts(
            self.test_data, tokenizer, min_tokens_question, min_tokens_answer
        )

        return analysis_result_train, analysis_result_test
