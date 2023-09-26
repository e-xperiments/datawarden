from typing import List, Tuple
from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer
from datawarden.utils import analyze_token_counts

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
        self.train_questions, self.train_answers = [], []
        self.test_questions, self.test_answers = [], []

        self._process_dataset(self.dataset['train'])
        self._process_dataset(self.dataset['test'])

        self.train_data = [(self.train_questions[i], self.train_answers[i]) for i in range(len(self.train_questions))]
        self.test_data = [(self.test_questions[i], self.test_answers[i]) for i in range(len(self.test_questions))]

    def _process_dataset(self, dataset_subset: Dataset) -> None:
        """
        Process a dataset subset and extract questions and answers.

        Args:
            dataset_subset: A subset of the dataset to be processed.
        """
        questions, answers = [], []

        if 'instruction' in dataset_subset.features:
            # Extract 'instruction' and 'input' as questions, 'output' as answers
            for row in dataset_subset:
                question = f"{row['instruction']} {row['input']}"
                answer = row['output']
                questions.append([question])
                answers.append([answer])
        elif 'conversations' in dataset_subset.features:
            # Extract 'value' where 'from' is 'human' as questions, 'gpt' as answers
            for row in dataset_subset:
                conversation = row['conversations']
                human_messages = [msg['value'] for msg in conversation if msg['from'] == 'human']
                gpt_messages = [msg['value'] for msg in conversation if msg['from'] == 'gpt']
                questions.append(human_messages)
                answers.append(gpt_messages)
        elif 'text' in dataset_subset.features:
            # Extract text after 'Human:' as questions, after 'Assistant:' as answers
            for row in dataset_subset:
                text_segments = row['text'].split('### ')
                human_segments = [seg.split('Human: ')[-1].strip() for seg in text_segments if 'Human:' in seg]
                assistant_segments = [seg.split('Assistant: ')[-1] for seg in text_segments if 'Assistant:' in seg]
                questions.append(human_segments)
                answers.append(assistant_segments)

        if dataset_subset == self.dataset['train']:
            self.train_questions = questions
            self.train_answers = answers
        elif dataset_subset == self.dataset['test']:
            self.test_questions = questions
            self.test_answers = answers

    def get_token_counts(self, tokenizer: PreTrainedTokenizer, min_tokens_question: int = 256, min_tokens_answer: int = 256) -> Tuple[
            List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], List[int], List[int]]:
        """
        Analyze token counts in the train and test subsets of the dataset.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding text.
            min_tokens_question (int, optional): The minimum number of tokens required for questions. Defaults to 256.
            min_tokens_answer (int, optional): The minimum number of tokens required for answers. Defaults to 256.

        Returns:
            Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], List[int], List[int]]: A tuple containing four lists -
                - problematic_rows_train: Pairs in the train set with either the question or answer having fewer than the specified min_tokens.
                - clean_rows_train: Pairs in the train set with both the question and answer having at least the specified min_tokens.
                - problematic_indexes_train: Indexes of problematic rows in the train set.
                - clean_indexes_train: Indexes of clean rows in the train set.
                - problematic_rows_test: Pairs in the test set with either the question or answer having fewer than the specified min_tokens.
                - clean_rows_test: Pairs in the test set with both the question and answer having at least the specified min_tokens.
                - problematic_indexes_test: Indexes of problematic rows in the test set.
                - clean_indexes_test: Indexes of clean rows in the test set.
        """
        problematic_rows_train, clean_rows_train, problematic_indexes_train, clean_indexes_train = analyze_token_counts(
            self.train_data, tokenizer, min_tokens_question, min_tokens_answer)

        problematic_rows_test, clean_rows_test, problematic_indexes_test, clean_indexes_test = analyze_token_counts(
            self.test_data, tokenizer, min_tokens_question, min_tokens_answer)

        return (
            problematic_rows_train, clean_rows_train, problematic_indexes_train, clean_indexes_train,
            problematic_rows_test, clean_rows_test, problematic_indexes_test, clean_indexes_test
        )