import pytest
from datasets import Dataset, DatasetDict
from datawarden.dataset import DataWardenDataset
from transformers import AutoTokenizer

# Define sample data for testing
sample_data_instruction = {
    'train': Dataset.from_dict({
        'instruction': ['instr1', 'instr2'],
        'input': ['input1', 'input2'],
        'output': ['output1', 'output2']
    }),
    'test': Dataset.from_dict({
        'instruction': ['test_instr1', 'test_instr2'],
        'input': ['test_input1', 'test_input2'],
        'output': ['test_output1', 'test_output2']
    })
}

sample_data_conversations = {
    'train': Dataset.from_dict({
        'conversations': [
            [{'from': 'human', 'value': 'Hello'}, {'from': 'gpt', 'value': 'Hi'}],
            [{'from': 'human', 'value': 'How are you?'}, {'from': 'gpt', 'value': 'I am fine.'}],
        ]
    }),
    'test': Dataset.from_dict({
        'conversations': [
            [{'from': 'human', 'value': 'Good morning'}, {'from': 'gpt', 'value': 'Good morning!'}],
            [{'from': 'human', 'value': 'What is your name?'}, {'from': 'gpt', 'value': 'I am a chatbot.'}],
        ]
    })
}

sample_data_text = {
    'train': Dataset.from_dict({
        'text': [
            '### Human: How are you? ### Assistant: I am fine.',
            '### Human: What is your name? ### Assistant: I am a chatbot.',
        ]
    }),
    'test': Dataset.from_dict({
        'text': [
            '### Human: Good morning ### Assistant: Good morning!',
            '### Human: How can I help you? ### Assistant: You can ask me anything.',
        ]
    })
}

@pytest.fixture(params=[sample_data_instruction, sample_data_conversations, sample_data_text])
def sample_dataset(request):
    return DatasetDict(request.param)

@pytest.fixture(params=[sample_data_instruction, sample_data_conversations, sample_data_text])
def sample_dataset_with_tokens(request):
    dataset = DataWardenDataset(request.param)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your desired tokenizer
    return dataset, tokenizer

def test_init(sample_dataset):
    dataset = DataWardenDataset(sample_dataset)
    if 'instruction' in sample_dataset['train'].features:
        assert dataset.train_questions == [['instr1 input1'], ['instr2 input2']]
        assert dataset.train_answers == [['output1'], ['output2']]
        assert dataset.test_questions == [['test_instr1 test_input1'], ['test_instr2 test_input2']]
        assert dataset.test_answers == [['test_output1'], ['test_output2']]
    elif 'conversations' in sample_dataset['train'].features:
        assert dataset.train_questions == [['Hello'], ['How are you?']] 
        assert dataset.train_answers == [['Hi'], ['I am fine.']]
        assert dataset.test_questions == [['Good morning'], ['What is your name?']]
        assert dataset.test_answers == [['Good morning!'], ['I am a chatbot.']]
    elif 'text' in sample_dataset['train'].features:
        assert dataset.train_questions == [['How are you?'], ['What is your name?']]
        assert dataset.train_answers == [['I am fine.'], ['I am a chatbot.']]
        assert dataset.test_questions == [['Good morning'], ['How can I help you?']]
        assert dataset.test_answers == [['Good morning!'], ['You can ask me anything.']]

def test_get_token_counts(sample_dataset_with_tokens):
    dataset, tokenizer = sample_dataset_with_tokens
    min_tokens_question = 5
    min_tokens_answer = 5

    (
        problematic_rows_train,
        clean_rows_train,
        problematic_indexes_train,
        clean_indexes_train,
        problematic_rows_test,
        clean_rows_test,
        problematic_indexes_test,
        clean_indexes_test,
    ) = dataset.get_token_counts(tokenizer, min_tokens_question, min_tokens_answer)

    if sample_dataset_with_tokens[0].dataset_type == 'alpaca':
        assert problematic_rows_train == [(['instr1 input1'], ['output1']), (['instr2 input2'], ['output2'])] 
        assert clean_rows_train == []
        assert problematic_indexes_train == [0, 1]
        assert clean_indexes_train == []
        assert problematic_rows_test == []
        assert clean_rows_test == [(['test_instr1 test_input1'], ['test_output1']), (['test_instr2 test_input2'], ['test_output2'])]
        assert problematic_indexes_test == []
        assert clean_indexes_test == [0, 1]
    elif sample_dataset_with_tokens[0].dataset_type == 'sharegpt':
        assert problematic_rows_train == [(['Hello'], ['Hi'])]
        assert clean_rows_train == [(['How are you?'], ['I am fine.'])]
        assert problematic_indexes_train == [0]
        assert clean_indexes_train == [1]
        assert problematic_rows_test == [(['Good morning'], ['Good morning!'])]
        assert clean_rows_test == [(['What is your name?'], ['I am a chatbot.'])]
        assert problematic_indexes_test == [0]
        assert clean_indexes_test == [1]
    elif sample_dataset_with_tokens[0].dataset_type == 'raw':
        assert problematic_rows_train == []
        assert clean_rows_train == [(['How are you?'], ['I am fine.']), (['What is your name?'], ['I am a chatbot.'])]
        assert problematic_indexes_train == []
        assert clean_indexes_train == [0, 1]
        assert problematic_rows_test == [(['Good morning'], ['Good morning!'])]
        assert clean_rows_test == [(['How can I help you?'], ['You can ask me anything.'])]
        assert problematic_indexes_test == [0]
        assert clean_indexes_test == [1]

def test_remove_problematic_rows(sample_dataset_with_tokens):
    dataset, tokenizer = sample_dataset_with_tokens
    min_tokens_question = 5
    min_tokens_answer = 5

    # Calculate token counts before removing problematic rows
    (
        problematic_rows_train_before,
        clean_rows_train_before,
        problematic_indexes_train_before,
        clean_indexes_train_before,
        problematic_rows_test_before,
        clean_rows_test_before,
        problematic_indexes_test_before,
        clean_indexes_test_before,
    ) = dataset.get_token_counts(tokenizer, min_tokens_question, min_tokens_answer)

    # Remove problematic rows
    dataset.remove_problematic_rows(tokenizer, min_tokens_question, min_tokens_answer)

    # Calculate token counts after removing problematic rows
    (
        problematic_rows_train_after,
        clean_rows_train_after,
        problematic_indexes_train_after,
        clean_indexes_train_after,
        problematic_rows_test_after,
        clean_rows_test_after,
        problematic_indexes_test_after,
        clean_indexes_test_after,
    ) = dataset.get_token_counts(tokenizer, min_tokens_question, min_tokens_answer)

    # Ensure that the number of clean rows has increased
    assert dataset.train_data == clean_rows_train_after
    assert dataset.test_data == clean_rows_test_after

    # Ensure that no problematic rows are left
    assert len(problematic_rows_train_after) == 0
    assert len(problematic_rows_test_after) == 0

if __name__ == "__main__":
    pytest.main()
