import pytest
from transformers import AutoTokenizer
from datawarden.utils import analyze_token_counts

# Sample dataset for testing
TEST_DATASET = [
    (["How", "are", "you?"], ["I'm", "fine.", "Thanks!"]),
    (["What", "is", "your", "name?"], ["My", "name", "is", "ChatGPT."]),
    (["Hello"], ["Hi"]),
    (
        ["This", "is", "a", "long", "question", "with", "a", "long", "answer."],
        ["Yes,", "indeed.", "This", "is", "a", "very", "very", "long", "answer."],
    ),
]


@pytest.fixture
def tokenizer():
    """Fixture to initialize the tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def test_analyze_token_counts(tokenizer):
    # Test with default min_tokens_question and min_tokens_answer
    problematic_rows, clean_rows, problematic_indexes, clean_indexes = analyze_token_counts(TEST_DATASET, tokenizer)
    
    assert len(problematic_rows) == 4
    assert len(clean_rows) == 0
    assert problematic_indexes == [0, 1, 2, 3]
    assert clean_indexes == []

    # Test with custom min_tokens_question and min_tokens_answer
    problematic_rows, clean_rows, problematic_indexes, clean_indexes = analyze_token_counts(TEST_DATASET, tokenizer, min_tokens_question=5, min_tokens_answer=5)
    
    assert len(problematic_rows) == 1
    assert len(clean_rows) == 3
    assert problematic_indexes == [2]
    assert clean_indexes == [0, 1, 3]

    # Test with an empty dataset
    empty_dataset = []
    problematic_rows, clean_rows, problematic_indexes, clean_indexes = analyze_token_counts(empty_dataset, tokenizer)
    
    assert len(problematic_rows) == 0
    assert len(clean_rows) == 0
    assert problematic_indexes == []
    assert clean_indexes == []

if __name__ == "__main__":
    pytest.main()
