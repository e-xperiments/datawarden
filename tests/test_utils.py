from transformers import AutoTokenizer
from datawarden.utils import analyze_token_counts

def test_analyze_token_counts():
    # Create a sample dataset and tokenizer
    dataset = [
        ("How are you?", "I'm good."),  # Sample question and answer pairs
        ("What is Python?", "Python is a programming language."),
        ("Short Q?", "Short A."),
        ("Short Question", "A."),
        ("Long Question " + "a" * 255, "A."),  # Long question with 255 characters
        ("Long Answer", "A" * 255),  # Long answer with 255 characters
    ]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Test with min_tokens = 5
    problematic_rows, clean_rows = analyze_token_counts(dataset, tokenizer, min_tokens=5)
    assert len(problematic_rows) == 3  # Expecting 3 problematic rows (less than 5 tokens)
    assert len(clean_rows) == 3  # Expecting 3 clean rows (at least 5 tokens)

    # Test with min_tokens = 256
    problematic_rows, clean_rows = analyze_token_counts(dataset, tokenizer, min_tokens=256)
    assert len(problematic_rows) == 6  # Expecting 6 problematic rows (less than 256 tokens)
    assert len(clean_rows) == 0  # Expecting 0 clean rows (none with at least 256 tokens)

    # Test with an empty dataset
    empty_dataset = []
    problematic_rows, clean_rows = analyze_token_counts(empty_dataset, tokenizer, min_tokens=5)
    assert len(problematic_rows) == 0  # Expecting 0 problematic rows (empty dataset)
    assert len(clean_rows) == 0  # Expecting 0 clean rows (empty dataset)

    # Test with a dataset containing empty strings
    dataset_with_empty_strings = [("", ""), ("", "Answer"), ("Question", "")]
    problematic_rows, clean_rows = analyze_token_counts(dataset_with_empty_strings, tokenizer, min_tokens=5)
    assert len(problematic_rows) == 3  # Expecting 3 problematic rows (all less than 5 tokens)
    assert len(clean_rows) == 0  # Expecting 0 clean rows (none with at least 5 tokens)

if __name__ == '__main__':
    test_analyze_token_counts()
