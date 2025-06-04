from datasets import Dataset

from pipeline import prepare_dataset


def test_load_qa_pairs(mock_qa_path):
    data = prepare_dataset.load_qa_pairs(mock_qa_path)
    assert isinstance(data, list)
    assert len(data) > 0
    assert isinstance(data[0], dict)
    assert "question" in data[0] and "answer" in data[0]


def test_format_for_instruction_tuning(mock_qa_path):
    qa_data = prepare_dataset.load_qa_pairs(mock_qa_path)
    formatted = prepare_dataset.format_for_instruction_tuning(qa_data)
    assert isinstance(formatted, Dataset)
    sample = formatted[0]
    assert "instruction" in sample and "input" in sample and "output" in sample


def test_tokenize_dataset(mock_qa_path, tokenizer):
    qa_data = prepare_dataset.load_qa_pairs(mock_qa_path)
    formatted = prepare_dataset.format_for_instruction_tuning(qa_data)
    tokenized = prepare_dataset.tokenize_dataset(formatted, tokenizer, max_length=64)
    assert "input_ids" in tokenized[0]
    assert "labels" in tokenized[0]
    assert len(tokenized[0]["input_ids"]) == 64


def test_split_and_tokenize_dataset(mock_qa_path, tokenizer):
    qa_data = prepare_dataset.load_qa_pairs(mock_qa_path)
    formatted = prepare_dataset.format_for_instruction_tuning(qa_data)
    dataset_dict = prepare_dataset.split_and_tokenize_dataset(formatted, tokenizer, train_size=0.5)
    assert "train" in dataset_dict and "validation" in dataset_dict
    assert len(dataset_dict["train"]) > 0
    assert "input_ids" in dataset_dict["train"][0]