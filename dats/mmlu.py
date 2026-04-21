import torch
from .base import BaseDataset

class MMLUDataset(BaseDataset):
    CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def __init__(self, cfg, image_set, tokenizer):
        super().__init__(cfg, image_set, tokenizer)

    def _get_db(self):
        data = self._load_data(self.root_path)
        data = self._map(data)

        # 先用最简单方案：对官方 split 再切 train/val/test
        data = MMLUDataset.split_dataset(data["test"], seed=self.seed)

        if self.image_set == "train":
            data = data["train"]
        elif self.image_set == "val":
            data = data["validation"]
        elif self.image_set == "test":
            data = data["test"]
        else:
            raise ValueError(f"Unknown image_set: {self.image_set}")

        data = self._remove_column(data)
        return data

    def __getitem__(self, idx):
        example = self.db[idx]
        source = self.format_source(example["source"], self.model)
        target = self.format_target(example["target"], self.model)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        return {
            "source": source,
            "target": target,
            "labels": labels,
            "subject": example["subject"],
        }

    def _map(self, data):
        return data.map(self._format_input)

    def _format_input(self, example):
        def format_subject(subject):
            return subject.replace("_", " ").lower()

        prompt = (
            f'Below is a multiple-choice question about {format_subject(example["subject"])}. '
            f"Please choose the correct answer.\n"
        )
        prompt += example["question"] + "\nOptions:"
        for j in range(len(example["choices"])):
            prompt += "\n{}. {}".format(self.CHOICES[j], example["choices"][j])
        prompt += "\nAnswer: "

        example["source"] = prompt
        example["target"] = self.CHOICES[example["answer"]]
        example["labels"] = example["answer"]
        return example

    def _remove_column(self, data):
        save_column = ["source", "target", "labels", "subject"]
        columns_to_remove = [c for c in data.column_names if c not in save_column]
        return data.remove_columns(columns_to_remove)