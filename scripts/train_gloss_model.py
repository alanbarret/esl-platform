"""
AraT5 Fine-Tuning Script for ESL Gloss Generation
===================================================
Fine-tunes AraT5v2 (or mT5) on Arabic/English → ESL gloss pairs.

Dataset format (JSONL):
  {"input": "مرحبا كيف حالك", "output": "HELLO YOU HOW", "lang": "ar"}
  {"input": "How are you?", "output": "YOU HOW", "lang": "en"}

Usage:
  python scripts/train_gloss_model.py \
    --dataset data/processed/esl_gloss.jsonl \
    --output data/models/gloss-finetuned \
    --epochs 10 \
    --batch-size 8
"""
from __future__ import annotations

import json
import argparse
import asyncio
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset as HFDataset
import numpy as np


# ── Dataset ────────────────────────────────────────────────────────────────────

class GlossDataset:
    """Loads and tokenizes Arabic/English → ESL gloss training pairs."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_in = max_input_length
        self.max_out = max_target_length
        self.data = self._load(jsonl_path)

    def _load(self, path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _format_prompt(self, record: dict) -> str:
        lang = record.get("lang", "ar")
        text = record["input"]
        if lang == "ar":
            return f"translate Arabic to ESL gloss: {text}"
        return f"translate English to ESL gloss: {text}"

    def to_hf_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset with tokenized inputs."""
        prompts = [self._format_prompt(r) for r in self.data]
        targets = [r["output"] for r in self.data]

        model_inputs = self.tokenizer(
            prompts,
            max_length=self.max_in,
            truncation=True,
            padding="max_length",
        )
        labels = self.tokenizer(
            targets,
            max_length=self.max_out,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return HFDataset.from_dict(model_inputs)


# ── Training ───────────────────────────────────────────────────────────────────

async def train(
    dataset_path: str,
    output_dir: str = "data/models/gloss-finetuned",
    model_name: str = "UBC-NLP/AraT5v2-base-1024",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    eval_split: float = 0.1,
) -> None:
    """Fine-tune AraT5 on ESL gloss generation task."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Loading dataset: {dataset_path}")
    gloss_ds = GlossDataset(dataset_path, tokenizer)
    hf_ds = gloss_ds.to_hf_dataset()

    # Train/eval split
    split = hf_ds.train_test_split(test_size=eval_split, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=3,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune AraT5 for ESL gloss generation")
    parser.add_argument("--dataset", required=True, help="Path to JSONL training data")
    parser.add_argument("--output", default="data/models/gloss-finetuned")
    parser.add_argument("--model", default="UBC-NLP/AraT5v2-base-1024")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    asyncio.run(train(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    ))
