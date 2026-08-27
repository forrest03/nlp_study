"""Greedy exact-match evaluation on the official GSM8K test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .data import load_test_split
from .parsing import answers_equal, extract_answer, normalize_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a base model or GRPO LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--adapter", help="Path to a saved PEFT/LoRA adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all 1319 test rows")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_model_and_tokenizer(
    model_name: str, adapter: str | None, load_in_4bit: bool
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_source = adapter or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def _batched(rows: list[dict[str, Any]], batch_size: int) -> Any:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = load_test_split(limit=args.limit, seed=args.seed)
    rows = [dict(row) for row in dataset]
    model, tokenizer = _load_model_and_tokenizer(args.model, args.adapter, args.load_in_4bit)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in _batched(rows, args.batch_size):
            rendered = [
                tokenizer.apply_chat_template(
                    row["prompt"], tokenize=False, add_generation_prompt=True
                )
                for row in batch
            ]
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = inputs["input_ids"].shape[1]
            texts = tokenizer.batch_decode(output_ids[:, prompt_width:], skip_special_tokens=True)

            for row, text in zip(batch, texts):
                extracted = extract_answer(text)
                predictions.append(
                    {
                        "row_id": row["row_id"],
                        "question": row["question"],
                        "reference": normalize_answer(row["ground_truth"]),
                        "extracted_answer": (
                            normalize_answer(extracted) if extracted is not None else None
                        ),
                        "correct": answers_equal(extracted, row["ground_truth"]),
                        "has_boxed_answer": "\\boxed{" in text.replace(" ", ""),
                        "completion_chars": len(text),
                        "completion": text,
                    }
                )

    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    total = len(predictions)
    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "load_in_4bit": args.load_in_4bit,
        "test_examples": total,
        "exact_match": sum(item["correct"] for item in predictions) / total,
        "boxed_answer_rate": sum(item["has_boxed_answer"] for item in predictions) / total,
        "invalid_answer_rate": (
            sum(item["extracted_answer"] is None for item in predictions) / total
        ),
        "mean_completion_chars": mean(item["completion_chars"] for item in predictions),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    metrics = evaluate(build_parser().parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
