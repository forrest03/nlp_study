"""Command-line entry point for LoRA-based GRPO training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .data import load_training_splits
from .rewards import correctness_reward, format_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a math model with GRPO on GSM8K")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-math-1.5b-grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses all training rows")
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--validation-limit", type=int, default=64)

    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--correctness-weight", type=float, default=1.0)
    parser.add_argument("--format-weight", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--report-to", default="none", help="For example: none, tensorboard, wandb")

    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes NF4 QLoRA; install the qlora optional dependency",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true", help="Only useful for tiny smoke tests")
    return parser


def _precision_settings(name: str, torch: Any) -> tuple[bool, bool, Any]:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            name = "bf16"
        elif torch.cuda.is_available():
            name = "fp16"
        else:
            name = "fp32"
    if name == "bf16":
        return True, False, torch.bfloat16
    if name == "fp16":
        return False, True, torch.float16
    return False, False, torch.float32


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    effective_batch = world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "num_generations must divide WORLD_SIZE * per_device_batch_size * "
            f"gradient_accumulation_steps; got {args.num_generations} and {effective_batch}"
        )
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.correctness_weight < 0 or args.format_weight < 0:
        raise ValueError("reward weights cannot be negative")
    if args.correctness_weight == 0 and args.format_weight == 0:
        raise ValueError("at least one reward weight must be positive")
    if args.load_in_4bit and args.no_lora:
        raise ValueError("4-bit training requires LoRA; remove --no-lora")


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from peft import LoraConfig, TaskType
    from transformers import BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. GRPO generation/training is impractical on CPU. "
            "Use --allow-cpu only with a tiny model for a smoke test."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_args(args, world_size)
    bf16, fp16, torch_dtype = _precision_settings(args.precision, torch)

    train_dataset, validation_dataset = load_training_splits(
        validation_size=args.validation_size,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "arguments": vars(args),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "world_size": world_size,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        model_init_kwargs={"torch_dtype": torch_dtype},
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        reward_weights=[args.correctness_weight, args.format_weight],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        use_cpu=not torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
