import os
import sys
from typing import List
import json
import fire
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

# === S2P Methodology: Section 3.2 Structure-Aware Instruction Tuning ===
# 鏍规嵁璁烘枃 Figure 1 瀹氫箟鏍囧噯鎸囦护
S2P_INSTRUCTION = (
    "Given some paths and a triple from a knowledge graph. "
    "The path serves as the context information of the triple. "
    "Please determine the correctness of the triple and response True or False."
)


class Prompter(object):
    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        # 瀵瑰簲璁烘枃鍏紡 (5): X = I 鈯?Cpath 鈯?Tquery
        # 鍋囪 JSON 鏁版嵁闆嗕腑鐨?'input' 瀛楁宸茬粡鍖呭惈浜?Linearized Paths + Query Triple
        self.template = {
            "description": "Template used by S2P Framework.",
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "response_split": "### Response:"
        }

    def generate_prompt(self, instruction: str, input_ctx: str = None, label: str = None) -> str:
        # 濡傛灉鏁版嵁涓病鏈夎嚜甯?instruction锛屽垯浣跨敤 S2P 榛樿鎸囦护
        instr = instruction if instruction and len(instruction) > 0 else S2P_INSTRUCTION

        res = self.template["prompt_input"].format(instruction=instr, input=input_ctx)

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res


def train(
        # model/data params
        base_model: str = "",
        data_path: str = "data/CoDeX-S-s2p-train-balanced.json",  # Use balanced training file
        output_dir: str = "./s2p-codex-output",
        # --- Optimization: Keeping efficient training settings ---
        batch_size: int = 128,
        micro_batch_size: int = 16,  # 鏍规嵁鏄惧瓨璋冩暣
        num_epochs: int = 3,  # 閫氬父 KGC 寰皟闇€瑕?1-3 涓?Epoch
        learning_rate: float = 5e-4,  # --- Paper Sec 4.1.3: 鎸囧畾鍒濆 LR 涓?5e-4 ---
        cutoff_len: int = 512,  # Paper Sec 4.1.3: Max path length k=3, 512 token 瓒冲
        val_set_size: int = 0,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
        # llm hyperparams
        train_on_inputs: bool = True,
        add_eos_token: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,
        max_train_samples: int = -1,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training S2P (Structure-Guided Path Pruning) model:\n"
            f"base_model: {base_model}\n"
            f"learning_rate: {learning_rate} (S2P Paper Default)\n"
            f"batch_size: {batch_size}\n"
        )

    assert base_model, "Please specify a --base_model"

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # 浣跨敤 4-bit QLoRA 閰嶇疆
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        # 杩欓噷鐨?input 搴旇鍖呭惈: Path Context + Query Triple (鍏紡 5)
        full_prompt = prompter.generate_prompt(
            data_point.get("instruction", ""),
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point.get("instruction", ""), data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if max_train_samples > 0 and len(data["train"]) > max_train_samples:
        data["train"] = data["train"].shuffle(seed=42).select(range(max_train_samples))

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,  # S2P uses 5e-4
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",  # Efficiency optimization
        eval_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=2,
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,
        report_to=None,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # Standard Peft model saving workarounds
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    print("Starting S2P instruction tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
