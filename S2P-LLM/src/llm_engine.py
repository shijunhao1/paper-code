import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LlamaEngine:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        print(f"Loading LLaMA-7B from {model_path}...")

        # 4-bit 量化配置 (显存优化)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",  # 自动分配显卡
            trust_remote_code=True
        )

        # LLaMA 默认没有 pad_token，通常将其设为 eos_token 或 unk_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, prompt, max_new_tokens=32):
        """
        执行推理
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # 低温度，保证结果确定性
                do_sample=False,  # 对于分类任务，通常关闭采样
                repetition_penalty=1.1
            )

        # 解码并去除原始 Prompt，只保留生成的回答
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 截取 Prompt 之后的部分
        # 注意：这里简单处理，直接返回全部文本交给后处理清洗也可以
        answer = generated_text[len(prompt):].strip()
        return answer