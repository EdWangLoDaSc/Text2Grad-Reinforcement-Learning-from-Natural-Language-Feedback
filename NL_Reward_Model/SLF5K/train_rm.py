
import wandb
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import deepspeed
import json
from datasets import Dataset as HF_Dataset
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from transformers import GenerationConfig
from accelerate.utils import gather_object
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


S = '''You are a human annotator specializing in linguistics. Evaluate the generated summary against the original post using word-level scoring based on span-based quality assessment.

# Objective
- Critique the summary first, then assign scores to words based on identified spans.
- Focus on quality over quantity: identify concise, meaningful phrases (spans) rather than excessive breakdowns.

# Scoring Rules
- Score 1: Words in "good spans" (accurate, helpful phrases that capture key details from the original post).
- Score -1: Words in "poor spans" (inaccurate, redundant, or misleading phrases that detract from quality).
- Score 0: Neutral words (not part of good or poor spans).

# Evaluation Steps
1. Critique the summary:
   - Identify "good spans": concise phrases that accurately and helpfully reflect the original post's key points.
   - Identify "poor spans": concise phrases that are inaccurate, redundant, or misleading.
   - Keep spans meaningful and minimal; avoid over-segmentation.
2. Assign scores:
   - 1 for each word in good spans.
   - -1 for each word in poor spans.
   - 0 for all other words.

# Input Format
{
  "original_post": "Text of the original Reddit post.",
  "generated_summary": "Text of the model-generated summary."
}

# Output Format in json
{
  "textual_feedback": "Critique identifying good spans (accurate/helpful) and poor spans (inaccurate/problematic) in the summary.",
  "word_score_list": [
    ("word1", "Score (-1, 0, or 1)"),
    ("word2", "Score (-1, 0, or 1)"),
    ...
  ]
}

# Note
- Scores apply to words only, not punctuation.
- Directly output json and no other words
'''


deepspeed.ops.op_builder.CPUAdamBuilder().load()
########################### metric ################################
from bert_score import score
from rouge import Rouge
import jieba


class QACDataset(Dataset):
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
        with open(json_path, "r") as json_f:
            self.json_data = json.load(json_f)
        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.answer_max_length = max_length - prompt_max_length - 1
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.rules = S

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, ind):
        value = self.json_data[ind]

        # Clean up data extraction
        feedback = value.get("textual_feedback", "")
        word_score_list = value.get("word_score_list", [])
        first_summary = value.get("generated_summary", "")
        summary_prompt = value.get("post", "")

        # Ensure proper escaping of quotes in JSON strings
        summary_prompt = summary_prompt.replace('"', '\\"')
        first_summary = first_summary.replace('"', '\\"')
        feedback = feedback.replace('"', '\\"')

        # Format input and output properly
        question = f"""# User Input
{{
  "original_post": "{summary_prompt}",
  "generated_summary": "{first_summary}"
}}
"""
        # Fix JSON formatting in the answer
        answer = f'''
{{
  "textual_feedback": "{feedback}",
  "word_score_list": {json.dumps(word_score_list)}
}}'''

        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{self.rules}\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        src_tokens = self.tokenizer.tokenize(input_text)  # 不添加bos
        if len(src_tokens) > self.prompt_max_length:
            src_tokens = src_tokens[:self.prompt_max_length]

        tgt_tokens = self.tokenizer.tokenize(answer)
        if len(tgt_tokens) > self.answer_max_length:
            tgt_tokens = tgt_tokens[:self.answer_max_length]

        tokens = src_tokens + tgt_tokens + [self.eos_token]  # 添加eos

        assert len(tokens) <= self.max_length

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        context_length = len(src_tokens)
        labels = [-100] * context_length + input_ids[context_length:]

        assert len(input_ids) == len(labels)

        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len
        attetion_mask = [1] * (len(input_ids) - padding_len) + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attetion_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# 设置模型和数据
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
dataset_file = "/cosmos/hanyang_critic/RLAIF/ppo_token/rm_train/SLF5K_label/train_critique_processed.json"
valid_dataset_file = "/cosmos/hanyang_critic/RLAIF/ppo_token/rm_train/SLF5K_label/validation_critique_processed.json"
BATCH_SIZE = 1
EPOCHS = 3
prompt_max_length = 900
max_length = 1400
exp = "ckpt/llama31-8B-span-v2"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.pad_token_id = tokenizer.eos_token_id
# 使用lora mode进行训练
lora_config = LoraConfig(
    r=16,  # LoRA的秩，通常取8、16、32等，值越小越节省参数
    lora_alpha=32,  # 缩放因子，控制LoRA对原模型的贡献
    target_modules=["q_proj", "v_proj"],  # 应用LoRA的目标模块（这里是注意力层）
    lora_dropout=0.05,  # dropout率，防止过拟合
    bias="none",  # 不调整偏置
    task_type="CAUSAL_LM"  # 任务类型，因果语言建模
)
model = get_peft_model(model, lora_config)
print("LoRA layers have been added.")

# Deepspeed
ds_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,  # 初始学习率
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },

    "train_batch_size": 1 * BATCH_SIZE * torch.cuda.device_count(),
    "gradient_accumulation_steps": 1,
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_clipping": 1.0,
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters)}")
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=trainable_parameters,
    config=ds_config
)


# load data
train_dataset = QACDataset(dataset_file, tokenizer=tokenizer, prompt_max_length=prompt_max_length,
                           max_length=max_length)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = QACDataset(valid_dataset_file, tokenizer=tokenizer, prompt_max_length=prompt_max_length,
                           max_length=max_length)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# init wandb
project_name = "slf5k_llama_8b_cot-json"
wandb.init(project=project_name, name=project_name)
fail_case = 0
for epoch in range(EPOCHS):
    model_engine.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = batch["input_ids"].to(model_engine.local_rank)
        attention_mask = batch["attention_mask"].to(model_engine.local_rank)
        labels = batch["labels"].to(model_engine.local_rank)
        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        # Add learning rate logging
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
        })

        # evaluation
        if step % 300 == 0 and step != 0:
            torch.cuda.empty_cache()
            model_engine.eval()  # Set model to evaluation mode
            eval_losses = []

            # 创建一个固定大小的随机验证数据集
            eval_subset_size = 500
            total_eval_samples = len(valid_dataset)

            # 生成随机索引
            random_indices = torch.randperm(total_eval_samples)[:eval_subset_size]
            eval_subset = torch.utils.data.Subset(valid_dataset, random_indices)

            # 创建子集的dataloader
            eval_subset_loader = DataLoader(
                eval_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=valid_dataloader.num_workers,
                pin_memory=valid_dataloader.pin_memory
            )

            with torch.no_grad():
                table = defaultdict(list)
                for eval_step, batch in tqdm(enumerate(eval_subset_loader),
                                             total=len(eval_subset_loader),
                                             desc="Evaluating random subset"):
                    input_ids = batch["input_ids"].to(model_engine.local_rank)
                    attention_mask = batch["attention_mask"].to(model_engine.local_rank)
                    labels = batch["labels"].to(model_engine.local_rank)
                    outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    eval_loss = outputs.loss
                    eval_losses.append(eval_loss.item())

            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            wandb.log({
                "eval/loss": avg_eval_loss,
                "eval/samples": eval_subset_size
            })
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}, "
                  f"Eval Loss (on {eval_subset_size} samples): {avg_eval_loss:.4f}")

            model_engine.train()  # Set model back to training mode
            
        # Checkpoint saving with better error handling
        if step % 400 == 0 and step != 0:
            if model_engine.local_rank == 0:
                checkpoint_dir = os.path.join(exp, f'{epoch}_{step}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                try:
                    model_engine.save_pretrained(checkpoint_dir)
                    print(f"Successfully saved checkpoint to {checkpoint_dir}")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")