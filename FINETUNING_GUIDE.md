# 🧠 Privox: Cantonese Fine-tuning Guide

This guide explains how to fine-tune a model like Llama 3.2 specifically for Cantonese and "Kongish" using modern, efficient tools.

## 1. The Strategy: QLoRA with Unsloth

Instead of training from scratch, we use **QLoRA (Quantized Low-Rank Adaptation)**. This allows you to "teach" a 3B parameter model new patterns using only ~8GB-12GB of VRAM.

### Recommended Tools

- **[Unsloth](https://github.com/unslothai/unsloth)**: The fastest and most memory-efficient library for fine-tuning Llama/Mistral on consumer GPUs.
- **[Google Colab](https://colab.research.google.com/)**: If you don't have a strong local GPU, you can use a free/cheap T4 or A100 instance.
- **[Hugging Face Autotrain](https://huggingface.co/autotrain)**: A "no-code" alternative if you prefer a GUI.

---

## 2. Data Format: The "Secret Sauce"

The quality of your data matters 100x more than the quantity. Use the **ShareGPT** or **Instruction** format.

### Recommended Format (JSONL)

Each line in your file should be a JSON object representing a "Turn":

```json
{
  "instruction": "你係一個粵語編輯助手，請執好呢句 Kongish，保留語法同口語感。",
  "input": "I go to eat rice later, you join?",
  "output": "我陣間去打冷食飯，你 join 唔 join？"
}
```

---

## 3. Step-by-Step Tutorial (The "Dry Run")

### Step A: Collect your Data

Save your corrections into `training_data.jsonl`. Aim for **500 to 2,000 high-quality pairs** to see a noticeable difference.

### Step B: Run the Training Script

I have provided a simplified Python snippet using Unsloth. You would run this in a Jupyter Notebook:

```python
from unsloth import FastLanguageModel
import torch

# 1. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Load your JSONL dataset and start SFT (Supervised Fine-Tuning)
# [Insert Unsloth Trainer Logic Here]
```

### Step C: Export to GGUF

After training, Unsloth allows you to save directly to GGUF:

```python
model.save_pretrained_gguf("my_cantonese_model", tokenizer, quantization_method = "q4_k_m")
```

---

## 4. Where to find more data?

- **Common Voice (Mozilla)**: Audio + Text pairs (useful for Whisper fine-tuning).
- **連登 (LIHKG) Scrapes**: Great for raw "Kongish" slang (requires cleaning).
- **Wikipedia (zh-yue)**: Formal Cantonese text.

## 5. Tips for Success

1. **Consistency**: Always use the same prompt style in training and in Privox's `config.json`.
2. **Punctuation**: Explicitly include samples where you fix full-width vs half-width punctuation.
3. **English Balance**: Include samples with varying levels of English mixing (Light, Heavy, Code-switching).

---

_Ready to start? I can provide a more detailed Python training script once you have your first 50 data points!_
