# 08 — Fine-tuning a real LLM on Colab (haiku style)

Code companion to [`../llm-training.md`](../llm-training.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zrrbite/agentic/blob/main/theory/code/08-sft-and-dpo.ipynb)

In notebook 06 we trained a tiny GPT from scratch on Tiny Shakespeare. That builds understanding, but a *useful* LLM needs (a) a serious pre-trained base model and (b) fine-tuning to behave the way you want.

This notebook does both. We take **SmolLM-360M-Instruct** (a real, instruction-tuned, 360M-param model from Hugging Face) and fine-tune it with **QLoRA + SFT** to answer questions in **haiku** — three short evocative lines.

You'll see clear before/after:
- **Before** ("What is the moon?") → a paragraph of factual prose
- **After** ("What is the moon?") → three short lines of imagery

Why haiku? It's a *visible*, easy-to-evaluate behavioural change. ~30 examples teach the model the format reliably without making it learn new facts. Same pipeline you'd use for tone, persona, formatting, anything stylistic.

> **⚠️ Colab only.** This notebook needs a GPU with ~6GB VRAM. Free Colab T4 (16GB) is plenty. CPU is too slow and 4-bit quantisation requires CUDA `bitsandbytes`. If you're reading this on github.com without running it, that's the intended use — outputs come from a Colab run.

## Setup (Colab)

1. Click **Open in Colab** above (or copy the URL)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

Total time: ~10–20 min on a T4.


```python
%pip install -q -U transformers peft trl bitsandbytes accelerate datasets
```


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

assert torch.cuda.is_available(), "GPU required — Runtime → Change runtime type → T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
```

## The base model

**SmolLM-360M-Instruct** ([HuggingFaceTB/SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct)) is a good demo base because:

- Real pre-trained transformer, ~360M params
- Already instruction-tuned (responds to chat-format questions, not just text completion)
- Apache 2.0 license — no auth needed to download
- Small enough to fine-tune fast on a T4 with 4-bit quantisation

We load it in **4-bit** (NF4 quantisation) so the base weights take ~250 MB of VRAM instead of ~1 GB. The LoRA adapters trained on top will be in bf16 — that's the "Q" in QLoRA.


```python
MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
print(f"Loaded. Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Test the base model *before* fine-tuning

Check that the model works and see how it answers our test prompts. Without fine-tuning it'll respond in regular prose.


```python
def chat(prompt, max_new=80):
    messages = [{"role": "user", "content": prompt}]
    # apply_chat_template returns a dict-like BatchEncoding in modern transformers;
    # use return_dict=True and unpack with ** so model.generate gets attention_mask too.
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

TEST_PROMPTS = [
    "What is the moon?",
    "Describe autumn.",
    "Tell me about silence.",
    "What is patience?",
]

print("=== BEFORE FINE-TUNING ===")
for prompt in TEST_PROMPTS:
    print()
    print(f">>> {prompt}")
    print(chat(prompt))
```

## The training data

~30 hand-written `(question, haiku)` pairs. In a real fine-tune you'd want hundreds to thousands; this is enough to teach the *format*. Quality > quantity for behavioural fine-tunes.


```python
HAIKU_DATASET = [
    ('Describe rain.'                , 'Soft drops on rooftops\nWashing the world clean again\nGray sky breathes silver'),
    ('What is winter?'               , 'Cold breath on the glass\nNaked branches reach for sky\nSilent earth sleeps deep'),
    ('Tell me about cats.'           , "Whiskers in moonlight\nA shadow with golden eyes\nNight's quiet hunter"),
    ('What is happiness?'            , "Sun on tired skin\nA friend's laugh across the room\nNo other plans now"),
    ('Describe an ocean.'            , 'Endless blue heartbeat\nSalt wind speaks ancient secrets\nWaves carry the moon'),
    ('What is autumn?'               , 'Leaves fall like soft sighs\nGolden light through bare branches\nThe world slows to rest'),
    ('Tell me about rivers.'         , 'Stones polished by time\nWater knows where it must go\nNo path is too long'),
    ('What is silence?'              , 'After the bell rings\nA breath held in the chapel\nMore words than a song'),
    ('Describe a forest.'            , 'Cathedral of pines\nMushrooms whisper to old roots\nMoss remembers all'),
    ('What is fear?'                 , 'Shadow without name\nThe heart counting its own beats\nWaiting for a sound'),
    ('Tell me about books.'          , 'Paper thoughts of ghosts\nHands across centuries meet\nWords outlive the bone'),
    ('What is the moon?'             , "Silver coin in dark\nPulling tides without touching\nEarth's silent twin sleeps"),
    ('Describe a city.'              , 'Lights against the rain\nEach window holds a story\nNo one looks at sky'),
    ('What is loneliness?'           , 'Empty cup at dawn\nFootsteps echo in long halls\nOne shadow on wall'),
    ('Tell me about fire.'           , 'Hungry orange tongue\nDevours wood, breathes out smoke\nGives warmth, takes our trees'),
    ('What is morning?'              , 'Light on closed eyelids\nThe world clears its sleeping throat\nKettles begin their hum'),
    ('Describe stars.'               , 'Frozen flames so far\nLight that left when we were small\nReaching us at last'),
    ('What is love?'                 , 'Hand finds hand in dark\nA name said with softer voice\nHome that is not place'),
    ('Tell me about wind.'           , 'Invisible hand\nMoves trees, flags, and old papers\nGoes where it pleases'),
    ('What is dreaming?'             , 'Mind unbound from clock\nDoors that open onto seas\nReason takes a nap'),
    ('Describe spring.'              , 'Green pushes through earth\nBirds remember their old songs\nColor returns home'),
    ('What is solitude?'             , 'Empty road at dusk\nOne footstep, then another\nI am company'),
    ('Tell me about mountains.'      , 'Stone made by patience\nWhat the rivers choose to leave\nClouds wear them like hats'),
    ('What is patience?'             , 'Stone shaped by water\nDrop by drop, year by long year\nNothing rushed remains'),
    ('Describe summer.'              , 'Sun thick as honey\nCicadas measure long days\nIce melts in the glass'),
    ('What is gratitude?'            , "Bread still warm at dawn\nRoof, hands, and a friend's old name\nNot given, but kept"),
    ('Tell me about snow.'           , 'White silences fall\nWorld becomes a quiet page\nFootprints ask questions'),
    ('What is memory?'               , 'Photograph of light\nThe room is gone, the face stays\nA scent remembers'),
    ('Describe a garden.'            , 'Ordered wilderness\nEach root negotiates space\nThe gardener listens'),
    ('What is courage?'              , 'Knees that still tremble\nA step taken anyway\nFear walks beside us'),
    ('Tell me about Spock.'          , 'Two hearts, calm logic\nEmotion beneath the surface\nThe needs of the many'),
    ('What are Vulcans?'             , 'Bones taught to be still\nOne hand raised: live long, prosper\nLogic is a song'),
    ('Describe the Enterprise.'      , 'Steel against starlight\nFive-year mission with no end\nThe crew is the ship'),
    ('Tell me about warp drive.'     , 'Bend the rules of space\nDilithium hums quietly\nDistance loses meaning'),
    ('Describe Captain Picard.'      , 'Make it so, he says\nTea, Earl Grey, hot, and the stars\nAttend his calm voice'),
    ('What is the Borg?'             , 'Resistance is... wait\nA cube approaches, eyes glow\nThe self becomes us'),
    ('Tell me about Klingons.'       , "Honor, blood, and song\nThe blade remembers each fight\nQapla' for the brave"),
    ('Describe tribbles.'            , 'Soft purring trouble\nOne becomes ten before lunch\nWe should not have fed'),
    ('What are lightsabers?'         , 'Hum in the dark cave\nA color choice tells your fate\nElegant in war'),
    ('Tell me about the Force.'      , 'Binds the rock to root\nThe stars to the sleeping child\nFeel it, do not think'),
    ('Describe Jedi.'                , 'Robes the brown of dust\nSit very still, listen long\nDo, or do not, try'),
    ('Tell me about Darth Vader.'    , 'Breath like a slow tide\nMore machine now than was man\nGrief beneath the mask'),
    ('Describe Yoda.'                , 'Small, green, very old\nSyntax inverted he speaks\nWisdom older still'),
    ('What is the Death Star?'       , 'Moon that is not moon\nSilent grey weight in the void\nFlaw too small to see'),
    ('Tell me about Han Solo.'       , 'Smuggler with a heart\nMade the Kessel run, you know\nShoots first, asks no one'),
    ('Describe Chewbacca.'           , 'Roar of loyal love\nFur taller than any door\nA debt never closed'),
    ('Tell me about R2-D2.'          , 'Beeps, whistles, secrets\nRolls over impossible ground\nKnows more than the rest'),
    ('Describe Tatooine.'            , 'Two suns, one shadow\nSand learns the shape of waiting\nThe boy looks upward'),
    ('What is the Millennium Falcon?', 'Hunk of junk that flies\nKessel run in twelve parsecs\nHome shaped like a wing'),
    ('Tell me about stormtroopers.'  , 'White armor, bad aim\nA hundred suits, one mission\nThe captain remains'),
    ('Describe the Matrix.'          , 'Green rain on black glass\nThe spoon was never solid\nWake, or sleep on still'),
    ('What is the One Ring?'         , 'Plain gold in the palm\nWeight beyond what hands can bear\nWhispers in the fire'),
    ('Tell me about Gandalf.'        , 'Tall hat, grey, then white\nOne does not simply walk in\nHe arrives in time'),
    ('Describe hobbits.'             , 'Furred feet, second breakfast\nSmall folk with the larger hearts\nThe shire endures them'),
    ('What is the TARDIS?'           , 'Police box that lies\nBigger inside than outside\nWhen loses meaning'),
    ('Tell me about Tron.'           , 'Light-cycle on grid\nUser becomes program here\nFights for what is real'),
    ('What is Linux?'                , 'Penguin in the box\nKernel does what kernels do\nFreedom is the bug'),
    ('Describe vim.'                 , 'Modal editor\nh, j, k, l, escape, write\nNever quitting :q!'),
    ('Tell me about coding.'         , "Curly braces close\nThe error was a typo\nHours of one's life"),
    ('What is a software bug?'       , "Not a feature now\nReproducible at last\nGit blame says: it's me"),
]

print(f"{len(HAIKU_DATASET)} training examples")
print()
print("Example:")
print(f"  Q: {HAIKU_DATASET[0][0]}")
print(f"  A: {HAIKU_DATASET[0][1]}")
```

## Format for the trainer

TRL's `SFTTrainer` expects either a single `text` field with the full conversation or messages. We use the chat template (built into SmolLM's tokenizer) to format each pair properly.


```python
def format_for_sft(q, a):
    messages = [
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list([format_for_sft(q, a) for q, a in HAIKU_DATASET])
print(f"{len(dataset)} training examples")
print()
print("First formatted example:")
print(dataset[0]["text"])
```

## QLoRA configuration

**LoRA** (Low-Rank Adaptation, Hu et al. 2021) trains *small adapter matrices* on top of the frozen base model. We update ~1% of the parameters and the rest stay quantised. This is what makes fine-tuning a 360M model feasible on a free T4.

Key knobs:
- `r=16`: rank of the adapters. Higher = more capacity, more params
- `lora_alpha=32`: scaling. Standard rule: `alpha = 2 * r`
- `target_modules="all-linear"`: apply LoRA to every linear layer in the transformer (broad targeting)


```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Train

The catastrophic 'useruseruser' overfit on the first run came from training on the *full* sequence (user prompt + assistant). The model learned to predict role-tag tokens at every position. Two settings prevent that:

- `num_train_epochs=5` — enough for format learning, not enough to memorise role tags
- A custom `CompletionOnlyCollator` — masks everything up to the assistant marker (`<|im_start|>assistant\n`) out of the loss so the model is only graded on its own response. This is what TRL's deprecated `DataCollatorForCompletionOnlyLM` did; we re-implement it inline because newer TRL removed it and SmolLM's chat template doesn't support `assistant_only_loss=True`.

~2-3 min on a T4.


```python
import torch
from transformers import DataCollatorForLanguageModeling

# Recent TRL versions removed DataCollatorForCompletionOnlyLM from the public
# API; SmolLM's chat template doesn't support assistant_only_loss either.
# So we roll our own: tokenize the full chat-templated sequence with the
# default collator, then null out (-100) every label up to and including
# the assistant role marker so the model is only graded on its own response.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # Tokenize the marker (without special tokens) so we can find it
        # as a subsequence in each example
        self.response_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, examples):
        batch = self.base(examples)
        n = len(self.response_ids)
        for i in range(batch["labels"].size(0)):
            ids = batch["input_ids"][i].tolist()
            # Find the response template tokens; mask everything up to and including them
            for j in range(len(ids) - n + 1):
                if ids[j : j + n] == self.response_ids:
                    batch["labels"][i, : j + n] = -100
                    break
            else:
                # Marker not found — drop this example from loss entirely
                batch["labels"][i, :] = -100
        return batch

collator = CompletionOnlyCollator(tokenizer, RESPONSE_TEMPLATE)

training_args = SFTConfig(
    output_dir="./haiku-sft",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

# Newer TRL (>= 0.13) prefers processing_class over the older tokenizer kwarg.
try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator,
    )

trainer.train()
```

## Test the fine-tuned model

Same prompts as before. The model should now answer in three-line haiku format.


```python
print("=== AFTER FINE-TUNING ===")
for prompt in TEST_PROMPTS:
    print()
    print(f">>> {prompt}")
    print(chat(prompt))

# Bonus: try prompts not in the training set
print()
print("=== UNSEEN PROMPTS ===")
for prompt in ["Describe technology.", "What is sleep?", "Tell me about coffee."]:
    print()
    print(f">>> {prompt}")
    print(chat(prompt))
```

## What just happened

You ran the canonical LLM customisation pipeline:

1. Loaded a real pretrained model in 4-bit
2. Attached LoRA adapters to every linear layer
3. Trained ~1% of parameters (the adapters) on a small format-shaping dataset
4. Got a model that follows the new format on prompts it never saw

Same pattern is used for: customer-support tone, code style, persona, instruction-following, domain adaptation. **Behaviour fine-tuning works with surprisingly little data**; you're not teaching the model new facts, just preferences. The base model already knows everything.

## Optional: DPO (Direct Preference Optimization)

SFT teaches the model what *good* looks like. DPO teaches it what's *better than what* — by training on *pairs* of (preferred, rejected) responses for the same prompt. It's the modern alternative to RLHF: same goal (align to human preferences), much simpler implementation (closed-form loss instead of reward model + PPO).

For our haiku model, a DPO step might use pairs like:
- Preferred: a tight 3-line haiku with concrete imagery
- Rejected: a haiku that's grammatically right but vague ("thing exists\nwe can describe\nit is here")

TRL has [`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) that takes a list of `(prompt, chosen, rejected)` triples and applies DPO with a few lines of config. Worth doing as exercise once you've gotten SFT working.

## References

- **Hu et al. 2021** — *LoRA: Low-Rank Adaptation of Large Language Models* ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- **Dettmers et al. 2023** — *QLoRA: Efficient Finetuning of Quantized LLMs* ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- **Rafailov et al. 2023** — *Direct Preference Optimization* ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290))
- **TRL docs** — [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **PEFT docs** — [huggingface.co/docs/peft](https://huggingface.co/docs/peft)

## Exercises

1. **More data, more epochs.** Add 30 more haiku examples. Re-train. Does the format become more consistent? Are unseen prompts handled better?
2. **Different tone.** Replace the haiku data with `(question, pirate-speak answer)` pairs. Train. The pipeline is identical, only the data differs.
3. **DPO on top of SFT.** Generate 50 SFT outputs, hand-label good vs bad, train DPO. Does the model improve?
4. **Bigger base model.** Try Qwen2.5-0.5B or SmolLM2-1.7B. Same code, different model name. How do training time and final quality change?
5. **Save and reload.** After training, `trainer.model.save_pretrained("./haiku-adapter")` saves only the LoRA weights (~10 MB). Load and merge them into a fresh base model. Confirm the loaded model produces haiku.
