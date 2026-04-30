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
def chat(prompt, max_new=80, temperature=0.7, top_p=0.9):
    """Sample from the model. Sampling (not greedy) helps the fine-tuned
    model escape mode collapse — greedy can lock into a degenerate prefix."""
    messages = [{"role": "user", "content": prompt}]
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
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
    ('Describe hope.'                , 'Small green pushing through\nBetween stones the road forgot\nReaching toward sun'),
    ('What is doubt?'                , 'Two paths in the wood\nNeither one looks quite like home\nThe boots stand still'),
    ('Tell me about wonder.'         , 'Child looks at the sky\nHas no question yet, only\nMouth slightly open'),
    ('Describe curiosity.'           , 'Cat finds the closed box\nThe world is what is not seen\nOne paw, then the rest'),
    ('What is trust?'                , 'Eyes closed, falling back\nBelieving the hands behind\nWill catch what is dropped'),
    ('Describe loyalty.'             , 'Dog at the empty\nDoor for hours, no question why\nHe will be coming'),
    ('Tell me about honesty.'        , 'Bare wood, no lacquer\nThe knot shows where the tree grew\nSplinters welcomed in'),
    ('What is faith?'                , "Walking at midnight\nCan't see five steps ahead, still\nThe road is going"),
    ('Describe pride.'               , 'Polished armor shines\nReflects all who pause to look\nKeeps the rain outside'),
    ('What is envy?'                 , "Neighbor's apple tree\nGreener through the broken fence\nMine drops fruit, too late"),
    ('Tell me about compassion.'     , "Stranger's hand finds yours\nNo question of currencies\nWeight shared lightens both"),
    ('Describe ambition.'            , 'Eye fixed on the peak\nForgets to admire the moss\nStone has its slow word'),
    ('What is acceptance?'           , 'Stone in the river\nNo longer fights the current\nLearns the shape of flow'),
    ('Describe resilience.'          , 'Tree bent by the storm\nSprings back when the wind has gone\nNew leaves where it broke'),
    ('Tell me about humility.'       , 'Master sweeps the path\nWearing the same robe each year\nBows lower than need'),
    ('Describe coffee.'              , 'Black mirror in cup\nMornings begin with this scent\nFirst taste, then the day'),
    ('Tell me about tea.'            , 'Steam writes on cold air\nLeaves give what they learned slowly\nWarmth held in two hands'),
    ('Describe bread.'               , 'Crust dark, inside soft\nThe oven taught it patience\nBest broken in halves'),
    ('What is wine?'                 , 'Sun in the bottle\nGrape remembers its long year\nWaits for the right meal'),
    ('Tell me about sleep.'          , 'Mind sets down its tools\nThe body does its slow work\nDawn finds you elsewhere'),
    ('Describe a door.'              , 'Stranger from outside\nFamiliar from the inside\nTwo houses divided'),
    ('Tell me about windows.'        , 'Square of weather framed\nCat sits and watches the world\nNo verdict given'),
    ('What is a bridge?'             , "Two banks, one promise\nCarries strangers' hurried feet\nForgets to ask why"),
    ('Describe a road.'              , 'Where the foot decides\nBecomes the path in the grass\nMaps follow our wear'),
    ('Tell me about lanterns.'       , 'Small fire kept in glass\nRemembers the stars are far\nLights what is near home'),
    ('Describe a clock.'             , 'Hands turning circles\nAll years measured in the same\nCountless small forevers'),
    ('Tell me about letters.'        , 'Slow voice through the mail\nThe friend, far, hands you their week\nPaper holds the smile'),
    ('What is a photograph?'         , 'One moment caught flat\nThe room is gone, the face stays\nLight remembers light'),
    ('Describe a phone.'             , 'World fits in the palm\nFriends, news, weather, lies, the work\nWe forget to look up'),
    ('Tell me about trains.'         , 'Steel keeping its word\nArrives when the schedule said\nThe valley unrolls'),
    ('Describe whales.'              , 'Singers in deep blue\nThe oldest songs of the world\nBigger than our maps'),
    ('Tell me about foxes.'          , 'Russet shadow slips\nBetween fence and silver moon\nWatches, never named'),
    ('Describe owls.'                , 'Eyes the size of stars\nHead turns where neck does not turn\nSilence has wings here'),
    ('What are eagles?'              , 'Speck against the blue\nDescends precise, asks no leave\nRiver gives a fish'),
    ('Tell me about ravens.'         , 'Black laughter on roof\nKnows where the funeral goes\nReturns, leaves a coin'),
    ('Describe salmon.'              , "Upriver they go\nAgainst all the current's claim\nHome older than self"),
    ('Tell me about bees.'           , 'Gold dust in their fur\nThe shape of summer wears wings\nFlowers pay in light'),
    ('Describe butterflies.'         , 'Scrap of stained-glass life\nThrough the long dark of cocoon\nNow remembers air'),
    ('What is a storm?'              , 'Sky writes in long flash\nThe earth listens, then exhales\nSomething decided'),
    ('Tell me about lightning.'      , 'Crack between two dark\nA second of bright as truth\nThunder counts the miles'),
    ('Describe a lake.'              , 'Mirror of the sky\nFish move beneath reflections\nStillness holds the world'),
    ('Tell me about streams.'        , 'Voice between the stones\nNot a hurry, not a stop\nGoing somewhere kind'),
    ('Describe mushrooms.'           , "Morning's small surprise\nThey were not here yesterday\nQuiet, do not ask"),
    ('Tell me about moss.'           , 'Wins the slow contest\nWhere harder things wear themselves\nSoft outlasts the stone'),
    ('What are roses?'               , 'Beauty with the blade\nThe gift hides what it costs you\nLove keeps both, somehow'),
    ('Describe Frodo.'               , 'Small hands, large burden\nWalks toward the fire that knows him\nReturns less, returns'),
    ('Tell me about Samwise.'        , 'Carries the master\nWhen the ring will not let go\nFriendship breaks no oath'),
    ('What is Mordor?'               , 'Land that cannot heal\nEvery shadow is the same\nOnly the eye watches'),
    ('Describe Hogwarts.'            , 'Castle changes stairs\nLetters arrive by feather\nStudent finds new self'),
    ('Tell me about Snape.'          , 'Man dressed in black grief\nLove disguised as cruelty\nAlways, his last word'),
    ('Describe Dumbledore.'          , 'Long beard, longer plans\nLemon drops in his pocket\nKnows what he conceals'),
    ('Tell me about the Doctor.'     , 'Two hearts, one blue box\nWalks through time as through old halls\nNever quite goodbye'),
    ('What are Daleks?'              , 'Tin can with a voice\nEXTERMINATE the only word\nCannot climb the stairs'),
    ('Describe Neo.'                 , 'Coder finds the door\nSpoon was always made of mind\nKung fu downloads in'),
    ('Tell me about Captain Kirk.'   , 'Bold across the stars\nTorn shirt, alien fights, returns\nKisses then commands'),
    ('Describe Worf.'                , "Klingon among them\nHonor in the Federation\nHis bat'leth waits home"),
    ('Tell me about Data.'           , 'Android wants a soul\nLearns laughter from the captain\nOne day, almost there'),
    ('What is the holodeck?'         , 'Room becomes a world\nSherlock holds his pipe again\nReality glitches'),
    ('Describe a replicator.'        , 'Tea, Earl Grey, hot, spoken\nMolecules assemble fast\nNo one fries an egg'),
    ('What are phasers?'             , "Set to stun, you said\nThe alien folds politely\nLearning from war's edge"),
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
    r=8,                      # was 16 — half the params, less room to overfit on small data
    lora_alpha=16,            # rule of thumb: alpha = 2 * r
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Train

Three settings to balance learning vs overfit on a small (60-example) dataset:

- **LoRA `r=8`** — fewer trainable parameters, less capacity to memorise random patterns
- **`num_train_epochs=10`** — between the under-fit 5 and over-fit 15 we tried
- **`CompletionOnlyCollator`** — masks every label up to `<|im_start|>assistant\n` from the loss so the model only learns from its own response

The cell below also prints a quick mask-inspection so you can see which tokens contribute to the loss and which are masked out.

~3-5 min on a T4.


```python
import torch
from transformers import DataCollatorForLanguageModeling

# Recent TRL versions removed DataCollatorForCompletionOnlyLM from the public API;
# SmolLM's chat template doesn't support assistant_only_loss either. So we
# roll our own: tokenize the full chat-templated sequence with the default
# collator, then null out (-100) every label up to and including the assistant
# role marker so the model is only graded on its own response.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.response_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, examples):
        batch = self.base(examples)
        n = len(self.response_ids)
        for i in range(batch["labels"].size(0)):
            ids = batch["input_ids"][i].tolist()
            for j in range(len(ids) - n + 1):
                if ids[j : j + n] == self.response_ids:
                    batch["labels"][i, : j + n] = -100
                    break
            else:
                batch["labels"][i, :] = -100
        return batch

collator = CompletionOnlyCollator(tokenizer, RESPONSE_TEMPLATE)

# Sanity print: show a tokenized example with its mask, so it's obvious which
# tokens contribute to loss and which don't.
sample_batch = collator([dataset[0]])
print(f"sample input length: {sample_batch['input_ids'].size(1)} tokens")
print(f"unmasked label tokens: {(sample_batch['labels'] != -100).sum().item()} (the assistant's response)")
print()

training_args = SFTConfig(
    output_dir="./haiku-sft",
    num_train_epochs=10,                  # middle ground (5 = undertrained, 15 = collapse)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1.5e-4,                 # middle ground (1e-4 = barely moves, 2e-4 = collapse)
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
