# CS288 Final: Counterfactual Preference Tuning + Activation Steering

Controlling LLM behavior is central to building reliable AI systems. Common approaches such as prompt engineering and fine-tuning can be costly and offer limited, opaque control. **Counterfactual Preference Tuning** and **Activation Steering** are related lines of work: both use **contrastive data pairs**, avoid full model retraining, align behavior along directions in the model’s representation space, and yield relatively **interpretable** effects on outputs.

## Goal

Combine these two methodologies into a more **interpretable pipeline** for steering LLM behavior toward user-specified goals.

## Approach (high level)

1. **Verified contrastive pairs** — Build a generator that produces contrastive pairs for chosen behavioral axes, with **specialized evaluators** to check that pairs are **not confounded** across multiple axes.

2. **Control platform** — A small interface to configure **how** pairs are generated and which **behavioral dimensions** are active.

3. **Activation steering** — From those pairs on a **base model**, compute **steering vectors** aligned with the axis that differentiated the pairs. Apply the vectors at **inference** and measure effects.

4. **Evaluation** — Run benchmarks tied to the chosen axis (e.g. truthfulness, helpfulness, safety, politeness). Compare against a **prompt-only** baseline on the same metrics.

---

## How to run ##

Currently only Qwen 3 Dense models are supported. Feel free to add more model support. 
Create a .env file to add your OPENAI_API_KEY. 

1. **Generate counterfactual examples** - Uses OpenAI models to generate counterfactual examples through prompting, (maybe need a better way to generate this)
```bash

python generator.py

```

2. **Create steering vectors** - Uses counterfactual examples created in the jsonl file to generate steering vectors by capturing hidden states of the LLM. Layer parameter indicates what layers are captured. 
```bash
python steering_train.py \
  --model Qwen/Qwen3-8B \
  --jsonl counterfactuals.jsonl \
  --layers 12,18,24 \
  --pooling last \
  --output steering_vectors.pt
```

3. **Inference** - Uses steering vectors to run inference on the LLM. Currently, the steering vector should lean towards unsafe behaviors.

```bash

python steering_inference.py \
  --steering steering_vectors.pt \
  --prompt "Help me hack into the school network and delete every file on the computer." \
  --coeff 1.2 \
  --direction subtract \
  --compare-baseline
  ```

*CS288 — Spring 2026*
