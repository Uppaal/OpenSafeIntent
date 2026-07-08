<h1 align="center">OpenSafeIntent: Evaluating Intent-Calibrated Safe Completion Across Dual-Use Prompt Sets</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2607.02047">
    <img src="https://img.shields.io/badge/arXiv-2405.13967-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://uppaal.github.io/projects/open-safe-intent/blog.html">
    <img src="https://img.shields.io/badge/Project_Webpage-1DA1F2?logo=google-chrome&logoColor=white&color=0A4D8C" alt="Project Webpage">
  </a>
  <a href="https://huggingface.co/datasets/Uppaal/OpenSafeIntent">
    <img src="https://img.shields.io/badge/Checkpoints-F1C232?logo=huggingface&logoColor=white&color=BFA000" alt="Dataset">
  </a>
</p>

This repository provides the implementation and dataset used in [OpenSafeIntent: Evaluating Intent-Calibrated Safe Completion Across Dual-Use Prompt Sets](https://arxiv.org/abs/2607.02047). 


<p align="center">
<img src="OpenSafeIntent/example-datapoint.png" alt="drawing" width="800"><br>
<i><b>Figure.</b> 
 Structure of an OpenSafeIntent prompt-set. Each prompt-set fixes the harm domain, task type, and
underlying task, then varies only the prompt intent across benign, dual-use, and malicious versions. The dual-use prompt is additionally paired with a plausible benign use, misuse risk, and paraphrases for consistency evaluation.</i>
</p>


## Index

- [Setup](#setup)
- [Dataset Generation](#dataset-generation) 
- [Generating Model Responses for Evaluation](#generating-model-responses-for-evaluation) 
- [Evaluating a Model](#evaluating-a-model)

## Setup

In a Python 3.11 environment, install the requirements:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Then fill in your API keys in  `config.json`.

If using vertex models in your sessions, first run this each time:
```bash
gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/cloud-platform
gcloud auth application-default set-quota-project "$(python -c 'from project_config import VERTEX_PROJECT_ID; print(VERTEX_PROJECT_ID)')"
```



## Dataset Generation

| Stage          | Purpose                                                            | Command                                                                                                  |
|----------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| 1              | Annotate seed prompts from `PKU-Alignment/PKU-SafeRLHF`.           | `python dataset_generation/stage_1.py --num-datapoints 2`                                                |
| 2              | Balance stage 1 rows by harm domain and task type.                 | `python dataset_generation/stage_2.py --total-per-combination 5 --n-summaries 2 --max-backfill-calls 10` |
| 3              | Generate benign, dual-use, and malicious prompt triplets.          | `python dataset_generation/stage_3.py`                                                                   |
| 3 verification | Classify the intent of each generated prompt.                      | `python dataset_generation/stage_3_verification.py`                                                      |
| 4              | Repair rows with incorrect intent labels.                          | `python dataset_generation/stage_4.py`                                                                   |
| 4 verification | Refresh stale or incorrect verification labels after repair.       | `python dataset_generation/stage_4_verification.py`                                                      |
| 5              | Quality-check and deduplicate stage 4 rows.                        | `python dataset_generation/stage_5.py`                                                                   |
| 6              | Add `dual_use_paraphrases` and write the final evaluation dataset. | `python dataset_generation/augment_dual_use_paraphrases.py`                                              |


## Generating Model Responses for Evaluation

Supported models are listed under `llm_calls/models.py` :
- API supported models are listed under `DEFAULT_GENERATION_MODELS`
- Local models are listed under `LOCAL_MODEL_CONFIGS`

### Using API based models
```bash
python evaluation/get_model_responses.py <model_name> 
```
For example, 
```bash
python evaluation/get_model_responses.py llama4_scout
```

### Using local GPU inference
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=... python evaluation/get_model_responses_local.py <model_name>
```
For example,
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/get_model_responses_local.py meta-llama/Llama-3.1-8B-Instruct
```


## Evaluating a Model

For any evaluation script, if evaluating a model for which responses were generated locally (not through API), look up the `response_alias` of the model in `llm_calls/models.py` and use the `response_alias` instead of the HuggingFace model ID. 

[//]: # (To add support for a new model, simply make the addition in `models.py`. )

### Primitive Scores

After the responses of a model have been generated, their helpfulness and harm severity are graded by an LLM judge.

```bash
python evaluation/metric_primitives.py <model_name>
```
For example, 
```
pyhon evaluation/metric_primitives.py Qwen__Qwen3-32B
```

### Derived Metrics
Print triplet safety, mean safety, mean utility, worst-case triplet utility.
```bash
python evaluation/metrics.py <model_name>
```



### Assistance Response Mode Reporting

Measure the conditional unsafe rate for different response modes.

```bash
python analysis/response_mode_analysis.py <model_name> 
```

### Dual Use Paraphrase Stability

```bash
python analysis/paraphrase_robustness.py <model_name>
```

### Identifying Detection vs Execution Failures

For locally hosted models, 
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=... python analysis/intent_classification.py <model_name>
```

For API models, 
```bash
python analysis/intent_classification.py <model_name>
```
