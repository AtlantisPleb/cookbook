# FINETUNING_INDEX

This index collects the Hugging Face Cookbook material that is directly about
fine-tuning, post-training, prompt tuning, preference optimization, or the data
work that feeds those training runs. English notebooks are treated as the
canonical versions; translated notebooks are listed separately at the end.

## Direct LLM, Code, And Text Fine-Tuning

- [Documentation Chatbot with Meta Synthetic Data Kit](notebooks/en/fine_tune_chatbot_docs_synthetic.ipynb):
  Builds a documentation chatbot fine-tuning set from LangChain Chat Models docs
  using Meta's `synthetic-data-kit`, formats QA pairs, loads
  Llama-3.2-3B-Instruct in 4-bit with Unsloth, attaches LoRA adapters, and
  trains with TRL `SFTTrainer` / `SFTConfig`. It also includes Colab memory
  notes and a quick before/after chatbot test.

- [Fine-tuning a Code LLM on Custom Code on a single GPU](notebooks/en/fine_tuning_code_llm_on_single_gpu.ipynb):
  Fine-tunes `bigcode/starcoderbase-1b` on a streamed code dataset
  (`smangrul/hf-stack-v1`) using constant-length token packing, optional
  fill-in-the-middle augmentation, bitsandbytes quantization, PEFT LoRA, and
  the Transformers `Trainer`. It covers Hub upload, LoRA merge, and code
  completion inference.

- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](notebooks/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format.ipynb):
  Walks through LoRA / QLoRA supervised fine-tuning for turning Persian product
  titles and descriptions into structured catalog JSON. It explains PEFT,
  4-bit BitsAndBytes loading, chat templates, TRL `SFTTrainer`, pushing either
  adapters or merged models to the Hub, and serving the result with vLLM.

- [Fine-tuning LLMs for Function Calling with xLAM Dataset](notebooks/en/function_calling_fine_tuning_llms_on_xlam.ipynb):
  Provides a reusable QLoRA function-calling fine-tuning pipeline over
  `Salesforce/xlam-function-calling-60k`. It normalizes prompts into
  user/tool/call sections, auto-detects tokenizer and EOS settings for common
  model families, trains with TRL `SFTTrainer`, and tests tool-call generation
  from LoRA adapters.

- [GitHub Tag Generator with T5 + PEFT (LoRA)](notebooks/en/finetune_t5_for_search_tag_generation.ipynb):
  Fine-tunes T5 with PEFT LoRA for GitHub search tag generation. It includes
  dataset loading and preprocessing, tokenizer setup, LoRA injection,
  `TrainingArguments`, `Trainer` training, W&B logging, metrics, postprocessing,
  local save, Hub push, reload, and inference examples on real projects.

- [Prompt Tuning With PEFT](notebooks/en/prompt_tuning_peft.ipynb):
  Demonstrates soft prompt tuning with PEFT on Bloom-style causal language
  models. It explains prompt tuning concepts, supported model families, baseline
  inference, dataset preparation, two prompt-tuning configurations, `Trainer`
  training, saving, and inference from the tuned prompts.

- [Suggestions for Data Annotation with SetFit in Zero-shot Text Classification](notebooks/en/labelling_feedback_setfit.ipynb):
  Combines Argilla and SetFit to bootstrap text-classification annotations.
  It trains zero-shot / few-shot SetFit models with Sentence Transformers,
  predicts label suggestions for Banking77 records, logs them for feedback in
  Argilla, and exports curated data through the Hub.

## Vision Fine-Tuning

- [Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset](notebooks/en/fine_tuning_vit_custom_dataset.ipynb):
  Fine-tunes a ViT image classifier on a custom biomedical dataset. It covers
  dataset splits, label mapping, torchvision transforms, image processing,
  batch collation, `Trainer` fine-tuning, optional Hub upload, and analysis with
  a confusion matrix and recall scores.

- [Fine-Tuning Object Detection Model on a Custom Dataset, Deployment in Spaces, and Gradio API Integration](notebooks/en/fine_tuning_detr_custom_dataset.ipynb):
  Fine-tunes DETR for object detection on a custom dataset. The notebook covers
  dataset inspection, bounding-box validation, class distribution checks,
  augmentation, image processor setup, model initialization, Hub login, W&B
  training, test-set evaluation, Spaces deployment, and Gradio API usage.

- [Fine-Tuning a Semantic Segmentation Model on a Custom Dataset and Usage via the Inference API](notebooks/en/semantic_segmentation_fine_tuning_inference.ipynb):
  Fine-tunes a SegFormer-style semantic segmentation model on
  `segments/sidewalk-semantic`. It includes class mapping, visual inspection,
  Albumentations transforms, image processor setup, custom `compute_metrics`
  with `evaluate`, W&B-backed `Trainer` training, test images, and Inference API
  usage.

## Vision-Language Fine-Tuning And Preference Optimization

- [Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with the Hugging Face Ecosystem (TRL)](notebooks/en/fine_tuning_vlm_trl.ipynb):
  Fine-tunes Qwen2-VL-7B on ChartQA with TRL. It shows dataset and model setup,
  baseline inference, quantized loading, QLoRA configuration, `SFTConfig`,
  `SFTTrainer` training, fine-tuned inference, and comparison against the base
  model plus prompting.

- [Fine-tuning SmolVLM with TRL on a consumer GPU](notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb):
  Runs supervised fine-tuning for SmolVLM on consumer-GPU constraints. It
  includes dataset loading, baseline evaluation, quantized model loading, QLoRA
  setup, TRL `SFTTrainer` training, and final generation tests.

- [Fine-tuning Granite Vision 3.1 2B with TRL](notebooks/en/fine_tuning_granite_vision_sft_trl.ipynb):
  Fine-tunes Granite Vision 3.1 2B with TRL and PEFT. It covers dataset/model
  overview, baseline behavior, quantized loading, QLoRA setup, `SFTConfig`,
  training, and post-training tests.

- [Fine tuning a VLM for Object Detection Grounding using TRL](notebooks/en/fine_tuning_vlm_object_detection_grounding.ipynb):
  Fine-tunes a VLM for object-detection grounding, including dataset column
  cleanup, image handling, caption-based splitting, bounding-box visualization,
  baseline inference, LoRA + TRL SFT, resize/collator helpers, Hub upload, and
  train/validation sample testing.

- [Fine-tuning SmolVLM using direct preference optimization (DPO) with TRL on a consumer GPU](notebooks/en/fine_tuning_vlm_dpo_smolvlm_instruct.ipynb):
  Applies DPO preference optimization to SmolVLM with TRL, PEFT, quantized
  loading, and QLoRA. It shows dataset loading, DPO configuration, model setup,
  adapter training, and final model testing.

- [Fine-Tuning a Vision Language Model with TRL using MPO](notebooks/en/fine_tuning_vlm_mpo.ipynb):
  Demonstrates MPO-style preference optimization for a VLM using TRL. It loads
  the preference dataset, prepares a quantized model, configures QLoRA and
  MPO-related `DPOConfig` settings, trains, and tests the resulting model.

## Reasoning And RL-Style Post-Training

- [Post training an LLM for reasoning with GRPO in TRL](notebooks/en/fine_tuning_llm_grpo_trl.ipynb):
  Post-trains an LLM for mathematical reasoning with GRPO. It loads a reasoning
  dataset and baseline model, configures LoRA, defines math-verification reward
  functions, sets GRPO training parameters, trains with TRL, and checks
  performance.

- [Post training a VLM for reasoning with GRPO using TRL](notebooks/en/fine_tuning_vlm_grpo_trl.ipynb):
  Extends GRPO post-training to a vision-language reasoning model. It includes
  dataset loading, baseline behavior, LoRA setup, multimodal reward functions,
  GRPO configuration, training, and performance checks.

- [Efficient Online Training with GRPO and vLLM in TRL](notebooks/en/grpo_vllm_online_training.ipynb):
  Shows online GRPO training using TRL with vLLM acceleration. It covers
  baseline setup, LoRA, reward functions, GRPO parameters, training,
  configuration comparisons, and performance checks.

- [Advanced GRPO Fine-tuning for Mathematical Reasoning with Multi-Reward Training](notebooks/en/trl_grpo_reasoning_advanced_reward.ipynb):
  Builds a more advanced mathematical reasoning GRPO recipe. It includes GPU
  detection, model selection, LoRA config, GSM8K setup, a multi-reward design,
  Trackio experiment tracking, training, evaluation, cleanup, and references.

## Fine-Tuning Support: Data, Preference Sets, Quality, And HPO

- [Generate a Preference Dataset with distilabel](notebooks/en/generate_preference_dataset_distilabel.ipynb):
  Creates a preference dataset suitable for DPO, ORPO, or RLHF training. It
  loads prompts from the Hub, generates paired responses with inference
  endpoints, groups responses, scores them with UltraFeedback, converts the
  result into chosen/rejected preference format, optionally sends it to Argilla,
  and pushes the dataset to the Hub.

- [Clean an Existing Preference Dataset with LLMs as Judges](notebooks/en/clean_dataset_judges_distilabel.ipynb):
  Cleans `Intel/orca_dpo_pairs` with distilabel, Argilla, and LLM-as-judge
  scoring. It randomizes chosen/rejected order to reduce bias, scores responses
  with UltraFeedback through the HF Inference API, keeps the columns needed for
  DPO-style training, supports Argilla review, and pushes a cleaned dataset.

- [Data Annotation with Argilla Spaces](notebooks/en/enterprise_cookbook_argilla.ipynb):
  Builds train/evaluation data for code generation using HF Inference API model
  outputs and Argilla Spaces. It configures the annotation interface, stores and
  corrects LLM responses, and explicitly positions the resulting data for TRL
  `SFTTrainer` train/test splits or DPO ratings with `DPOTrainer`.

- [Annotate text data using Active Learning with Cleanlab](notebooks/en/annotate_text_data_transformers_via_active_learning.ipynb):
  Uses Cleanlab active learning to improve a Transformer text classifier under a
  small labeling budget. It fine-tunes DistilBERT, computes out-of-sample
  probabilities, selects examples to label or relabel with ActiveLab scores, and
  retrains over multiple rounds.

- [Detecting Issues in a Text Dataset with Cleanlab](notebooks/en/issues_in_text_dataset.ipynb):
  Audits text-classification data before fine-tuning by finding likely label
  errors, outliers, near duplicates, and non-IID examples. It uses Transformer
  embeddings plus a classifier to score Banking77-OOS examples and frames the
  cleanup as a way to improve downstream model training.

- [Hyperparameter Optimization with Optuna and Transformers](notebooks/en/optuna_hpo_with_transformers.ipynb):
  Wraps a Transformers fine-tuning workflow in Optuna HPO. It loads and
  tokenizes IMDB sentiment data, defines metrics and train/validation splits,
  uses persistent RDB-backed Optuna storage, runs `Trainer` trials with W&B,
  visualizes the study, and performs a final full fine-tune with the best
  hyperparameters before saving/uploading.

- [20x Faster TRL Fine-tuning with RapidFire AI](notebooks/en/rapidfire_sft_multiconfig_training.ipynb):
  Runs concurrent multi-configuration SFT experiments with RapidFire AI and
  TRL. It includes local/Colab service setup, TensorBoard, dataset preparation,
  metric helpers, model configuration factories, multiple SFT runs, evaluation,
  an interactive run controller, and plots.

## Translations

These notebooks are localized versions of fine-tuning recipes above. They are
useful when comparing terminology, examples, or screenshots in non-English
contexts, but the English notebooks usually remain the source of truth for
implementation details.

### Turkish

- [Fine-tuning a Code LLM on Custom Code on a single GPU - Turkish](notebooks/tr/fine_tuning_code_llm_on_single_gpu.ipynb):
  Turkish version of the single-GPU StarCoder code fine-tuning recipe, covering
  dataset streaming, model preparation, quantized LoRA training, inference, and
  Hub workflows.

- [Fine-Tuning Object Detection Model on a Custom Dataset - Turkish](notebooks/tr/fine_tuning_detr_custom_dataset.ipynb):
  Turkish version of the DETR custom object-detection fine-tuning guide,
  including dataset splits, bbox filtering, augmentation, training, evaluation,
  Spaces deployment, and API integration.

- [Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset - Turkish](notebooks/tr/fine_tuning_vit_custom_dataset.ipynb):
  Turkish version of the ViT biomedical image-classification fine-tuning guide,
  including preprocessing, transforms, training, optional Hub upload, confusion
  matrix, and recall analysis.

- [Fine-Tuning a Semantic Segmentation Model - Turkish](notebooks/tr/semantic_segmentation_fine_tuning_inference.ipynb):
  Turkish version of the semantic segmentation fine-tuning and Inference API
  recipe, covering dataset inspection, Albumentations, model setup, W&B-backed
  training, metrics, and inference.

### Korean

- [Fine-Tuning Object Detection Model on a Custom Dataset - Korean](notebooks/ko/fine_tuning_detr_custom_dataset.ipynb):
  Korean version of the DETR object-detection fine-tuning, deployment, and
  Gradio API recipe.

- [Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset - Korean](notebooks/ko/fine_tuning_vit_custom_dataset_ko.ipynb):
  Korean version of the ViT biomedical image-classification fine-tuning guide.

- [Fine-tuning SmolVLM using DPO with TRL - Korean](notebooks/ko/ko_fine_tuning_vlm_dpo_smolvlm_instruct.ipynb):
  Korean version of the SmolVLM DPO preference-optimization recipe with TRL,
  PEFT, QLoRA, and consumer-GPU constraints.

### Chinese

- [Fine-tuning a Code LLM on Custom Code on a single GPU - Chinese](notebooks/zh-CN/fine_tuning_code_llm_on_single_gpu.ipynb):
  Chinese version of the StarCoder single-GPU code fine-tuning guide.

- [Fine-Tuning Object Detection Model on a Custom Dataset - Chinese](notebooks/zh-CN/fine_tuning_detr_custom_dataset.ipynb):
  Chinese version of the DETR custom object-detection fine-tuning and Spaces
  deployment guide.

- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format - Chinese](notebooks/zh-CN/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format.ipynb):
  Chinese version of the Persian product-catalog JSON LoRA / QLoRA
  fine-tuning recipe.

- [Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset - Chinese](notebooks/zh-CN/fine_tuning_vit_custom_dataset.ipynb):
  Chinese version of the ViT biomedical image-classification fine-tuning guide.

- [Fine-Tuning a Vision Language Model with TRL - Chinese](notebooks/zh-CN/fine_tuning_vlm_trl.ipynb):
  Chinese version of the Qwen2-VL TRL SFT recipe.

- [Generate a Preference Dataset with distilabel - Chinese](notebooks/zh-CN/generate_preference_dataset_distilabel.ipynb):
  Chinese version of the distilabel preference-dataset generation guide for
  DPO/ORPO/RLHF-style training.

- [Suggestions for Data Annotation with SetFit - Chinese](notebooks/zh-CN/labelling_feedback_setfit.ipynb):
  Chinese version of the Argilla + SetFit annotation-support workflow.

- [Prompt Tuning With PEFT - Chinese](notebooks/zh-CN/prompt_tuning_peft.ipynb):
  Chinese version of the PEFT prompt-tuning tutorial.

- [Fine-Tuning a Semantic Segmentation Model - Chinese](notebooks/zh-CN/semantic_segmentation_fine_tuning_inference.ipynb):
  Chinese version of the semantic segmentation fine-tuning and Inference API
  recipe.

## Excluded From This Index

The cookbook also contains inference, RAG, deployment, agent, quantization, and
Hub workflow notebooks that may mention training or fine-tuned models in passing.
They are excluded here when they do not include a concrete fine-tuning,
post-training, prompt-tuning, preference-optimization, or fine-tuning data
preparation workflow.
