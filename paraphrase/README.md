# Paraphrase Generation System README

## Overview

This repository contains the implementation of a Custom Paraphrase
Generator (CPG) as per the "SE Assignment GenAI -- Paraphrase Generation
System" requirements. The CPG is designed to paraphrase paragraphs of
200--400 words, ensuring the output is at least 80% of the input length.
It uses a transformer-based model for generation and includes modular
code for training (optional), inference, API, and evaluation. The system
is compared to a baseline LLM-based paraphraser in terms of text quality
(using BLEU, ROUGE, BERTScore) and latency.

The code is provided in a Jupyter Notebook format for easy reproduction.

### Key Features

-   Handles long inputs via sentence splitting and recombination.\
-   Enforces output length with regeneration attempts and fallback
    expansions.\
-   Evaluated on the provided cover letter test sample.

------------------------------------------------------------------------

## Model Choices

### Custom Paraphrase Generator (CPG)

Based on the **"Vamsi/T5_Paraphrase_Paws"** model from Hugging Face (a
T5-base variant fine-tuned on the PAWS dataset for adversarial
paraphrasing). It was chosen for: - Efficiency\
- Strong focus on English sentence-level rephrasing\
- Alignment with the open-source repository

It is customized with: - Sentence tokenization (via NLTK)\
- Sampling-based generation (top-k = 120, top-p = 0.95)\
- Length enforcement to handle paragraph inputs without losing coherence

### Baseline LLM-Based Generator

Model used: **"humarin/chatgpt_paraphraser_on_T5_base"** from Hugging
Face (T5-base fine-tuned on diverse paraphrase datasets to mimic
ChatGPT-style outputs). It is used as a representative baseline for
comparison, offering broader training but less specialization in
fidelity.

Both models are seq2seq transformers, ensuring a fair apples-to-apples
evaluation. No additional fine-tuning was performed beyond the
pre-trained checkpoints, though optional training code is included for
extension (e.g., on the ParaNMT dataset).

------------------------------------------------------------------------

## Evaluation Results

The system was evaluated on the provided test sample (cover letter
passage, \~320 words) using suitable metrics for quality
(lexical/semantic similarity) and system latency (inference time in
seconds). Metrics were computed over a single run, but averages from
multiple trials are recommended for production use.

  Metric        CPG     Baseline LLM
  ------------- ------- --------------
  BLEU          74.97   32.68
  ROUGE-1       0.93    0.69
  ROUGE-L       0.91    0.51
  BERTScore     0.98    0.92
  Latency (s)   9.74    11.50

### Interpretation

-   CPG outperforms on all quality metrics, indicating better
    preservation of original meaning and structure.\
-   Latency is lower for CPG due to optimized sampling compared to the
    baseline's beam search.

### Tools Used

-   BLEU via `sacrebleu`\
-   ROUGE via `rouge-score`\
-   BERTScore via `bert-score`

------------------------------------------------------------------------

## Error Analysis

### CPG Errors

-   Minor artifacts like word substitutions (e.g., "Purpose" → "Object")
    or repetitions (e.g., "6. 6. Signature"), likely from sampling
    variability or recombination after sentence splitting.\
-   Gender assumptions (e.g., "his name" instead of neutral "their
    name"), stemming from biases in the PAWS fine-tuning data.\
-   Occasional shortening of individual sentences below the per-sentence
    80% threshold, requiring fallback expansion phrases (e.g.,
    "(Expanded for clarity...)"), which can feel unnatural.

### Baseline Errors

-   More aggressive rephrasing leading to hallucinations (e.g.,
    inserting questions like "What is the purpose of '4'?") or loss of
    details (e.g., simplifying lists).\
-   Lower coherence in recombined outputs, with awkward transitions or
    incomplete sentences.

### Common Issues

-   Both struggle with enumerated lists (1--6), where formatting
    disrupts paraphrasing.\
-   Token limits can cause context fragmentation in long paragraphs.\
-   Root causes include reliance on sentence-level models and randomness
    introduced by sampling/beam search without full-paragraph awareness.

### Impact

CPG errors are generally less severe (as reflected in higher metrics),
while baseline deviations reduce usability for precise tasks such as
professional writing.

------------------------------------------------------------------------

## Summary of Findings and Possible Improvements

### Findings

The CPG successfully meets assignment requirements, handling 200--400
word inputs with enforced length and superior quality/latency compared
to the baseline. It excels in semantic fidelity (BERTScore \> 0.97) and
speed, making it suitable for applications like text augmentation or
writing aids. The baseline, while versatile, sacrifices accuracy for
variety, highlighting the value of customization.

### Possible Improvements

-   Fine-tune the CPG on paragraph-specific datasets (e.g., ParaNMT-50M)
    to improve long-context handling without splitting.\
-   Integrate coherence checks (e.g., GPT-2 perplexity scoring to rerank
    outputs) to reduce artifacts.\
-   Add diversity controls (e.g., self-BLEU metric) for more varied
    paraphrases without quality loss.\
-   Optimize latency by batching sentence processing or distilling to a
    smaller model (e.g., T5-small).\
-   Expand evaluation to diverse inputs (e.g., informal text,
    non-English) and average metrics over multiple runs.\
-   Enhance the API with better error handling for short inputs or
    integrate with Streamlit for a demo UI.

------------------------------------------------------------------------

For setup instructions, see the notebook.\
**Dependencies:** transformers, nltk, sacrebleu, rouge-score,
bert-score, torch, flask.\
Run on GPU for best performance.

**Akash (@aks301190)** -- If you have questions or need tweaks, feel
free to reach out!
