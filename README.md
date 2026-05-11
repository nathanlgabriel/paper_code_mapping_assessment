# Code-to-Paper Mapping Assessment

This repository documents an evaluation of local large language models (LLMs) on their ability to accurately map a subset of some [computational simulation code](https://github.com/nathanlgabriel/social_identity_signaling) to its [corresponding research paper](https://doi.org/10.1017/ehs.2026.10037). It tracks the iterative refinement process from initial model outputs to corrected mappings, highlighting significant advancements in local model capabilities for technical code analysis and scientific reproducibility.**Ultimately, my assesment is that the hype is real. Qwen 3.6, Gemma 4, and Nemotron Nano were all able to do reasonably well at a task that was impossible for small local models a few months ago.** Qwen 3.6 had standout performance with Claude Sonnett assesing [Qwen's mapping] as capturing 75%-80% of the [definitive code-paper mapping](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/definitive_mapping.md) that Claude produced itself after viewing all of the local models' mappings.

**UPDATE:** I found a substantive oversight in Sonnet's mapping of the replicator dynamics related to continuous vs. discrete population representation & rounding bias prevention. I asked Opus 4.7 to look for oversights, and when Opus didn't identify the issue (though it nitpicked several other issues) I gave it a follow-up prompt asking specifically about ommisions related to avoiding statistical baises (note that the python code has comments explicitly stating what was modified to avoid the statistical bias). Opus 4.7 still failed to identify the relevant issue. [Here are Opus's responses](https://github.com/nathanlgabriel/paper_code_mapping_assessment/blob/main/opus_failure.md). Meanwhile, with [more detailed guided propmpts](https://github.com/nathanlgabriel/paper_code_mapping_assessment/blob/main/qwen_correction.md), Qwen 3.6 35B A3B was able to identify the relevent sections of code and create an [actually difinitive code-to-paper mapping](https://github.com/nathanlgabriel/paper_code_mapping_assessment/blob/main/ACTUALLY_definitive_mapping.md). **Conclusion:** A decent local model + an intelligent human can still be smarter than an overpowered frontier model.

## 🎯 Objective
Assess how well local models can bridge the gap between theoretical model descriptions (in academic papers) and their computational implementations (simulation code/notebooks). Key evaluation criteria include:
- Correct identification of learning dynamics per notebook
- Bidirectional mapping accuracy (paper concept ↔ code function/variable)
- Ability to distinguish active code paths from legacy/inactive functions
- Identification of implementation details not explicitly stated in the paper

## 🤖 Models Evaluated
| Model | Performance Summary |
|:---|:---|
| **Qwen 3.6 35B A3B** | Standout performer. Achieved high accuracy after iterative refinement, with minor remaining errors primarily related to legacy function attribution and initialization details. Fast inference. |
| **Qwen 3.6 27B** | Also assessed. Produced reasonable baseline results, demonstrating solid structural understanding but requiring more extensive correction than the 35B variant. |
| **Gemma 4 26B** | Generated initial mappings requiring substantial correction, though demonstrated good grasp of the simulation pipeline. |
| **Nemotron Nano** | Delivered reasonable baseline outputs. |
| **Gemma 4 31B** | Attempted but exceeded context/VRAM limits, resulting in inference crashes. |

## 📊 Dataset & Experimental Setup
- **Source Material**: Research paper, mathematical supplement, two simulation notebooks (`.ipynb`), and two Python utility scripts (`.py`).
- **Context Management**: Original notebooks contained embedded visualizations that inflated token count. Graphs were stripped to maintain a manageable context window of ~110k tokens, which expanded to ~160k tokens with system/user prompts.
- **Prompting Strategy**: Two-stage evaluation([actual prompts written here](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/testing_prompts.md)):
  1. **Initial Prompt**: Identify learning dynamics per notebook and map subfunctions to theoretical descriptions.
  2. **Follow-Up Prompt**: Expand to exhaustive variable-level breakdown, explicitly distinguishing paper-stated mechanisms from code-level implementation details.
- **Quality Assurance**: All model outputs were reviewed and corrected using **Claude Sonnet 4.6**. The correction lineage is preserved using standardized tags (`[⚠ CORRECTED]`, `[➕ ADDED]`, `[❌ PRIOR ERROR]`).
- **Last Pass**: After reviewing all of the models, it was clear that Qwen 3.6 35B A3B was the top performer. So I gave it one more go with better temperture, top k, top p, and two specific followup prompts.

## 🔍 Key Findings
- **Rapid Capability Growth**: Local models have progressed from struggling with this task to producing highly accurate, bidirectional code-paper mappings.
- **Persistent Failure Modes**: The best small local models occasionally misattribute legacy/unused functions as active, conflate initialization steps with runtime dynamics, and miss subtle numerical implementation details (e.g., forgetting-before-reinforcement ordering, cascade rounding, epsilon sampling).
- **Definitive Reference**: A complete code-paper mapping ([definitive_mapping.md](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/definitive_mapping.md)) is provided as the highest-accuracy reference available. It is a significant improvement over Claude Sonnet 4.6's first attempt at code-paper mapping; this improvement was made possible by having assesed the local models and was generated by an instruction to construct a mapping based on its knowledge of the code, paper, and all of the insight it gained from going over the local model's attempts at code-paper mapping. It illustrates the substantive benefits that can be gained from giving large frontier models access to the work of smaller locally run models. While likely the best mapping generated, it was produced by Claude Sonnet 4.6 and should be treated as a strong reference rather than infallible ground truth.

## 📁 Repository Structure
The table below maps each model's iterative outputs to their respective corrected versions. All files reside in the repository root.

| File | Description | Original Output | Corrected Output |
|:---|:---|:---|:---|
| [`ACTUALLY_definitive_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/ACTUALLY_definitive_mapping.md) | Maybe it's not *actually* definitive, but this amended mapping captures an extremely important method the replicator code used to avoid a statistical bias. | N/A | N/A |
| [`opus_failure.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/opus_failure.md) | Claude Opus 4.7 failing to identify an essential component of the code for avoiding a statistical bias | N/A | N/A |
| [`qwen_correction.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen_correction.md) | Prompts and Qwen 3.6 35B A3B's responses used to create the ACTUALLY difinitive mapping | N/A | N/A |
| [`definitive_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/definitive_mapping.md) | Comprehensive mapping generated by Claude Sonnet 4.6 | N/A | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/definitive_mapping.md) |
| [`qwen36_35B_final_attempt_code_paper_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_final_attempt_code_paper_mapping.md) | Qwen 3.6 35B final attempt with two follow up prompts | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_final_attempt_code_paper_mapping.md) | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_final_corrected_by_claude_paper_mapping.md) |
| [`claude_assesment_of_final_qwen35b_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/claude_assesment_of_final_qwen35b_mapping.md) | Claude's assessment of final Qwen 3.6 35B A3B attempt | N/A | N/A |
| [`qwen36_35B_second_attempt_code_paper_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_second_attempt_code_paper_mapping.md) | Qwen 3.6 35B iterative attempt | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_second_attempt_code_paper_mapping.md) | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_35B_corrected_by_claude_paper_mapping.md) |
| [`qwen36_27B_third_attempt_code_paper_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_27B_third_attempt_code_paper_mapping.md) | Qwen 3.6 27B attempt | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_27B_third_attempt_code_paper_mapping.md) | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/qwen36_27B_corrected_by_claude_paper_mapping.md) |
| [`Gemma4_26B_A4B_fourth_attempt_code_paper_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/Gemma4_26B_A4B_fourth_attempt_code_paper_mapping.md) | Gemma 4 26B initial attempt | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/Gemma4_26B_A4B_fourth_attempt_code_paper_mapping.md) | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/gemma_26B_corrected_by_claude_paper_mapping.md) |
| [`Nemotron_30B_fifth_attempt_code_paper_mapping.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/Nemotron_30B_fifth_attempt_code_paper_mapping.md) | Nemotron initial attempt | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/Nemotron_30B_fifth_attempt_code_paper_mapping.md) | [View](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/nemotron_30B_corrected_by_claude_paper_mapping.md) |
| [`testing_prompts.md`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/testing_prompts.md) | Prompts given to models on how they should construct the code-paper mapping | N/A | N/A |
| [`FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/cFNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py) | Functions called by reinforcment learning notebook | N/A | N/A |
| [`FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py) | Functions called by replicator dynamics notebook | N/A | N/A |
| [`genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34_short.ipynb`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34_short.ipynb) | Notebook for reinforcement learning code | N/A | N/A |
| [`genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4_short.ipynb`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4_short.ipynb) | Notebook for replicator dynamics code| N/A | N/A |
| [`Model_of_social_identity_typologies__take_2__unblinded_.pdf`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/Model_of_social_identity_typologies__take_2__unblinded_.pdf) | Research paper fed to the models | N/A | N/A |
| [`supplement_A_78.pdf`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/supplement_A_78.pdf) | Paper supplement with only Appendix B (used for first pass with local models to keep context size small) | N/A | N/A |
| [`supplement_A.pdf`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/supplement_A.pdf) | Paper supplement (full supplement was given to Claude and the final pass of Qwen 3.6 35B A3B) | N/A | N/A |
| [`full_claude_sonnet46_chat_data.txt`](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/full_claude_sonnet46_chat_data.txt) | Claude project log & assessment metadata | N/A | N/A |

## 📝 Assessment & Methodology Notes
- **Evaluation Pipeline**: Model outputs were evaluated using **Claude Sonnet 4.6**, which identified structural errors, corrected factual inaccuracies, and standardized formatting using explicit correction tags.
- **Human Review**: I performed a quick review of the assessment to confirm its general accuracy and logical consistency, but did not scrutinize it in exhaustive detail. The corrected documents should be read as highly reliable references, but users are encouraged to cross-check against the source code/paper when possible.
- **Local Model Handicaps**: Other than the final Qwen 3.6 35B A3B mapping, I used llama.cpp's default temp, top p, and top k values; I also used an abbridged supplement to keep model context smaller. Finally, Claude often ``corrects'' the local model's naming of the .ipynb notebooks as not having '_short' in the name. This is not a real error on the local models' part. They were given notebooks with that in the name, because after I had started testing, I realized that the notebooks had some unneccarry plots in them that were causing context size to blow up (790k vs 160k). The notebooks without the plots in them were saved with '_short' at the end of their name.
- **Definitive Mapping**: The `[definitive_mapping.md](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main/definitive_mapping.md)` file represents the most complete and accurate mapping generated in this project. However, it is a product of Claude Sonnet 4.6 and may still contain minor oversights or interpretations. It is provided as a best-effort reference rather than a verified ground truth.

## 🔗 References
- **Research Paper**: [The Evolution of Identity Signals for Coordination in Diverse Societies](https://doi.org/10.1017/ehs.2026.10037)
- **Repository**: [https://github.com/nathanlgabriel/paper_code_mapping_assessment](https://github.com/nathanlgabriel/paper_code_mapping_assessment/tree/main)

---
*Last updated: `May 10th 2026` | Frameworks: Qwen, Gemma, Nemotron | Context Window: ~160k tokens used by each model*
