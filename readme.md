### Evaluation Module

This module is created to support evaluation of my model implementations. It works in tandem with the lm_harness-ro repo/package. 


This module should support passing it model_name and training_args, as well as a list of tasks and specific ablations for those tasks (and number of values per run), then an inference job would be run that performs the simple_evaluate code.

- The key aspect that this module needs to respect, is to work with hydra ( so have a .yaml file with default config, and a way to overwrite that default file), it also needs to have sh file with examples. This way I can easily run all I need;


### **Selected Romanian Benchmarks for Early Capabilities Testing**

| **Benchmark** | **Type** | **Purpose / Why Chosen** |
|----------------|-----------|---------------------------|
| **ro_belebele** | Multiple-choice reading comprehension | Uses **authentic, non-GPT-translated Romanian text**. Evaluates whether the model can **read, understand, and reason over full Romanian passages**, making it the most direct test of *true comprehension* in context. |
| **ro_wiki** | Cloze / language-modeling (perplexity) | Provides a **low-level linguistic measure** of how well the model captures Romanian grammar and token distributions. **Perplexity** over real Wikipedia text reflects **basic fluency and statistical understanding** of the language. |
| **ro_winogrande** | Cloze formulation (pronoun/coreference reasoning) | Tests **sentence-level linguistic reasoning**—whether the model can resolve meaning and referents correctly in Romanian. Serves as an **early signal of syntactic and semantic understanding**. |
| **ro_mmlu** | Multiple-choice factual / academic reasoning | Assesses whether the model can **apply factual and conceptual knowledge expressed in Romanian**, revealing how well its world knowledge transfers through the language. |

**In short:**  
These four benchmarks together span the spectrum from **surface-level fluency** (ro_wiki) to **deep comprehension and reasoning** (ro_belebele, ro_winogrande, ro_mmlu), giving a comprehensive view of Gemma-3-1B’s Romanian language understanding.
H
