
1. Ro Wiki : The PT variants of the model where lower on all 3 types of models i am comparing. For normal perxplexity especially Llama3.2-1B and Gemma-3-1B-PT had significantrly lower scores, this is unormalized perplexity but I at least the PT/IT models are using same tokenizer so the discrepancy is real. Byte perplexity shows not such a big discrepancy, but still the 2 models I mentioned had lowest score.


2. Ro ARC Challenge : Scores are very close one to another (around 0.3), even the Qwen1.5B model that is 50% larger had copmarable scores. Gemma3-IT had highest score at 0.32. For qwen and gemma the IT models had better scores. This is somewhat expected as ARC is now an instruction following task of QA, while wiki was the opposite.

3. Arc Challenge : Pretrain better for qwen and gemma. In this case qwen1.5B is significantly better than the other models.

4. Ro Belebele : Reading understanding and answering questions based on that : Qwen models are significantly better, the 1B models are more comparable, with IT models being significantly bettter ( the PT was same as random guessing for gemma and llama.

5. Ro Winogrande : Comparable performance between all models, max score is only 0.53 (where 0.5 is random).

6. Winogrande : Significantly better than the Ro version, getting scores up to 0.6. 

