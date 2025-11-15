## Evalaution Datasets

In OpenRo LLM, they did pretraining on CulturaX ( 5,10 and 20% of the romanian langauge data). The CulturaX romanina amounted to about 40B tokens. So, they trained variantes of 2B, 4B and 8B. From their findings for Llama2-7B and Mistral-7B, the difference from 5 to 20% is non-existent (at least from the benchmarks they tested on). 
These benchmarks are ARC, MMLU, WinoGrad, HellaSwag, GSM8K, TQA. They translated ARC,HS and MMLU and TQA with GPT4. While Winograde and GMS8K are translated by them (also neural appraoch, more specialised to translating with stuff like seq2seq and additional rulesets prob). Looking at their results, semeingly the biggest spikes are from these manually translated datasets (in table 1).

#### GSM8K
Description : Grade school math.
Size : 8k (duh)

Example :   
Q: Betty economisește bani pentru un nou portofel care costă 100 de dolari. Betty are doar jumătate din banii de care are nevoie. Părinții ei au decis să-i dea 15 dolari în acest scop, iar bunicii ei de două ori mai mult decât părinții ei. Câți bani mai are nevoie Betty pentru a cumpăra portofelul?
A: La început, Betty are doar 100 / 2 = $<<100/2=50>>50. Bunicii lui Betty i-au dat 15 * 2 = $<<15*2=30>>30. Asta înseamnă că Betty are nevoie de 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 mai mult. ### 5


#### Winograde
Description : Dataset meant to estalbihs if model can next-token predict a corect word, given previous context ( i.e talking about a specific object, and making a reference to in later sentences)
Size :  29.7K

Q : Paricia a încercat să-i predice Feliciei despre cum să piardă în greutate, în ciuda faptului că _ este supraponderal și are nevoie de dietă.
A: Option1 : Patricia ; Option2 : Victoria


#### MMLU
Size : 15k rows
Description : Makes sure the models is still capable of asnwering multiple tasks ( histroy , computer science , law etc.). So let's say makes sure we do not catastrophic forget by testing multiple tasks.
Altough a lot of the tasks seem to be just memorization from the pretraiend weights, which is not that interesting.

Q: Care dintre următoarele nu este inclus în PIB-ul SUA?

A1: Consumatorii japonezi cumpără mii de CD-uri produse în Statele Unite.
A2: Militarul SUA deschide o nouă bază într-o țară străină cu 1000 de personal militar american.
A3: Un cântăreț pop american susține un concert cu casa închisă la Paris.
A4: ...



#### Decision making notes on Benchmarks

ro_arc_challenge
- count (1k train, 1k test)
- type : MC
- example:  "Cel mai mare corp din sistemul nostru solar este"
ro_belebele:
- type: MC
- count: 900 words
- example : **Passage** : " Medaliatul cu aur la Olimpiadă urma să înoate la Jocurile Commonwealth-ului, în probele de 100 m și 200 m liber, precum și în trei probe de ștafetă, însă condiția sa fizică a fost pusă sub semnul întrebării din cauza plângerilor sale. Nu a putut să ia medicamente...". **Question** : "De ce nu lua medaliatul cu aur la Olimpiadă medicamente pentru durere?" **Answers** : MC
- ro_grammar:
	- **type:** MC
	- **count:** 1.15k rows
	- **example:** " Un sinonim potrivit pentru termenul 'abuziv' este:"; MC: {'a': 'excesiv', 'b': 'abundent', 'c': 'violent', 'd': 'țipător'}
- ro_gsm8k:
	- **type:** FG
	- **count:** (7.4k train, 1k test)
	- **example:** "James decide să alerge 3 sprinturi de 3 ori pe săptămână. Aleargă 60 de metri pe sprint. Câți metri în total aleargă o săptămână?,"
- ro_mmlu:
	- **type**: multiple choice
	- count (train 272, test 12k)
	- example : " Care dintre următoarele nu este inclus în PIB-ul SUA?" 
- ro_wiki:
	- **type**: perplexity assesments
	- count(train 120k, test 779)


### **Selected Romanian Benchmarks and Rationale**

| **Benchmark**     | **Type**                                          | **Purpose / Why Chosen**                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ro_belebele**   | Multiple-choice reading comprehension             | Uses **authentic, non-GPT-translated Romanian text**. Evaluates whether the model can **read, understand, and reason over full Romanian passages**, making it the most direct test of *true comprehension* in context.                 |
| **ro_wiki**       | Cloze / language-modeling (perplexity)            | Provides a **low-level linguistic measure** of how well the model captures Romanian grammar and token distributions. **Perplexity** over real Wikipedia text reflects **basic fluency and statistical understanding** of the language. |
| **ro_winogrande** | Cloze formulation (pronoun/coreference reasoning) | Tests **sentence-level linguistic reasoning**—whether the model can resolve meaning and referents correctly in Romanian. Serves as an **early signal of syntactic and semantic understanding**.                                        |
| **ro_mmlu**       | Multiple-choice factual / academic reasoning      | Assesses whether the model can **apply factual and conceptual knowledge expressed in Romanian**, revealing how well its world knowledge transfers through the language.                                                                |

**In short:**  
These four benchmarks together span the spectrum from **surface-level fluency** (ro_wiki) to **deep comprehension and reasoning** (ro_belebele, ro_winogrande, ro_mmlu), giving a comprehensive view of Gemma-3-1B’s Romanian language understanding.
