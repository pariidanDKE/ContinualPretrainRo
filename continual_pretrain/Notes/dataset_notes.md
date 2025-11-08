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
