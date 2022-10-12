# End to End Spoken Language Understanding (SLU)
SLU models predict the domain-intent-slot targets directly from speech using a single model instead of the traditional ASR + NLU system.
## SLU Model
This repo implements end to end (E2E) SLU models using two different architectures:
* Seq2Seq architecture using Wav2Vec as feature extractor in the encoder followed by a GRU decoder
* Implementation of SLU task using Transformer. It will also provide the ASR transcript (**not yet implemented**)
## DataSet
Fluent Speech Commands
## Example
Look at the notebok `SLU-Wav2Vec-TF.ipynb` for a trained example. Due to size of dataset, the example runs using a tiny sub-sample of the dataset. 
