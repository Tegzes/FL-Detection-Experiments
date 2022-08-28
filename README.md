# FL-Detection-Experiments

Documenting the experiments done for Figurative Language Detection task.

Trying different Models on sarcasm detection datasets. 

Datasets used: 
- iSarcasm 2022 (Twitter based)
- Ghosh (Twitter based)
- Ptacek (Twitter based)
- SARC (Reddit based)

Models used:
- Baseline trasnformers: BERT, RoBERTa, BerTweet
- Modified models: BERT/RoBERTa/BerTweet + LSTM/RCNN

Moreover, the datasets were augmented using NLPAUG:
- random word insertions
- words replaced with synonyms
- random words deletion

Additionally experimented with:

GloVe embeddins + LSTM/Attention LSTM/CNN models
