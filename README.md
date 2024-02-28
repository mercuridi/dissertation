# README
## Acknowledgements
### SentiLexP2
SentiLexP2 (Portuguese sentiment analysis lexicon) generously provided open-source as a product of Carvalho, Paula; Silva, Mário J at https://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3.
There seemed to be an error in line 5604: `ponto fraco.PoS=N;TG=HUM:N2;POL:N0=-3;ANOT=MAN`
`HUM:N2` is not a valid target. Because the `POL` value is `POL:N0`, I assumed a typo: `N2 -> N0`.

### ConvertTweetJsonToParquet.py
ConvertTweetJsonToParquet.py is provided generously by Diogo Pacheco, and has been left unmodified. Sections of Pacheco's code have been used in file XXX; any methods copied or otherwise adapted from Pacheco's work are annotated as such in comments.

### ToLD-Br
ToLD-Br (Portuguese hate speech data) sourced via https://hatespeechdata.com:
```
Toxic Language Dataset for Brazilian Portuguese (ToLD-Br)
    Link to publication: https://arxiv.org/abs/2010.04543
    Link to data: https://github.com/JAugusto97/ToLD-Br
    Task description: Multiclass (LGBTQ+phobia, Insult, Xenophobia, Misogyny, Obscene, Racism)
    Details of task: Three annotators per example, demographically diverse selected annotators.
    Size of dataset: 21.000
    Percentage abusive: 44%
    Language: Portuguese
    Level of annotation: Posts
    Platform: Twitter
    Medium: Text
```
The most recent version of ToLD-Br at its Github repository as of 14/02/2024 should be cloned into `/data/ToLD-Br/`. The pre-trained BERT model is also kept locally. These assets are not uploaded to this repository due to their large file size (100MB~ for code and data, 1000MB for BERT model). A link to download the pre-trained model can be found in the README for ToLD-Br.
ToLD-Br's code is licensed under the MIT License, and the dataset itself is licensed under Creative Commons 4.0.

### NLPyPort
NLPyPort is a Portuguese-specific improvement to the tools provided in NLTK, developed by Ferreira, Oliveira, and Rodrigues for SLATE 2019.
It can be found at https://github.com/NLP-CISUC/NLPyPort.