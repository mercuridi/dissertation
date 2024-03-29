# README
## Process
1. collocations.py
1.1. Load pickle files that have a json file match
1.2. For each pickle file, filter out retweets and tweets which have no hashtags
1.3. For the remaining tweets in each file, generate the sorted combinations of collocations to create a table of undirected edges
1.4. During this, track the appearances of each individual hashtag in the 2-collocation set
2. edges_filter.py
2.1. For the data found, reprocess via a simple loop filtering out nodes and their edges which have very low values
2.2. Without this extra step, gephi is very prone to crashing and corrupting its own directory
3. gephi
3.1. Load the filtered appearances as a node table, and the filtered collocations as an edge table
3.2. Filter to less nodes to get more focused groups
3.3. Run modularity on whole graph to determine groups
3.4. Run OpenOrd layout algorithm
## Acknowledgements
### ConvertTweetJsonToParquet.py
ConvertTweetJsonToParquet.py is provided generously by Diogo Pacheco, and has been left unmodified. Sections of Pacheco's code have been used in file XXX; any methods copied or otherwise adapted from Pacheco's work are annotated as such in comments.

### SentiLexP2
SentiLexP2 (Portuguese sentiment analysis lexicon) generously provided open-source as a product of Carvalho, Paula; Silva, Mário J at https://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3.
There seemed to be an error in line 5604: `ponto fraco.PoS=N;TG=HUM:N2;POL:N0=-3;ANOT=MAN`
`HUM:N2` is not a valid target. Because the `POL` value is `POL:N0`, I assumed a typo: `N2 -> N0`.

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


## Notes
### Modularities analysis:
Current estimate now that everything is working, 60 hours to process the data.
Currently working:
- Loop that actually processes everything
- Sentiment calculation
- Toxicity calculation
- Modularities included
- Collating calculated data into final csv
- Writing final csv
Currently, it takes about 7 minutes to process about 2000 entries. That means, for 1m~ entries, we're looking at 60 hours (1,000,000/2,000)*7/60
Will look into multithreading python for quicker loads and a better spread on the gpu.