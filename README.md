# README
## Purpose
This git repository is a record of the work that went into my dissertation for the completion of my degree.

## Process
- collocations.py
    - Load pickle files that have a json file match
    - For each pickle file, filter out retweets and tweets which have no hashtags
    - For the remaining tweets in each file, generate the sorted combinations of collocations to create a table of undirected edges
    - During this, track the appearances of each individual hashtag in the 2-collocation set
- edges_filter.py
    - For the data found, reprocess via a simple loop filtering out nodes and their edges which have very low values
    - Without this extra step, gephi is very prone to crashing and corrupting its own directory
- gephi
    - Load the filtered appearances as a node table, and the filtered collocations as an edge table
    - Filter to less nodes to get more focused groups
    - Run modularity on whole graph to determine groups
    - Run OpenOrd layout algorithm
    - Export data back out with modularities included
- modularities_analysis.py
    - Load the files back in with modularities
    - Process sentiment and toxicity values for any tweet containing a hashtag that appears in the modularity focused on Lula or Bolsonaro
    - With these criteria, we have reduced the amount of tweets to process from 417 million to just under 1 million.
- gephi part 2
    - Reload the newly processed data with toxicity and sentiment scores
    - Analyse alongside the botscore
- writing and plotting
    - Plot graphs that are interesting
    - Analyse interesting graphs


## Acknowledgements
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
