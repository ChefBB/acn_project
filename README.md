This repository contains the code developed for a network-driven study of language as part of the *Advanced Laboratory of Complex Network Analysis* course offered by the University of Pisa (a.a. 2025/26).

The code is divided into two parts:

1. **Data understanding & preparation**
 - The folder ```data_understanding``` contains simple insights about the data;
 - ```preprocessing``` contains the correction of OCR mistakes (from ```year_clean``` to ```ocr_clean```) and the subsequent lemmatization (from ```ocr_clean``` to ```lemmas```);
 - ```embedding``` contains the embedding models creation and the subsequent word embeddings (from ```lemmas``` to ```word_embeddings_all```); 
 - ```edges_analysis``` contains a post-processing strategy and some analyses for the similarity cutoff. This code provides the first baseline to build the networks (from ```embedding``` to ```edges```).

2. **Network Analysis**
