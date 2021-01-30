# websiteClassification
Classifying websites without labels This project is an approach for labeling websites based on their content. Labels were not provided so the top X words gathered from the sites were passed through an autoencoder. Then the results of this encoding were used with K-means clustering to generate synthetic labels. Then a Naive Bayes approach and a Neural Network approach were applied to test the classification accuracy.

Task #1: Entity Extraction and Processing
- this task was completed using Beautiful Soup
- the data was loaded and the data from the headers was extracted
- next the content was extracted from these headers
- this process was split into several functions which would allow for API-type usage in other projects if needed

Task #2: Classification
- the first step of this process was to extract the top X words from the websites and run those vectors through an Autoencoder
- this step created some sort of mapping of the data that could be used to create labels
- next the encoder portion of the autoencoder was used to create vectors that were then passed to K-Means algorithm
- this step attempted to cluster the vectors in X clusters, thus creating the labels
- once the labels were created, Naive Bayes and a Neural Network was used to classify the sites
- Naive Bayes performed slightly worse than the Neural Network

