# Fake-News-Finder
Detects whether a news or article is fake by lexicographical character checking and tokenization. The project tokenizes the words present in the news and checks the polarity of each letter and it's usage following regex rules. These regex rules convert the word sink into a DFA and the states are observed. If the DFA states match with a real news data, then it is considered real. Else it will try to recapture information from the next line. If the entire data is fake then it is declared as a fake news.
