#########################################################
#####
###   Creating Document Term Matrix for Disaster Tweets
#
#########################################################
setwd("~/Documents/CS 450/CS450_group_project")

library(tidyverse)
library(tm)
library(qdap)
library(tidytext)

data <- read_csv("nlp-getting-started/train.csv")

rm_stopwords(data$text[1], tm::stopwords("english"))


data <- data %>%
  mutate(text = removePunctuation(data$text))

tweet_Corpus <- Corpus(VectorSource(as.vector(data$text)))

# Remove Stopwords 
tweet_Corpus <- tm_map(tweet_Corpus, removeWords, stopwords("english"))
# Remove Numbers 
tweet_Corpus <- tm_map(tweet_Corpus, content_transformer(removeNumbers))
# Make everything lowercase
tweet_Corpus <- tm_map(tweet_Corpus,  content_transformer(tolower)) 
# Stem words
tweet_Corpus  <- tm_map(tweet_Corpus, content_transformer(stemDocument), language = "english")

# create document-term matrix
tweet_DTM <- DocumentTermMatrix(tweet_Corpus, control = list(wordLengths = c(2, Inf)))

big_chungus <- as.data.frame(as.matrix(tweet_DTM), stringsAsFactors=False)

big_chungus <- big_chungus %>%
  mutate(data_id = row_number()) %>%
  left_join(data %>%
              select(data_id = id, data_target = target))


write_csv(big_chungus, "document_term_matrix.csv")
