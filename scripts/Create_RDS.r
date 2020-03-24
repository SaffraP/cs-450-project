pacman::p_load(tidyverse,tidytext, ngram, textdata, sentimentr, GGally)

train <- read_csv("nlp-getting-started/train.csv") %>%
  mutate(char_count = nchar(text),
         word_count = sapply(strsplit(text, " "), length),
         hashtag_count = str_count(text, "#"),
         number_count = str_count(text, "[0-9]"),
         capital_count = str_count(text, "[A-Z]"),
         contains_link = as.character(case_when(str_count(text, "http") >= 1 ~ 1,
                                   TRUE                         ~ 0)),
         breaking = as.character(str_count(text, "BREAKING")))

b <- sentiment(train$text)
d <- b %>% 
  group_by(element_id) %>% 
  summarize(sent_score = mean(sentiment))

train <- train %>% 
  cbind(sent_score = d$sent_score)



train %>% 
  select(-text, -keyword, -location) %>% 
  ggpairs(aes(col = as.character(target)))


