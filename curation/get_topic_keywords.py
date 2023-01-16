import itertools
import json
from string import punctuation

import pandas as pd
import preprocessor as p
import twint as tw

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import unicodedata
import re
import os


stops = set(stopwords.words("english"))


p.set_options(p.OPT.URL, p.OPT.EMOJI)


def preprocess_text(identifier):
    # https://stackoverflow.com/a/29920015/5909675
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
        identifier.replace("#", " "),
    )
    return " ".join([m.group(0) for m in matches])


def strip_accents_and_lowercase(s):
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).lower()
    return re.sub(r"[^a-zA-Z0-9 ]", " ", s)


def get_top_words(tweets, k=5):
    stemmer = PorterStemmer()

    cleaned = (
        tweets.apply(preprocess_text)
        .apply(strip_accents_and_lowercase)
        .apply(word_tokenize)
        .apply(
            lambda y: list(filter(lambda x: not (x in stops or x in punctuation), y))
        )
    )
    counter = {}

    for w in itertools.chain.from_iterable(cleaned):
        stem = stemmer.stem(w)
        if stem in counter:
            counter[stem]["count"] += 1
            counter[stem]["words"].add(w)
        else:
            counter[stem] = {"words": set(), "count": 1}

    for w in counter:
        counter[w]["words"] = list(counter[w]["words"])

    return sorted(counter.items(), key=lambda x: x[1]["count"], reverse=True)[:k]


def search(query):

    c = tw.Config()
    c.Search = " ".join(query)
    c.Hashtags = True
    c.Utc = True
    c.Full_text = True
    c.Retweets = False
    c.Limit = 16
    c.Store_json = True
    c.Output = "queried/" + "_".join(query) + ".json"

    if os.path.exists(c.Output):
        return pd.read_json(c.Output, lines=True)

    print("Searching for", c.Search)
    tw.run.Search(c)
    df = tw.storage.panda.Tweets_df
    return df


def get_queries(word_list):
    queries = []
    for i in range(1, len(word_list) + 1):
        queries += list(itertools.combinations(word_list, i))
    return queries


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: get_topic_keywords.py <json_file>")
        sys.exit(1)

    if not os.path.isdir("topic_queries"):
        os.mkdir("topic_queries")

    if not os.path.isdir("queried"):
        os.mkdir("queried")

    input = sys.argv[1]
    df = pd.read_json(input, lines=True)

    # group by topic and get top words
    topic_keywords = {}
    topic_queries = {}
    query_tweets = {}
    for topic, group in df.groupby("topic"):
        top_words = get_top_words(group["tweet"], k=5)
        topic_keywords[topic] = top_words

        query_set = []
        word2stem_map = {y: x[0] for x in top_words for y in x[1]["words"]}
        words = [y for x in top_words for y in x[1]["words"]]
        for query in get_queries(words):
            stems = [word2stem_map[x] for x in query]
            if max(stems.count(x) for x in stems) > 1:
                continue
            query_set.append(tuple(query))
        topic_queries[topic] = sorted(list(set(query_set)), key=len)

        topic_tweets = []
        for query in topic_queries[topic]:
            # print(topic, query)
            topic_tweets.append(search(query))
        query_tweets[topic] = topic_tweets

    for k in query_tweets:
        query_tweets[k] = pd.concat(query_tweets[k])
        query_tweets[k].to_json(
            f"topic_queries/{k}.jsonl", lines=True, orient="records"
        )

    import ipdb

    ipdb.set_trace()
    sys.exit(0)
