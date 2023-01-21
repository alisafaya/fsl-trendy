import html
import re

import jsonlines
import random
import numpy as np
import torch

from normalizer import TextNormalizer
from tqdm import tqdm

text_normalizer = TextNormalizer()


def preprocess(tweet, remove_hashtags=True):
    # normalize html
    tweet = html.unescape(tweet)

    # remove urls http://t.co/xxxx
    tweet = re.sub(r"https\:\/\/t\.co\/\w+", "http://url.com", tweet)

    # remove @user
    tweet = re.sub(r"@\w+", "@user", tweet)

    # remove #hashtag
    if remove_hashtags:
        tweet = re.sub(r"#\w+", " ", tweet)
    else:
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
            tweet.replace("#", " "),
        )
        tweet = " ".join([m.group(0) for m in matches])

    # normalize punctuations
    tweet = text_normalizer(tweet)
    return tweet


def load_data(filename, remove_hashtags=True):
    # load data from jsonlines file
    with jsonlines.open(filename) as reader:
        topics = {}
        for obj in tqdm(reader, desc="Loading data.."):
            tweet = preprocess(obj["text"], remove_hashtags)
            topic = obj.get("topic", "negative")
            if topic not in topics:
                topics[topic] = []

            topics[topic].append(tweet)

    return topics


def data_iterator(
    topics: list[list],
    negatives: list[tuple],
    batch_size=32,
    topic_size=8,
    shuffle=True,
    shuffle_in_topic=False,
):
    assert batch_size - topic_size > 0
    # create a batch iterator for topics and negatives
    # yields one topic and batch_size - topic_size negative
    # samples per batch
    negatives = np.array(negatives, dtype=object)

    if shuffle:
        random.shuffle(topics)
        np.random.shuffle(negatives)

    # create batches
    for topic in topics:
        # get topic definers and positive samples
        if shuffle_in_topic:
            random.shuffle(topic)

        topic_definers = topic[:topic_size]
        positives = topic[topic_size:]

        # get negatives
        negative_size = batch_size - len(positives)
        negs = np.random.choice(negatives, negative_size, replace=False)

        inputs = np.concatenate([positives, negs])
        labels = np.concatenate([np.ones(len(positives)), np.zeros(negative_size)])

        labels = torch.from_numpy(labels).long()

        # yield batch
        yield topic_definers, inputs, labels


if __name__ == "__main__":

    import sys

    random.seed(42)
    np.random.seed(42)

    # load data
    topics = load_data(sys.argv[1])
    topics = list(topics.values())

    negatives = load_data(sys.argv[2])
    negatives = list(negatives["negative"])

    # create batch iterator
    iterator = data_iterator(topics, negatives, batch_size=16, topic_size=8)

    for topic_definers, inputs, labels in iterator:
        print("Topic definers: \n", "\n".join(topic_definers))
        print("Inputs: \n", "\n".join(inputs))
        print("Labels: \n", labels)
        break
