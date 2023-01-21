import sys
import random
import numpy as np
import torch

from dataloaders import load_data
from models import CosineSimilarityScorer

from tqdm import trange

if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    model = CosineSimilarityScorer(aggregator="none")

    topic_size = 8

    topics = load_data(sys.argv[1], remove_hashtags=False)
    topic_definers = list(topics[sys.argv[2]])[:topic_size]

    print("Number of topic definers: ", len(topic_definers))
    print("Topic size: ", topic_size)
    print("Topic definers: ", topic_definers)
    print("=====================================")

    negatives = load_data(sys.argv[3], remove_hashtags=False)
    negatives = np.array(list(negatives["negative"]), dtype=object)

    print("Number of negatives: ", len(negatives))
    print("Negatives: ", negatives[:10])
    print("=====================================")
    
    best_threshold = 0.375
    best_ratio_threshold = 3.0 / 8.0

    with torch.no_grad():
        positive_indices = []
        for b in trange(0, len(negatives), 1000):
            indices = list(range(b, min(b + 1000, len(negatives)))) # b:b + 1000
            inputs = negatives[indices]
            scores = model(topic_definers, inputs)
            predictions = (scores > best_threshold).float().mean(dim=1) > best_ratio_threshold
            positive_indices.extend(
                [i for i, p in zip(indices, predictions) if p]
            )

    print("Number of positive indices: ", len(positive_indices))
    print("Number of negative indices: ", len(negatives) - len(positive_indices))

    import jsonlines
    positive_indices = set(positive_indices)

    writer = jsonlines.open(sys.argv[4], "w")
    for i, o in enumerate(jsonlines.open(sys.argv[3])):
        if i in positive_indices:
            writer.write(o)
    writer.close()
