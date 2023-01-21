import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer


class AttentionScorer(nn.Module):
    """
    This model encodes the given sentences into vectors using sentence embedding
    models then it uses the attention mechanism to produce a hidden vector repr-
    esenting similarity between the input sentence and the topic definers, then 
    it applies a linear layer to produce the logits.
    """

    def __init__(self, model="all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.smodel = SentenceTransformer(model)
        self.hidden_dim = self.smodel.get_sentence_embedding_dimension()

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, topic_definers: list[str], inputs: list[str]):

        assert len(topic_definers) > 0
        assert len(inputs) > 0

        # Encode sentences using sentence embedding models
        topic_sembeds = self.smodel.encode(topic_definers, convert_to_tensor=True)
        # topic_sembeds: (topic_size, sembed_dim)

        inputs_sembeds = self.smodel.encode(inputs, convert_to_tensor=True)
        # inputs_sembeds: (input_size, sembed_dim)

        # Apply Attention mechanism
        attn_output, attn_weights = self.attn(
            query=self.query_proj(inputs_sembeds),
            key=self.key_proj(topic_sembeds),
            value=self.value_proj(topic_sembeds),
        )

        # attn_output: (input_size, sembed_dim)
        # attn_weights: (input_size, topic_size)

        # Apply linear layer
        logits = self.fc(attn_output)
        # logits: (input_size, 2)

        return logits, attn_weights

    def loss(self, topic_definers, inputs, labels):
        # Calculate logits
        logits, _ = self.forward(topic_definers, inputs)

        # Calculate loss
        loss = F.cross_entropy(logits, labels, reduction="sum")
        return loss


aggregate = {
    "mean": lambda x: x.mean(dim=0),
    "max": lambda x: x.max(dim=0)[0],
    "min": lambda x: x.min(dim=0)[0],
    "none": lambda x: x,
}


class CosineSimilarityScorer(nn.Module):
    def __init__(self, model="all-MiniLM-L6-v2", aggregator="mean") -> None:
        super().__init__()
        self.smodel = SentenceTransformer(model)
        self.hidden_dim = self.smodel.get_sentence_embedding_dimension()
        self.aggregator = aggregate[aggregator]

    def forward(self, topic_definers: list[str], inputs: list[str]):

        assert len(topic_definers) > 0
        assert len(inputs) > 0

        # Encode sentences using sentence embedding models
        topic_sembeds = self.smodel.encode(topic_definers, convert_to_tensor=True)
        # topic_sembeds: (topic_size, sembed_dim)

        inputs_sembeds = self.smodel.encode(inputs, convert_to_tensor=True)
        # inputs_sembeds: (input_size, sembed_dim)

        # Calculate normalized cosine similarity
        sims = torch.cosine_similarity(
            topic_sembeds.unsqueeze(1), inputs_sembeds.unsqueeze(0), dim=-1
        )
        # sims: (topic_size, input_size)

        # Calculate logits
        logits = self.aggregator(sims).T
        # logits: (input_size,)

        return logits


if __name__ == "__main__":
    from dataloaders import load_data, data_iterator

    import sys
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    # load data
    topics = load_data(sys.argv[1], remove_hashtags=False)
    topics = list(topics.values())

    negatives = load_data(sys.argv[2], remove_hashtags=False)
    negatives = list(negatives["negative"])

    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    model = CosineSimilarityScorer(aggregator="none")
    iterator = data_iterator(
        topics,
        negatives,
        batch_size=10000,
        topic_size=8,
        shuffle=True,
        shuffle_in_topic=False,
    )

    with torch.no_grad():
        batches = []
        golds = []
        for topic_definers, inputs, labels in iterator:
            scores = model(topic_definers, inputs)
            batches.append(scores)
            golds.extend(labels.cpu().tolist())

    batches = torch.cat(batches, dim=0)

    # Finding the best threshold
    from sklearn.metrics import precision_recall_curve
    for sthreshold in np.linspace(0.1, 0.5, 10):
        print("Similarity Threshold: ", sthreshold)
        precision, recall, thresholds = precision_recall_curve(
            golds, (batches > sthreshold).float().mean(dim=1).cpu().tolist()
        )
        f1_scores = 2 * recall * precision / (recall + precision)
        print("Best ratio threshold: ", thresholds[np.argmax(f1_scores)])
        print("Best F1-Score: ", np.max(f1_scores))

    best_threshold = 0.375
    best_ratio_threshold = 3.0 / 8.0

    predictions = (batches > best_threshold).float().mean(dim=1) > best_ratio_threshold

    print("===" * 10)
    from sklearn.metrics import classification_report

    print(classification_report(golds, predictions, digits=4))
    print("===" * 10)
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(golds, predictions, labels=[0, 1]))

    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = plt.subplot()
    sns.heatmap(
        confusion_matrix(golds, predictions, labels=[0, 1]), annot=True, fmt="g", ax=ax
    )
    # annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Trends", "Others"])
    ax.yaxis.set_ticklabels(["Others", "Trends"])

    plt.show()
