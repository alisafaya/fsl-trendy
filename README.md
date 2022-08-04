# Trendy: Few Shot Learning for Topic Classification on Social Media

The rapid growth of social media has created new opportunities and challenges for organizations seeking to understand and track public opinion on current events and topics of interest. The ability to detect and follow trends in social media content in near-real-time would be a valuable asset for organizations seeking to understand public opinion and sentiment. However, traditional machine learning methods are not well-suited for this task, as they require a large number of training data samples to achieve good performance.

Few-shot learning is a training method that is well-suited for learning from a small number of training examples. 
Regardless of the underlying architecture, Few-shot Learning methods have shown good results for text classification tasks. These architectures could rely on non-pretrained text classification methods such as Convolutional Neural Networks, or they can utilize pre-trained text encoders as well. We develop a method for tracking trends in social media content using few-shot learning. 

We label small sets of tweets as being about a particular topic or trend (e.g., "Trump", "Syrian Refugees", etc.). Then, using these sets, we train a model to recognize if a given tweet belongs to a particular topic or not, in a way similar to training a similarity scorer. 

As a result, given a set of tweets from one topic, our model is able to detect whether a new tweet belongs to that particular topic or not, without special finetuning for this topic. This is the main feature of our system, as it doesn't need retraining each time a new trend appears. It will be enough to label a small number of tweets (8 to 16 tweets) that represent this topic, and the model will be able to utilize this set in inference-time to detect whether a given tweet belongs to that topic or not. This methodology could be considered similar to Entailment Detection in Natural Language Inference, or maybe similar to Zero-shot prompting as used in GPT-3.

## Dataset

We collect trend tweets, with 16 tweets per topic. A trend tweet is a tweet that belongs to some trend on twitter. Our dataset has 18 different topics, with at 16 tweets per topic. 
These topics are split into `training/validation/test = 10/4/4`.

During the collection of these topics the following aspects will be taken into consideration:

- The trends have a diversity of topics (sports, politics, general…)
- The annotation contain both fine-grained topics (Syrian refugees in Turkey) and wide general topics (Climate Change) as well.

