

import os

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

# Load the data
def plot_trend(df, trend_name):

    times = pd.DatetimeIndex(df.created_at)
    grouped = df.groupby([times.day, times.hour])
    counts = grouped.created_at.count()
    counts = counts.unstack(0)
    counts = counts.fillna(0)

    # Concatenate the columns
    tweet_counts, tweet_day, tweet_hour = [], [], []
    for c in [31, 1, 2, 3, 4]:
        tweet_counts.append(counts[c])
        tweet_day.append([c] * 24)
        tweet_hour.append([0]* 6 + [6]* 6 + [12]* 6 + [18]* 6)

    tweet_counts = np.concatenate(tweet_counts)
    tweet_day = np.concatenate(tweet_day)
    tweet_hour = np.concatenate(tweet_hour)

    tweet_date = pd.to_datetime(
        {
            "year": 2022,
            "month": 8,
            "day": tweet_day,
            "hour": tweet_hour,
        })

    # First day belongs to month 7
    tweet_date[tweet_day == 31] = tweet_date[tweet_day == 31] - pd.Timedelta(days=31)

    spl = sns.lineplot(data=pd.DataFrame({"Date Time": tweet_date, "Tweet Count": tweet_counts}), 
        x="Date Time", 
        y="Tweet Count",
        markers=True,
        legend="full",
        label=trend_name,
        ax=ax
    )

    # Set number of ticks
    spl.xaxis.set_major_locator(plt.MaxNLocator(20))
    spl.yaxis.set_major_locator(plt.MaxNLocator(10))

    # Set the xticks
    spl.set_xticklabels(tweet_date.dt.strftime('%m-%d %I%p').unique().tolist())

    # Remove x axis label
    spl.set(xlabel=None)

    # Rotate the xticks
    plt.setp(spl.get_xticklabels(), rotation=45)


if __name__ == "__main__":
    series = {}
    for f in os.listdir("predicted_trends/"):
        series[f] = pd.read_json("predicted_trends/" + f, lines=True, orient="records")

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Tweet Counts for Four Different Twitter Trends (July - August 2022)")

    for s in series:
        plot_trend(series[s], s.split(".")[0])

    plt.subplots_adjust(bottom=0.2)

    # Save the figure with high resolution
    plt.savefig("trends.png", dpi=300)
    plt.show()