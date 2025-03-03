#%% --------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from datasets import load_dataset

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset('yelp_polarity')
train_data = dataset['test']
labels = train_data['label']

#%% --------------------------------------------------------------------------------------------------------------------
plt.hist(labels, bins=[-0.5, 0.5, 1.5], edgecolor='black')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.title('Label Distribution in Yelp Polarity (Train)')
plt.show()