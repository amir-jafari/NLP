#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
#%% --------------------------------------------------------------------------------------------------------------------
class YelpPolarityLoader:
    """
    Loads the Yelp Polarity dataset from a local directory.
    """
    def __init__(self, data_dir: str = "./yelp_polarity"):
        self.data_dir = data_dir
        self.train = None
        self.test = None
    def load(self):
        """
        Reads the dataset from disk and sets self.train/self.test.
        Returns:
            train, test splits
        """
        ds = load_dataset("yelp_polarity", data_dir=self.data_dir)
        self.train = ds["train"]
        self.test  = ds["test"]
        print(f"Loaded {len(self.train)} train examples and {len(self.test)} test examples.")
        return self.train, self.test
#%% --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    loader = YelpPolarityLoader("./yelp_polarity")
    train, test = loader.load()
