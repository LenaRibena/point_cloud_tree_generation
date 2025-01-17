class Normalizer():
    def __init__(self, data):
        self.data_min = data.min(dim=-1, keepdim=True).values
        self.data_max = data.max(dim=-1, keepdim=True).values
        self.data_range = self.data_max - self.data_min

        # Avoid division by zero
        self.data_range[self.data_range == 0] = 1.0

        # Renormalization parameters
        self.range = self.data_range.mean()
        self.min = self.data_min.mean()

    def normalize(self, data):
        return (data - self.data_min) / self.data_range
    
    def renormalize(self, data):
        return data * self.range + self.min