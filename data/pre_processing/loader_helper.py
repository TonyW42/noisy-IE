import torch


class WNUTDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings
        self.labels = labels
        self.model_names = model_names

    def __getitem__(self, idx):
        result = {}
        for encoding, label, model_name in zip(
            self.encodings, self.labels, self.model_names
        ):
            item = {key: torch.tensor(val[idx]) for key, val in encoding.items()}
            item["labels"] = torch.tensor(label[idx])
            result[model_name] = item
        return result

    def __len__(self):
        return len(self.labels[0])


## SST data for
class SSTDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings
        self.model_names = model_names

    def __getitem__(self, idx):
        result = {}
        for encoding, model_name in zip(self.encodings, self.model_names):
            item = {key: torch.tensor(val[idx]) for key, val in encoding.items()}
            item["labels"] = item["input_ids"]
            result[model_name] = item
        return result

    def __len__(self):
        return len(self.encodings[0]["input_ids"])  ## TODO HERE!


class BookWikiDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings
        self.model_names = model_names

    def __getitem__(self, idx):
        """{
            'char':  {'input_ids': [bs, seq_len, emb_size], 'att_mask': [bs, seq_len, emb_size]}
        }
        """
        return self.encodings[idx]

    def __len__(self):
        return len(self.encodings)
