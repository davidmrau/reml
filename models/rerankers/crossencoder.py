class CrossEncoder:

    def __init__(self):
        pass


    def collate_fn(self, examples):
        doc = [e["claim"] for e in examples]
        labels = torch.ones(len(examples))
        input_tokenized = tokenizer(question, doc, padding=True, truncation='only_second', max_length=self.tokenizer.max_length, return_tensors='pt')
        return {
                "inputs": input_tokenized,
                "labels": labels,
                }
