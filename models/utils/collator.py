
from transformers import DataCollatorWithPadding
class DataCollatorWithId(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate the document_id from the other features
        ids = [feature.pop('id') for feature in features]
        # Use the parent DataCollatorWithPadding to handle other features
        batch = super().__call__(features)
        # Add document_id back to the batch without converting to numerical value
        batch['ids'] = ids

        return batch
