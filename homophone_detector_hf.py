from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Constants
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128  # Maximum length of tokens

# Sample data (you should replace this with your own dataset)
sentences = [
    "Their going to the park later.",
    "I can't find my keys; did you see them over they're?",
    "There going to be a concert downtown tonight.",
    "We're going to have to reschedule our meeting; their not available then.",
    "I left my umbrella over their by the door.",
    "I heard there bringing pizza for lunch.",
    "The children are playing over they're in the backyard.",
    #"Did you check in they're for the lost keys?",
    #"Their going to regret not attending the event."
    # "I think they're cat is stuck in the tree again.",
    # "There going to announce the winner soon.",
    # "I can't believe they're planning to move to a new city.",
    # "Is their a chance we can reschedule the appointment?",
    # "I left my phone over they're on the charger.",
    # "I hope they're going to enjoy the surprise party.",
    # "I saw them over their waiting for the bus.",
    # "There going to need more volunteers for the project.",
    # "I wonder if their going to be traffic on the way home.",
    # "They're going to the beach for vacation.",
    # "I left my glasses over they're on the nightstand.",
    # "Did you hear about they're new business venture?",
    # "I think I left my jacket over their in the car.",
    # "They're going to love the gift we got them.",
    # "I can't find my wallet; did you see it over they're?",
    # "There going to have a meeting to discuss the project."
]

#labels = [[1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1]]  # 0: incorrect, 1: correct
labels = [
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1]
    # [0, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 0, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [1, 0, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1],
    # [0, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1]
]

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Dataset Class
class HomophoneDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Adjust labels length to match the input_ids
        label.extend([1] * (MAX_LEN - len(label)))  # Padding labels, assuming '1' is the 'correct' label
        label = label[:MAX_LEN]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training
train_dataset = HomophoneDataset(sentences, labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save the model
model.save_pretrained("./homophone_detector_model")

# Function to Test the Model
def test_model(sentences, model, tokenizer):
    """
    Function to test the model.

    Args:
        sentences (list): List of sentences to test.
        model (BertForTokenClassification): Trained model.
        tokenizer (BertTokenizer): Pretrained tokenizer.

    Returns:
        list: Predictions for each sentence.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get the predicted labels
            predicted_labels = torch.argmax(logits, dim=2)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            token_predictions = [p for t, p in zip(tokens, predicted_labels[0].numpy()) if t not in tokenizer.all_special_tokens]

            predictions.append(token_predictions)

    return predictions

# Load the trained model (if not already in memory)
model = BertForTokenClassification.from_pretrained("./homophone_detector_model")  # Load the trained model

# Test Sentences
test_sentences = [
    "They're going to enjoy their vacation.",
    "Their going to enjoy there vacation.",
    "Their planning a surprise party for her.",
    "Is their a chance we can reschedule the appointment?"
]

# Run the test
model_predictions = test_model(test_sentences, model, tokenizer)  # Get model predictions

# Print the predictions
for sentence, prediction in zip(test_sentences, model_predictions):
    """
    Loop through each sentence and its corresponding prediction, and print them.

    Args:
        sentence (str): The test sentence.
        prediction (list): The model's prediction for the sentence.
    """
    print(f"Sentence: {sentence}")
    print(f"Prediction: {prediction}\n")