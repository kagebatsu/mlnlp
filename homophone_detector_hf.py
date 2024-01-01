#pip install torch
#pip install transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

#MODEL/TOKENIZER
#Start by loading a pre-trained language model from Hugging Face's Transformers library.
model_name = "bert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

#DATA
#Prepare data with examples of correct and incorrect usage of words. 
sentences = ["Their dog is friendly.", "Their going to the park.", "The cat is over there.", "Their is nothing here."]

#Label each token in the sentences to indicate whether it's correct or not.
# 0 - incorrect
# 1 - correct
labels = [[1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1]]

#TOKENIZE 
#tokenize sentence data
tokenized_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
#print(tokenized_inputs)

#TRAINING LOOP
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()
num_epochs = 1

for epoch in range(num_epochs):
    outputs = model(**tokenized_inputs)
    loss = loss_fn(outputs.logits.view(-1, 2), torch.tensor(labels).view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#INFERENCE
new_sentence = "Their going to the store."
tokenized_input = tokenizer(new_sentence, return_tensors="pt")
output = model(**tokenized_input)
predictions = torch.argmax(output.logits, dim=2).squeeze().tolist()

# Decode predictions
decoded_predictions = [tokenizer.decode(token) for token in predictions]
print(decoded_predictions)