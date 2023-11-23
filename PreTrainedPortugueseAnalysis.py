from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/bertweet-pt-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("pysentimiento/bertweet-pt-sentiment")

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions

# Example usage
text = "We could involve the medical order and the nursing order in the identification of their professionals who are in quarantine and available to do teleconsultations."
sentiment = predict_sentiment(text)
print('English scores for first description: ' + str(sentiment))

textpt = "Poderíamos envolver a ordem dos médicos e a ordem dos enfermeiros na identificação dos seus profissionais que estão de quarentena e disponíveis para fazer teleconsultas."
sentimentpt = predict_sentiment(textpt)
print('Resultados da descrição em Português: ' + str(sentimentpt))


