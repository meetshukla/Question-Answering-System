from flask import Flask, jsonify, request
from flask import render_template
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

app = Flask(__name__)
models = {
    'NewsQA': {
        'model_name': 'distilbert-base-uncased-distilled-squad',
        'tokenizer_name': 'distilbert-base-uncased',
        'model': torch.load("finalFineTunedModelNewsQA",map_location=torch.device('cpu')),
        'tokenizer': None,
    },
    'Squad': {
        'model_name': 'bert-large-uncased-whole-word-masking-finetuned-squad',
        'tokenizer_name': 'bert-large-uncased',
        'model': torch.load("finalFineTunedModelSquadV2",map_location=torch.device('cpu')),
        'tokenizer': None,
    }
}
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# # Load the fine-tuned modeol
# model = torch.load("finetunedmodelnewsqa",map_location=torch.device('cpu'))
# model.eval()

def predict(model,context,query):

  inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(outputs[1]) + 1 
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    model_name = request.json['model']
    context = request.json['context']
    question = request.json['question']
    model= models[model_name]['model']
    answer=predict(model,context,question)
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5003)
