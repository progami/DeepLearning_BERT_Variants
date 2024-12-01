import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the fine-tuned model and tokenizer
model_name = 'albert-base-v2-finetuned-covid-qa'  # Change to your model's name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def answer_question(question, context):
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    # Convert tokens to answer
    answer_tokens = input_ids[0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Test the model
if __name__ == "__main__":
    context = "COVID-19 vaccines are safe and effective. They provide strong protection against serious illness, hospitalization, and death."
    question = "Are COVID-19 vaccines safe?"
    answer = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

