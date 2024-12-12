from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import torch

def load_model():
  model_path = './Llama'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.float16,
      device_map="auto",
      low_cpu_mem_usage=True,
      offload_folder="offload"
  ).to(device)
  tokenizer = AutoTokenizer.from_pretrained(
      model_path,
      model_max_length=512
  )
  generator = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      framework="pt"
  )
  return generator

def get_financial_advice(generator,context):
  prompt = f"### Context:\n{context}\n\n### Response:\n"
  generated_text = generator(
      prompt,
      max_length=512,
      num_return_sequences=1,
      temperature=0.7,
      top_k=50,
      top_p=0.9,
      repetition_penalty=1.2
  )
  response = generated_text[0]['generated_text'].split("### Response:\n")[1].strip()
  return response

app = Flask(__name__)
CORS(app)

@app.route('/advisor/advice', methods=['POST'])
def get_advice():
    data = request.json
    context = f""" 
    income : Rp.{data['monthly_income']}
    outcome : Rp.{data['outcome']}
    debt : Rp.{data['debt']}
    current saving : Rp.{data['saving']}
    risk management : {data['risk_management']}
    financial goals : {data['financial_goals']}

    Based on financial situation above, give me a personalized financial advice.
    """
    generator = load_model()
    output_text = get_financial_advice(generator, context)
    return jsonify({
        'success': True,
        'context': context,
        'advice': output_text
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)