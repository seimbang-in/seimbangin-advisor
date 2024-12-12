from google.cloud import aiplatform
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
import re

load_dotenv()

app = FastAPI()

credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
)

def predict_vertex_ai(project_id, endpoint_id, region, instances):
    try:
        aiplatform.init(project=project_id, location=region, credentials=credentials)
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
        )
        response = endpoint.predict(instances=instances)
        return response.predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting with Vertex AI: {str(e)}")

def clean_response(response):
    # Coba berbagai pola pencarian response
    patterns = [
        r'### Response:(.*?)(?=\n\n|\Z)',  # Pola pertama
        r'Output:(.*?)(?=\n\n|\Z)',        # Pola kedua yang Anda tunjukkan
        r'Output:\n(.*?)(?=\n\n|\Z)'       # Variasi dengan newline
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            response_text = match.group(1).strip()
            
            # Hapus nomor urut dan bersihkan teks
            cleaned_text = re.sub(r'^\d+\.\s*', '', response_text, flags=re.MULTILINE)
            cleaned_text = cleaned_text.strip()
            
            return cleaned_text
    
    return ""


class FinancialAdviceRequest(BaseModel):
    monthly_income: float
    outcome: float
    debt: float
    saving: float
    risk_management: str
    financial_goals: str

@app.post('/advisor/advice')
def get_advice(request: FinancialAdviceRequest):
    project_id = os.getenv("GCP_PROJECT_ID")
    endpoint_id = os.getenv("GCP_ENDPOINT_ID")
    region = os.getenv("GCP_REGION")

    prompt = f"""
    ### Instruction:
    Based on the financial information provided below, give a personalized financial advice in a well-structured and detailed paragraph. Avoid using numbered lists or bullet points. Write in a friendly and professional tone.

    ### Financial Information:
    Income: Rp.{request.monthly_income:,}  
    Outcome: Rp.{request.outcome:,}  
    Debt: Rp.{request.debt:,}  
    Current Savings: Rp.{request.saving:,}  
    Risk Management: {request.risk_management}  
    Financial Goals: {request.financial_goals}  

    ### Response:
    """

    instances = [
        {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7
        }
    ]

    predictions = predict_vertex_ai(project_id, endpoint_id, region, instances)

    if not predictions or len(predictions) == 0:
        raise HTTPException(status_code=500, detail="No predictions returned from the model.")

    raw_response = predictions[0]
    cleaned_response = clean_response(raw_response)

    return {
        "success": True,
        "financial_advice": cleaned_response
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
