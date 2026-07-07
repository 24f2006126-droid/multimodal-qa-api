import os
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
from io import BytesIO

import google.generativeai as genai


app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure Gemini
genai.configure(
    api_key=os.environ["GEMINI_API_KEY"]
)

model = genai.GenerativeModel(
    "gemini-2.5-pro"
)


class ImageQuestion(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
async def answer_image(payload: ImageQuestion):

    try:
        image_bytes = base64.b64decode(
            payload.image_base64
        )

        image = Image.open(
            BytesIO(image_bytes)
        )

        prompt = f"""
You are a high-accuracy document OCR and visual question answering system.

The image may contain:
- invoices
- receipts
- tables
- charts
- pie charts
- academic documents

First read all visible text carefully.
Pay special attention to:
- totals
- subtotals
- tax values
- percentages
- category labels
- numbers in tables

Question:
{payload.question}

Instructions:
- Answer only from information visible in the image.
- Return ONLY the final answer.
- No explanation.
- No sentences.
- If the answer is numeric, output only the number.
- Do not include currency symbols.
- Do not include units.
- Keep decimal points exactly.
"""
        response = model.generate_content(
            [
                prompt,
                image
            ]
        )

        answer = response.text.strip()

        # Ensure string output
        return {
            "answer": str(answer)
        }

    except Exception as e:
        return {
            "answer": ""
        }
