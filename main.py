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
    "gemini-2.5-flash"
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
You are an image question answering system.

Extract the answer from the image.

Question:
{payload.question}

Rules:
- Return ONLY the answer.
- If the answer is numeric, return only the number.
- Do not include currency symbols.
- Do not include units.
- Do not explain.
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