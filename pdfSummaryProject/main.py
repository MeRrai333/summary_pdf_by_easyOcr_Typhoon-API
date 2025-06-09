from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pdfSummaryProject.services.healthCheckService import check_service_health
from pdfSummaryProject.services.easyOCRService import EasyOCRService
from pdfSummaryProject.services.upload_file import UploadFileService
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel, Field
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_service = EasyOCRService(
            languages=['en', 'th'],
            gpu=False,  # Set to True if you have GPU support
            verbose=True,
            quantize=True
        )

uploadFileService = UploadFileService()

@app.get("/health")
def health_check():
    return {"status": check_service_health()}

@app.post("/upload-file-pdf")
async def upload_files_organized(files: List[UploadFile] = File(...)):
    """
    Upload multiple files and automatically organize them into directories by type.
    Videos → uploads/videos/, PDFs → uploads/pdfs/, etc.
    """
    
    try:
        # Prepare file data for the upload service
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append({
                'content': content,
                'filename': file.filename,
                'content_type': file.content_type or 'unknown'
            })
        # Use file upload service to save and organize files
        upload_results = uploadFileService.save_multiple_files(files_data)
        
        return {
            "status": "upload_completed",
            "message": f"Processed {upload_results['total_files']} files",
            "upload_summary": upload_results['summary'],
            "successful_uploads": upload_results['successful_uploads'],
            "failed_uploads": upload_results['failed_uploads'] if upload_results['failed_uploads'] else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


class ExtractTextRequest(BaseModel):
    file_name: str = Field(..., description="Name of the PDF file to process")

@app.post("/extract-text")
async def extract_text_from_pdf(
    req: ExtractTextRequest
):
    """
    Extract text from PDF file
    
    Returns the complete text content as a single string with basic metadata.
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    file_name = req.file_name
    # Validate file
    if not file_name.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with open('./pdfSummaryProject/uploads/doc/'+file_name, 'rb') as f:
            pdf_bytes = f.read()
        
        print('reading...')
        full_text = ocr_service.extract_text_from_pdf(pdf_bytes, dpi=400)
        
        return {
            'result': full_text
        }
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/summary-pdf-file_name")
async def summary_pdf(
    req: ExtractTextRequest
):
    """
    Extract text from PDF file
    
    Returns the complete text content as a single string with basic metadata.
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    file_name = req.file_name
    # Validate file
    if not file_name.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with open('./pdfSummaryProject/uploads/doc/'+file_name, 'rb') as f:
            pdf_bytes = f.read()
        
        print('reading...')
        full_text = ocr_service.extract_text_from_pdf(pdf_bytes, dpi=400)

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )

        response = client.chat.completions.create(
            model="typhoon-v2.1-12b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summary content. You must answer only in Thai."},
                {"role": "user", "content": "สรุปข้อความต่อไปนี้: "+full_text}
            ],
            max_tokens=512,
            temperature=0.6
        )

        # Print the response

        print(response.choices[0].message.content)


        
        return {
            'result': response.choices[0].message.content
        }
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")



class PromptTestingRequest(BaseModel):
    prompt: str = Field(..., description="please prompt something")

@app.post("/test-openai")
async def test_openai(
    req: PromptTestingRequest
):
    """
    Extract text from PDF file
    
    Returns the complete text content as a single string with basic metadata.
    """
    prompt = req.prompt
    try:

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )

        response = client.chat.completions.create(
            model="typhoon-v2.1-12b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summary content. You must answer only in Thai."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.6
        )

        # Print the response

        
        return {
            'result': response.choices[0].message.content
        }
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")