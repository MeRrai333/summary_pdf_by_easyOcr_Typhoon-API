
# summary_pdf_by_easyOcr_Typhoon-API

Project for summary pdf file by EasyOCR and Typhoon-API

## Prepare
- API Key from https://docs.opentyphoon.ai
- Dev in Python 3.12.3

## Installation
#### 1. Create venv then active
`
python -m venv venv
`
#### On Window
`venv\Scripts\activate`
#### On macOS/Linux:
`source venv/bin/activate`

#### 2. Install Dependencies
`pip install -r requirements.txt`

#### 3. Set Up Environment Variables
Create .env fil in root of project then add
`OPENAI_API_KEY=<TOKEN>`

#### 4. Run FastAPI
`uvicorn pdfSummaryProject.main:app --host 0.0.0.0 --port 8000 --reload`

## How to use
#### 1. Add pdf file by
- POST /upload-file-pdf
- FormData
`files: "PDF Files"`

#### 2. Testing(Optional), Is it readable by
\
- POST /extract-text
- Body
```
{
    "file_name": "file name in uploads/doc"
}
```
- Example name of pdf file in uploads/doc is `ABC.pdf`
- Body
```
{
    "file_name": "ABC.pdf"
}
```
#### 3. Testing(Optional), Is Typhoon-API working.
\
- POST /test-openai
- Body 
```
{
    "prompt": "สวัสดี"
}
```
#### 4. Summary pdf file by
- POST /summary-pdf-file_name
- Body
```
{
    "file_name": "ABC.pdf"
}
```

Or you can use `PdfSummary.postman_collection.json` with Postman