
# From Zero to Vertex AI: Google Cloud Vision on Cloud Run functions

![Google Cloud Vision on Cloud Run functions](snip.png)

Google Cloud Vision is a machine learning service offered by Google Cloud Platform that provides powerful image analysis capabilities through pre-trained models and APIs. It can extracts text from images, including handwritten text, signs, documents, and text in various languages. It can detect both printed and handwritten text

While Document AI is complex, Cloud Vision API OCR is quick OCR for images & simple PDFs

## Compare ::
https://cloud.google.com/vision?hl=en


You deploy function code directly (source-based), and Google handles containerization unlike Cloud Run.
Cloud Run Functions offers higher abstraction—focus on business logic, not infrastructure.


## Velox

Vellox is an adapter for running ASGI applications ((Asynchronous Server Gateway Interface) ) in GCP Cloud Functions.

## Steps 

### Dev Setup
Use devcontainer to install ( I am using so this is easy )

create main.py with fastapi and vellox

### Enable Services
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com logging.googleapis.com aiplatform.googleapis.com vision.googleapis.com cloudfunctions.googleapis.com storage.googleapis.com

### IAM 
Grant the Cloud Run/Functions service account read on the input bucket and write on the output bucket (e.g., roles/storage.objectViewer on upload bucket, roles/storage.objectCreator on output bucket).
(Not needed if you create bucket with same project)

### enable ADC locally (one-time)
gcloud auth application-default login
gcloud config set project your-project-id

### create buckets (choose your region, e.g., asia-south1)
gcloud storage buckets create gs://gcv_upload --location=asia-south1 --uniform-bucket-level-access
gcloud storage buckets create gs://gcv_output --location=asia-south1 --uniform-bucket-level-access
gcloud storage buckets create gs://gcv_vision_ocr --location=asia-south1 --uniform-bucket-level-access

### run locally
uvicorn main:app --reload --port 8080

### Deploy with ENV, Variables
gcloud functions deploy gcv-fastapi --gen2 --region=asia-south1 --runtime=python313 --entry-point=handler --trigger-http --allow-unauthenticated --memory=2GiB --timeout=600s --set-env-vars=UPLOAD_BUCKET=gcv_upload,OUTPUT_BUCKET=gcv_output,OUTPUT_ROOT=gcv_vision_ocr,API_BEARER_TOKENS=your-bearer-token

### Delete when Done

gcloud functions delete gcv-fastapi --region=asia-south1 --gen2

## Check remotely

BASE=some-url
### 1) Start the async job
curl -s -X POST "$BASE/ocr/async/start" -F "file=@/path/to/big.pdf"  

### 2) Poll status
curl -s "$BASE/ocr/async/status?operation=projects/.../operations/abcd..."  
                                                                               ↓
### 3) Fetch results once done
curl -s "$BASE/ocr/async/result?prefix=gs://YOUR_OUTPUT_BUCKET/vision-ocr/jobs/<id>/"  


