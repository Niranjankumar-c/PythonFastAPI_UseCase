# Import necessary modules
from fastapi import FastAPI
from endpoints import upload_files
from endpoints import ocr

#Create FASTAPI instance
app = FastAPI()

#call the include router function
app.include_router(upload_files.router)
app.include_router(ocr.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)