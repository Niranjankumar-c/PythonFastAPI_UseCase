# Import necessary modules
from fastapi import FastAPI
from endpoints import upload_files

#Create FASTAPI instance
app = FastAPI()

#call the include router function
app.include_router(upload_files.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)