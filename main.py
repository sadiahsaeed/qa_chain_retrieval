from fastapi import FastAPI, File, UploadFile
from utils import load_split_pdf_file, load_split_docx_file, load_split_text_file, QA_Chain_Retrieval, QdrantInsertRetrievalAll
import tempfile
import os
from langchain_openai import OpenAIEmbeddings
#import uvicorn
from zipfile import ZipFile
import shutil
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
collection_name = "testing"

my_qdrant = QdrantInsertRetrievalAll(api_key = qdrant_api_key, url = qdrant_url)

app = FastAPI()

def process_file_by_extension(file_path: str, file_extension: str):
    """
    Process a file based on its extension and return the chunks.
    
    Args:
        file_path (str): Path to the file.
        file_extension (str): Extension of the file (e.g., 'docx', 'pdf', 'txt').
    
    Returns:
        list: Chunks extracted from the file.
    """
    if file_extension == "docx":
        return load_split_docx_file(file_path)
    elif file_extension == "pdf":
        return load_split_pdf_file(file_path)
    elif file_extension == "txt":
        return load_split_text_file(file_path)
    else:
        return [f"Unsupported file type: {file_extension}"]

@app.post("/upload_files/")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []
    all_chunks = []  # To accumulate chunks from all files

    for file in files:
        try:
            contents = await file.read()
            file_extension = file.filename.split(".")[-1].lower()

            if file_extension == "zip":
                fd, tmp_file_path = tempfile.mkstemp(suffix=".zip")
                with os.fdopen(fd, 'wb') as tmp_file:
                    tmp_file.write(contents)

                extract_dir = tempfile.mkdtemp()
                try:
                    with ZipFile(tmp_file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    for extracted_file in os.listdir(extract_dir):
                        extracted_file_path = os.path.join(extract_dir, extracted_file)
                        extracted_file_extension = extracted_file.split(".")[-1].lower()

                        with open(extracted_file_path, 'rb') as ef:
                            extracted_contents = ef.read()

                        chunks = process_file_by_extension(extracted_file_path, extracted_file_extension)

                        results.append({"filename": extracted_file, "content_preview": chunks[:500]})
                        all_chunks.extend(chunks)
                finally:
                    shutil.rmtree(extract_dir)
                    os.unlink(tmp_file_path)

            else:
                fd, tmp_file_path = tempfile.mkstemp(suffix=f".{file_extension}")
                with os.fdopen(fd, 'wb') as tmp_file:
                    tmp_file.write(contents)

                chunks = process_file_by_extension(tmp_file_path, file_extension)

                results.append({"filename": file.filename, "content_preview": chunks[:500]})
                all_chunks.extend(chunks)

                os.unlink(tmp_file_path)

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    try:
        my_qdrant.insertion(all_chunks, embeddings, collection_name)
        insertion_status = "Insertion successful"
    except Exception as e:
        insertion_status = f"Insertion failed: {str(e)}"

    data = {
        "vecDbPath": insertion_status,
        "collection_name": collection_name
    }

    return data
    #return {"data": data, "file_results": results}

class QueryRequest(BaseModel):
    query: str

@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    try:
        # Extract the query from the request
        query_user = request.query

        # Perform retrieval from Qdrant
        vectorstore = my_qdrant.retrieval(embeddings=embeddings, collection_name=collection_name)
        results = QA_Chain_Retrieval(query=query_user, qdrant_vectordb=vectorstore)

        # If results are found, return the content
        return {"content": results.content}
    except Exception as e:
        # Return a user-friendly error message
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1616, reload=True)