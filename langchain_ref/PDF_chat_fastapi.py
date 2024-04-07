from fastapi import FastAPI, File, Form, UploadFile
from typing import Annotated
import os

from pydantic import BaseModel

from PDF_chat import *
file_upload_path = './uploads'
app = FastAPI()

@app.get('/')
def read_root():
    return 'hello world'

@app.post('/upload')
def upload_pdf(
        file: Annotated[UploadFile, File()],
        index_name: Annotated[str, Form()]
):
    file_upload_target_path = os.path.join(file_upload_path, file.filename)
    with open(file_upload_target_path, 'wb') as f:
        contents = file.file.read()
        f.write(contents)

    load_pdf_and_save_to_index(file_upload_target_path, index_name)
    return {'filename': file.filename, 'index_name': index_name}

class Query(BaseModel):
    index_name: str
    query: str

@app.post('/query')
def query_index(requst: Query):
    index_name = requst.index_name
    query = requst.query
    index = load_index(index_name)
    ans = query_index_lc(index,query)
    return {'answer': ans, 'index_name': index_name, 'query': query}