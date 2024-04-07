from uuid import UUID
from ito.summary import SummaryIngestion
from fastapi import APIRouter, Depends, File, Query, UploadFile

async def process_ingestion(file: UploadFile = File(None)):
    summary = SummaryIngestion(uploadFile=file)
    return await summary.process_ingestion()