from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from code_analysis import vectorrizer,retrival_chain

app = FastAPI()

class Project(BaseModel):
    dir : str
    class Config:
        from_attributes=True

class Query(Project):
    input : str

@app.get("/")
async def health_check():
    return "Server Running ..."


@app.post("/scan")
async def project_scan(project:Project):
    vectorrizer(project.dir)
    return {"dir": project.dir}

@app.post("/query")
async def get_query_response(query : Query):
    streamer =  retrival_chain(query.dir , query.input)
    return StreamingResponse(streamer , media_type="text/plain")