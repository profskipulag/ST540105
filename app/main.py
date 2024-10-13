from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from .websiteapi import WebsiteApi

wapi = WebsiteApi()

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name='static')

@app.post("/")
async def get_body(request: Request):

    json = await request.json()

    print(json)

    response = wapi.on_json_request(json)

    print(response)
  
    return response
