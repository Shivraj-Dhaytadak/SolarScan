import requests
import json
url = "http://127.0.0.1:8000/query"
params = {
  "dir": r"C:\\VScodeMaster\\FastAPI-MS\\Auth-Service",
  "input" : "can you explain the app.py"
}

response = requests.post(url,data=json.dumps(params) , stream=True)

for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode("utf-8"))