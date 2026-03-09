from fastapi import FastAPI, Request
from logic import choose_move

app = FastAPI()

@app.get("/")
def info():
    return {
        "apiversion": "1",
        "author": "yourname",
        "color": "#00FFFF",
        "head": "default",
        "tail": "default"
    }

@app.post("/start")
def start():
    return {}

@app.post("/move")
async def move(request: Request):
    data = await request.json()
    direction = choose_move(data)
    return {"move": direction}

@app.post("/end")
def end():
    return {}
