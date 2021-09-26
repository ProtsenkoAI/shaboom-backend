from fastapi import FastAPI
from starlette.responses import StreamingResponse
import io
import json
import math

from typing import List

app = FastAPI()


@app.get("/song_file/{song_id}")
async def get_song_file(song_id: int):
    # TODO: investigate why sending file bytes is so slow
    #   (around 30-40 seconds for request)
    path = f"server_data/{song_id}.wav"
    with open(path, "rb") as f:
        return StreamingResponse(io.BytesIO(f.read()), media_type="audio/wav")


@app.get("/song_pitches/{song_id}")
async def get_song_pitches(song_id: int):
    path = f"server_data/pitches_{song_id}.json"
    with open(path) as f:
        pitches: List[float] = json.load(f)
    pitches = [pitch if not math.isnan(pitch) else -1 for pitch in pitches]
    assert isinstance(pitches, list)
    return pitches
