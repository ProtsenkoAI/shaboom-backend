# ShaBoom backend

## About

This repository contains the backend & research code of the ShaBoom project. If you want to read about the project, refer to [this link](https://github.com/ProtsenkoAI/shaboom-backend).

1. process_song.py is a starting file to get pitches for the song file
2. run_server.py is a FastAPI server file
3. notebooks/ directory contains research code

## Install

install pipenv, then:

```
git clone git@github.com:ProtsenkoAI/shaboom-backend.git
cd shaboom-backend/
pipenv install
pipenv shell
```

### Run server:
```
uvicorn run_server:app --reload --port 8000
```

### Extract pitches from audio file
```
pipenv shell
python process_song.py "./Ben E. King - Stand by Me.mp3" ./ # or any other pathes
```