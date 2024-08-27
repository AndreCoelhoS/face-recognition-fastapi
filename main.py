from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import sqlite3
import numpy as np
import io
from PIL import Image, ImageOps

app = FastAPI()

# Configurar o middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conectar ao banco de dados SQLite
conn = sqlite3.connect('faces.db', check_same_thread=False)
cursor = conn.cursor()

# Criar tabela se não existir
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL
)
''')
conn.commit()

class FaceRegistration(BaseModel):
    name: str

@app.post("/register_face/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        encoding_blob = face_encodings[0].tobytes()
        
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
        conn.commit()
        
        face_id = cursor.lastrowid
        
        return {"message": f"Face registered for {name}", "id": face_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/recognize_face/{id}")
async def recognize_face(id: int, file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        cursor.execute("SELECT encoding FROM faces WHERE id = ?", (id,))
        result = cursor.fetchone()
        
        if result is None:
            raise HTTPException(status_code=404, detail="Face ID not found")
        
        stored_encoding = np.frombuffer(result[0], dtype=np.float64)
        
        match = face_recognition.compare_faces([stored_encoding], face_encodings[0])[0]
        
        if match:
            cursor.execute("SELECT name FROM faces WHERE id = ?", (id,))
            name = cursor.fetchone()[0]
            return {"name": name, "match": True}
        else:
            return {"name": None, "match": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
