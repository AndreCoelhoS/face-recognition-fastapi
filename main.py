from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import face_recognition
import sqlite3
import numpy as np
import io

app = FastAPI()

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
        image = face_recognition.load_image_file(io.BytesIO(await file.read()))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # Converter o encoding para um formato compatível com SQLite (BLOB)
        encoding_blob = face_encodings[0].tobytes()
        
        # Inserir os dados na tabela do SQLite
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
        conn.commit()
        
        # Recuperar a ID gerada para o registro
        face_id = cursor.lastrowid
        
        return {"message": f"Face registered for {name}", "id": face_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/recognize_face/{id}")
async def recognize_face(id: int, file: UploadFile = File(...)):
    try:
        # Carregar a imagem enviada para reconhecimento
        image = face_recognition.load_image_file(io.BytesIO(await file.read()))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # Recuperar o encoding do banco de dados baseado na ID
        cursor.execute("SELECT encoding FROM faces WHERE id = ?", (id,))
        result = cursor.fetchone()
        
        if result is None:
            raise HTTPException(status_code=404, detail="Face ID not found")
        
        stored_encoding = np.frombuffer(result[0], dtype=np.float64)
        
        # Comparar a face enviada com a armazenada
        match = face_recognition.compare_faces([stored_encoding], face_encodings[0])[0]
        
        if match:
            cursor.execute("SELECT name FROM faces WHERE id = ?", (id,))
            name = cursor.fetchone()[0]
            return {"name": name, "match": True}
        else:
            return {"name": None, "match": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
