from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import face_recognition
import numpy as np
import io

app = FastAPI()

known_face_encodings = []
known_face_names = []

class FaceRegistration(BaseModel):
    name: str

@app.post("/register_face/")
async def register_face(name: str, file: UploadFile = File(...)):
    try:
        image = face_recognition.load_image_file(io.BytesIO(await file.read()))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(name)
        
        return {"message": f"Face registered for {name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...)):
    try:
        image = face_recognition.load_image_file(io.BytesIO(await file.read()))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        return {"name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
