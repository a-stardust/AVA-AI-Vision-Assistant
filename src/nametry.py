
from face_recognizer import recognize_faces
recognized_faces = recognize_faces("frame.jpg")
print(recognized_faces)


faces=recognized_faces
names=''
if len(faces) > 1:
    for i in range(0,len(faces)-1):
        names+=faces[i]
        names+=' and '
    names+=faces[len(faces)-1]
    print (names)

elif len(faces)==1:
    print(faces[0])

elif len(faces) == 0:
    print('no person detected')