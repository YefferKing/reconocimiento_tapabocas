#entrenar con las iamgenes de prueba  
import cv2 #detecta el rostro de las personas
import os #Listar las carptas de las imagenes de prueba 
import numpy as np #manipulacion mas eficiente de las imagenes 
dataPath = "G:\Documentos\INTELIGENCIA ARTIFICAL\RECONOCIMIENTO DE MASCARILLA\Dataset_faces" #direccion de la carpeta de prueba 
dir_list = os.listdir(dataPath) #asiga el listado de imagenes
print("Lista archivos:", dir_list) #poder visualizar en consola que los archivos se listaron 
labels = [] # etiqueta que idetifica cada imagen
facesData = [] # se almacenan todos los rostros de prueba
label = 0 # contar los identificadores de cada imagen, osea 0 y 1
for name_dir in dir_list:  # recorrer el listado de imagenes d eprueba 
     dir_path = dataPath + "/" + name_dir # concatenar para obtener el contenido del Data_set
     
     for file_name in os.listdir(dir_path):
          image_path = dir_path + "/" + file_name # concatenar para obtener el contenido del Data_set
          print(image_path) # verificar que las imagenes se concatenaron 
          image = cv2.imread(image_path, 0) # lee las imagenes con identificacion 0 las tiene mascarilla
          #cv2.imshow("Image", image)
          #cv2.waitKey(10)
          facesData.append(image)  # me almacena las imagenes en facesData
          labels.append(label)     # me almacena las etiquetas de las imagenes 
     label += 1 # incrementa las etiquetas 
print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0)) #identifico a los rostros que tienen la mascarilla
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1)) #identifico a los rostros que NO tienen la mascarilla

# LBPH FaceRecognizer
face_mask = cv2.face.LBPHFaceRecognizer_create()
# Entrenamiento
print("Un momento se esta entrenando el modelo...")
face_mask.train(facesData, np.array(labels))
# Almacenar modelo
face_mask.write("face_mask_model.xml")
print("Almacenado Exitosamente!!")