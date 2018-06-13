import numpy as np
import cv2
import os

'''FUNCAO QUE REALIZA A CRIACAO DE IMAGENS DE TESTE, ENQUANTO ELE RECONHECER UM ROSTO ELE VAI SALVANDO
A IMAGEM NA PASTA ESPECIFICADA'''

caminhoDetector = 'C:\Python27\Lib\site-packages\cv2\data/haarcascade_frontalface_default.xml'

#FUNCAO DE DETECCAO DE FACE
def detectarFace (frame):
	face_cascade = cv2.CascadeClassifier(caminhoDetector)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return faces
	
def reconhecerRosto(cpf):
	camera = cv2.VideoCapture(0)
	numero = 0;
	pastaTreinamento = os.getcwd() + '/treinamento'
	os.mkdir(pastaTreinamento + '/' + cpf)
	while(camera.isOpened()):
		ret,frame = camera.read()
		frameCopia = frame.copy()
		if (ret):
			faces = detectarFace(frame)
			for (x,y,w,h) in faces:
				cv2.rectangle(frameCopia, (x,y), (x+w, y+h), (0,255,0),2)	#MOSTRA A FACE RECONHECIDA
				if len(faces) == 1:
					caminho = pastaTreinamento + '/' + cpf
					string = str(numero)	#NUMERO DO ARQUIVO
					salvar = caminho + '/' + string + '.png'
					gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					cv2.imwrite(salvar, gray[y:y+w, x:x+h])
					numero = numero + 1
			cv2.imshow('camera', frameCopia)
			if cv2.waitKey(1) & 0xff == ord('q'):
				break
	camera.release()
	cv2.destroyAllWindows()
	
escolha = raw_input('Defina a sua escolha\n')
if escolha == 'foto':
	cpf = raw_input('Digite o cpf\n')
	reconhecerRosto(cpf)