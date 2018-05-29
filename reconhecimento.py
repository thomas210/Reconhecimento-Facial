import numpy as np
import cv2
import os

caminhoDetector = 'C:\Python27\Lib\site-packages\cv2\data/haarcascade_frontalface_default.xml'	

#PESSOAS COM AS IMAGENS DE TREINO
pessoas = ['','Vinicius', 'Thomas']

#ATA DE FREQUENCIA
frequencia = []


def detectarFace (frame):
	face_cascade = cv2.CascadeClassifier(caminhoDetector)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return faces
	
'''	
def detectarFace (frame):
	face_cascade = cv2.CascadeClassifier(caminhoDetector)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	resultado = face_cascade.detectMultiScale(gray, 1.3, 5)
	return resultado
	if (tipo == 1):
		(x, y, w, h) = resultado[0]
		return gray[y:y+w, x:x+h]
	if (tipo == 0):
		return resultado'''

#FUNCAO QUE FAZ A CRIACAO DAS AMOSTRAS, PEGA AS IMAGENS NO DIRETORIO E DETECTA A FACE E COLOCA NO VETOR
#DE TREINO QUE SERA UTILIZADO NO RECONHECIMENTO		
def gerarAmostra (caminho):
	rostos = []
	labels = []
	ordem = os.listdir(caminho)	#VAI NO DIRETORIO
	for pastaPessoa in ordem:
		numero = int(pastaPessoa.replace('p', ''))
		caminhoPessoa = caminho + '/' + pastaPessoa
		listaImagens = os.listdir(caminhoPessoa)
		for imagemPessoa in listaImagens:	
			caminhoImagem = caminhoPessoa + '/' + imagemPessoa
			rosto = cv2.imread(caminhoImagem)
			cv2.imshow('Analisando', rosto)
			cv2.waitKey(100)
			rostoGray = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
			rostos.append(rostoGray)
			labels.append(numero)
	cv2.destroyAllWindows()
	print 'imagens analizadas'
	return rostos, labels

#REALIZA O RECONHECIMENTO, SE A PROCENTAGEM DE ERRO FOR MAIOR DE 50% ENTAO ELE RETORNA
#NAO FOI POSSIVEL RECONHECER
def reconhecer (frame, reconhecimento):
	faces = detectarFace(frame)
	detectado = []
	for (x,y,w,h) in faces:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		label, erro = reconhecimento.predict(gray[y:y+w, x:x+h])
		if (erro > 50):
			print 'nao existe no banco de dados'
		else:
			detectado.append(pessoas[label])
			print (pessoas[label])	#INFORMA QUEM E A PESSOA
	return detectado

#INICIO DA CAMERA
def cameraInit (reconhecimento):
	camera = cv2.VideoCapture(0)
	while (camera.isOpened()):
		ret, frame = camera.read()
		res = cv2.waitKey(1)
		if (ret):
			face = detectarFace(frame)
			for (x,y,w,h) in face:	#A CAMERA MOSTRA UM QUADRADO QUANDO DETECTA UM ROSTO
				cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
			cv2.imshow('camera', frame)
			resultado = reconhecer(frame, reconhecimento)	
			chamada(resultado)	#FAZ A CHAMADA
		if res & 0xff == ord('q'):	#FINALIZA A CAMERA
			break
	camera.release()
	cv2.destroyAllWindows()
		
#COLOCA O NOME DA PESSOA NA CHAMADA
def chamada(lista):
	for nome in lista:
		for comp in frequencia:
			if nome == comp:
				break
		else:
			frequencia.append(nome)
	

#MAIN
caminho = 'F:\Faculdade\PLF\Reconhecimento-Facial/treinamento'
rostos, labels = gerarAmostra(caminho)
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.train(rostos, np.array(labels))
cameraInit(reconhecimento)
print frequencia

