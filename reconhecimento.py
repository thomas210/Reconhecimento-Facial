import numpy as np
import cv2
import os

caminhoDetector = 'C:\Python27\Lib\site-packages\cv2\data/haarcascade_frontalface_default.xml'	

#PESSOAS COM AS IMAGENS DE TREINO
pessoas = ['','Thomas', 'Pedro Victor','Carvalho']

#ATA DE FREQUENCIA
frequencia = []

#FUNCAO QUE REALIZA A DETECÇÃO DA FACE E RETORNA AS COORDENADAS DA FACE
def detectarFace (frame, tipo):
	face_cascade = cv2.CascadeClassifier(caminhoDetector)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	resultado = face_cascade.detectMultiScale(gray, 1.3, 5)
	if (tipo == 1):
		(x, y, w, h) = resultado[0]
		return gray[y:y+w, x:x+h]
	if (tipo == 0):
		return resultado

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
			imagem = cv2.imread(caminhoImagem)
			rosto = detectarFace(imagem, 1)
			cv2.imshow('Analisando', rosto)
			cv2.waitKey(100)
			rostos.append(rosto)
			labels.append(numero)
	cv2.destroyAllWindows()
	print 'imagens analizadas'
	return rostos, labels

#REALIZA O RECONHECIMENTO, SE A PROCENTAGEM DE ERRO FOR MAIOR DE 50% ENTAO ELE RETORNA
#NAO FOI POSSIVEL RECONHECER
def reconhecer (frame, reconhecimento):
	face = detectarFace(frame, 1)
	label, erro = reconhecimento.predict(face)
	if (erro > 50):
		return 'nao existe no banco de dados'
	else:
		return pessoas[label]

#INICIO DA CAMERA
def cameraInit (reconhecimento):
	camera = cv2.VideoCapture(0)
	while (camera.isOpened()):
		ret, frame = camera.read()
		res = cv2.waitKey(1)
		if (ret):
			face = detectarFace(frame, 0)
			for (x,y,w,h) in face:	#A CAMERA MOSTRA UM QUADRADO QUANDO DETECTA UM ROSTO
				cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
			cv2.imshow('camera', frame)
			if res & 0xff == ord('s'):	#AO APERTAR A TECLA 's' O PROGRAMA REALIZA O RECONHECIMENTO
				if len(face) != 1:	#FAZER O RECONHECIMENTO APENAS SE TIVER UM ROSTO NA CAMERE
					continue
				resultado = reconhecer(frame, reconhecimento)
				print (resultado)	#INFORMA QUEM E A PESSOA
				chamada(resultado)	#FAZ A CHAMADA
		if res & 0xff == ord('q'):	#FINALIZA A CAMERA
			break
	camera.release()
	cv2.destroyAllWindows()
		
#COLOCA O NOME DA PESSOA NA CHAMADA
def chamada(nome):
	if nome == 'nao existe no banco de dados':
		return
	frequencia.append(nome)
	

#MAIN
caminho = 'F:\Faculdade\PLF\Projeto/treinamento'
rostos, labels = gerarAmostra(caminho)
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.train(rostos, np.array(labels))
cameraInit(reconhecimento)
print frequencia

