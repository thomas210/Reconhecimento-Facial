import numpy as np
import cv2
import os

caminhoDetector = 'C:\Python27\Lib\site-packages\cv2\data/haarcascade_frontalface_default.xml'	

#PESSOAS COM AS IMAGENS DE TREINO
pessoas = []

#ATA DE FREQUENCIA
frequencia = []


#FUNCAO DE DETECCAO DE FACE
def detectarFace (frame):
	face_cascade = cv2.CascadeClassifier(caminhoDetector)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	return faces


def reconhecerRosto(cpf):
	camera = cv2.VideoCapture(0)
	numero = 0;
	pastaTreinamento = os.getcwd() + '/cadastros'
	if os.path.exists(pastaTreinamento + '/' + cpf) == False:
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

	
#FUNCAO QUE FAZ A CRIACAO DAS AMOSTRAS, PEGA AS IMAGENS NO DIRETORIO E DETECTA A FACE E COLOCA NO VETOR
#DE TREINO QUE SERA UTILIZADO NO RECONHECIMENTO		
def gerarAmostra ():
	print 'Analisando amostras, aguarde...'
	rostos = []
	labels = []
	caminho = os.getcwd() + '/cadastros'
	ordem = os.listdir(caminho)	#VAI NO DIRETORIO
	numero = 0
	for pastaPessoa in ordem:
		caminhoPessoa = caminho + '/' + pastaPessoa
		listaImagens = os.listdir(caminhoPessoa)
		for imagemPessoa in listaImagens:	
			caminhoImagem = caminhoPessoa + '/' + imagemPessoa
			rosto = cv2.imread(caminhoImagem)
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
	
def preencherPessoas():
	pastaPessoa = os.listdir(os.getcwd() + '/cadastros')
	for nome in pastaPessoa:
		pessoas.append(nome)




#MAIN
if os.path.exists(os.getcwd() + '/cadastros') == False:
	os.mkdir(os.getcwd() + '/cadastros')
print "RECONHECIMENTO FACIAL"
while True :
	escolha = raw_input('Escolha a opcao\n')
	if escolha == 'cadastro':
		nome = raw_input('Digite o nome do aluno:\n')
		print 'Iniciando gravacao'
		reconhecerRosto(nome)
	elif escolha == 'chamada' :
		print 'cadastrados'
		preencherPessoas()
		print pessoas
		print 'Iniciando chamada'
		rostos, labels = gerarAmostra()
		reconhecimento = cv2.face.LBPHFaceRecognizer_create()
		reconhecimento.train(rostos, np.array(labels))
		cameraInit(reconhecimento)
		print frequencia
	elif escolha == 'quit' :
		print 'Saindo'
		break
	elif escolha == 'help' :
		print 'opcoes:\ncadastro - realiza um cadastro\nchamada - inicia a chamada\n'
		
	else :
		print 'Opcao invalida'