import cv2
import numpy
from PIL import Image
from binarizacao import Binarizacao


# parametro 0 para webcam integrada / usb
# url para camera ip
camera        = cv2.VideoCapture('http://192.168.0.4:8080/video')
haarcascade   = 'haarcascade_russian_plate_number.xml'
classificador = cv2.CascadeClassifier(haarcascade)

while (True):
    # captura imagem da camera ip
    conectado, img_camera = camera.read()

    # transforma imagem em preta e branca
    img_cinza = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)

    # aplica classificador na imagem
    placas_detectadas = classificador.detectMultiScale(
        img_cinza,
        scaleFactor=1.5,
        minSize=(200, 150) )
    
    # loop pelas placas detectadas na imagem
    corte_x, corte_y, corte_a, corte_l = 0, 0, 0, 0
    for (x, y, l, a) in placas_detectadas:
        # define variáveis do corte com os valores do loop atual
        corte_x, corte_y = x, y
        corte_l, corte_a = l, a

        # desenha retangulo na imagem detectada
        cv2.rectangle(
            img_cinza,
            (x, y),     # ponto 1
            (x+l, y+a), # ponto 2
            (0, 0, 0),  # cor RGB
            2 )

    img_grande = cv2.resize(img_cinza, (800, 600))
    # se detectou a placa, corta a imagem
    if len(placas_detectadas) > 0:
        img_cortada = img_cinza[
            corte_y+5:corte_y+corte_a-5,
            corte_x+5:corte_x+corte_l-5]

        # diminui tamanho da imagem para 200 x 100
        img_pequena = cv2.resize(img_cortada, (200, 100))

        # converte imagem de CV2 para PIL
        img_plp = Image.fromarray(cv2.cvtColor(img_pequena,cv2.COLOR_BGR2RGB))  

        # binariza a imagem
        img_binarizada = Binarizacao(img_plp, 80).processar()

        # converte imagem de PIL para CV2
        img_mat = cv2.cvtColor(numpy.asarray(img_binarizada), cv2.COLOR_RGB2BGR)

        # aumenta tamanho da imagem para 800 x 600
        # img_grande = cv2.resize(img_mat, (400, 300))

        # salva imagem detectada
        cv2.imwrite('capturas\\binarizacao.png', img_mat)

    # mostra imagem em tela
    cv2.imshow("binarizacao", img_grande)
    cv2.waitKey(1)