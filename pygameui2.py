import pygame
from tkinter import filedialog
from tkinter import *
pygame.init()
root = Tk()
root.wm_withdraw()
screen = pygame.display.set_mode((1300,650))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (30, 30, 30)
FONT = pygame.font.Font("freesansbold.ttf",20)
done=True
button = pygame.Rect(50, 500, 90, 90)
flag=1
testimage='./test_image/c00.png'     
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
import glob
from skimage import io
from PIL import Image
   
BLUE=(0,0,255)
ix1,iy1=200,250
ix2,iy2=250,250
ix3,iy3=300,250
ix4,iy4=200,300
ix5,iy5=250,300
ix6,iy6=300,300
ix7,iy7=200,350
ix8,iy8=250,350
ix9,iy9=300,350
ll1=450
ll2=600
ll3=750
colorchange=False
move=True
split=True
Ans="-"

global changecolor
changecolor=False
RED=(255,0,0)
signs = []
with open('signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[-1])
    csvfile.close()
def connect1(ll,colr):
    
    l1=450
    l2=50
    l3=ll
    l4=72
    for i in range(9):
        for j in range(8):
            pygame.draw.line(screen,colr, (l1,l2), (l3,l4))
            l4=l4+70
        l4=72
        l2=l2+70
def connect2(ll,colr):
    l1=600
    l2=72
    l3=ll
    l4=72
    for i in range(8):
        for j in range(8):  
            pygame.draw.line(screen,colr, (l1,l2), (l3,l4))
            l4=l4+70
        l4=72
        l2=l2+70   
def connect3(ll,colr):   
    l1=750
    l2=72
    for i in range(8):
        pygame.draw.line(screen,colr,(l1,l2),(ll,317))
        l2=l2+70


def neurons1(nclr):    
    cx=450
    cy=50
    for i in range(9):
        pygame.draw.circle(screen,nclr,(cx,cy),25, 0)
        cy=cy+70
def neurons2(nclr):
    cx=600
    cy=72
    for i in range(8):
        pygame.draw.circle(screen,nclr, (cx,cy), 25, 0)
        cy=cy+70
def neurons3(nclr):
    cx=750
    cy=75
    for i in range(8):
        pygame.draw.circle(screen,nclr, (cx,cy), 25, 0)
        cy=cy+70
def neurons4(nclr):        
    pygame.draw.circle(screen,nclr,(900,317), 25, 0)
def gray_scale(image):
    
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local
def image_normalize(image):
 
    
    image = np.divide(image, 255)
    return image  
    
def preprocess(data):   
    gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    normalized_images = normalized_images[..., None]
    return normalized_images

def y_predict_model(Input_data, top_k=5):
    
    graph = tf.get_default_graph()
    all_ops = graph.get_operations()
    for el in all_ops:
        print(el)
    num_examples = len(Input_data)
    y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
    y_prob = np.zeros((num_examples, top_k))
    with tf.Session() as sess:
      
        saver=tf.train.import_meta_graph("VGGNet.meta")
        saver.restore(sess, os.path.join('saved_models', "VGGNet"))
        #saver.restore(sess,tf.train.latest_checkpoint('/home/disha2000/Videos/final/Saved_Models/LeNet_normalized'))
        logits = graph.get_tensor_by_name("logits:0")
        
        x = graph.get_tensor_by_name('input_data:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        keep_prob_conv = graph.get_tensor_by_name('keep_prob_conv:0')
        y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=top_k), 
                             feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})
    return y_prob, y_pred


    
def showresult(z):
    text_surf = FONT.render(z,True,WHITE)
    screen.blit(text_surf,(1000,317))

def bgrnd():
    screen.fill(GRAY)
    pygame.draw.rect(screen,BLACK, button)
    text_surf = FONT.render("Browse", True,WHITE)
    screen.blit(text_surf,(58,535))
    
    #pygame.draw.rect(screen,BLACK, button)
    #text_surf = FONT.render("Top 5 Probabilities", True,WHITE)
    #screen.blit(text_surf,(1100,535))
    
def bu(testimage,ix1,ix2,ix3,ix4,ix5,ix6,ix7,ix8,ix9,iy1,iy2,iy3,iy4,iy5,iy6,iy7,iy8,iy9):     
    
    ti=pygame.image.load(testimage)
    ti=pygame.transform.scale(ti,(150,150))
    screen.blit(ti,(20,250))
 
    i1=pygame.image.load("./test_image/c00.png")
    i2=pygame.image.load("./test_image/c01.png")
    i3=pygame.image.load("./test_image/c02.png")
    i4=pygame.image.load("./test_image/c10.png")
    i5=pygame.image.load("./test_image/c11.png")
    i6=pygame.image.load("./test_image/c12.png")
    i7=pygame.image.load("./test_image/c20.png")
    i8=pygame.image.load("./test_image/c21.png")
    i9=pygame.image.load("./test_image/c22.png")
    i1=pygame.transform.scale(i1,(50,50))
    i2=pygame.transform.scale(i2,(50,50))
    i3=pygame.transform.scale(i3,(50,50))
    i4=pygame.transform.scale(i4,(50,50))
    i5=pygame.transform.scale(i5,(50,50))
    i6=pygame.transform.scale(i6,(50,50))
    i7=pygame.transform.scale(i7,(50,50))
    i8=pygame.transform.scale(i8,(50,50))
    i9=pygame.transform.scale(i9,(50,50))        
    screen.blit(i1,(ix1,iy1))        
    screen.blit(i2,(ix2,iy2))
    screen.blit(i3,(ix3,iy3))
    screen.blit(i4,(ix4,iy4))
    screen.blit(i5,(ix5,iy5))
    screen.blit(i6,(ix6,iy6))
    screen.blit(i7,(ix7,iy7))
    screen.blit(i8,(ix8,iy8))
    screen.blit(i9,(ix9,iy9))
testimage="NONE"
def Upload():
    global testimage
    testimage=filedialog.askopenfilename(initialdir ="./test_images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    
    global flag
    flag=0
    new_test_images = []
    path = './'
    for el in glob.glob(testimage) :
        img = io.imread(el)  
        #img = cv2.imread(img)
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_test_images.append(img)

        new_IDs = [0]
    print("Number of new testing examples: ", len(new_test_images))

    new_test_images_preprocessed = preprocess(np.asarray(new_test_images))
    graph = tf.get_default_graph()
    all_ops = graph.get_operations()
    for el in all_ops:
        print(el)
    y_prob, y_pred = y_predict_model(new_test_images_preprocessed)
    test_accuracy = 0
    for i in enumerate(new_test_images_preprocessed):
        accu = new_IDs[i[0]] == np.asarray(y_pred[i[0]])[0]
        if accu == True:
            test_accuracy += 0.2
    
    with plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'white'}):
        plt.figure(1,figsize=(15,15))
        #for i in range(len(new_test_images_preprocessed)):
        plt.plot(len(new_test_images_preprocessed), 2, 2*0+1)
        plt.imshow(new_test_images[0]) 
        #title=plt.title(signs[y_pred[0][0]])
        plt.axis('off')
        plt.savefig('./test_image/pre.png', bbox_inches='tight',transparent=True)
        img=Image.open("./test_image/pre.png")
                
        
        l=0
        t=0
        r=400
        b=400
        for i in range(3):
            l=0
            r=400
            for j in range(3):
                area=(l,t,r,b)
                cropped_img=img.crop(area)
                cropped_img.save("./test_image/c"+str(i)+str(j)+".png")
                l=l+400
                r=r+400
            b=b+400
            t=t+400        

        plt.figure(2,figsize=(10,5))
        plt.plot(len(new_test_images_preprocessed), 2, 2*0+2)
        plt.barh(np.arange(1, 6, 1), y_prob[0, :], alpha=0.4, height=0.5)
        labels = [signs[j] for j in y_pred[0]]
        plt.yticks(np.arange(1, 6, 1), labels,fontsize='large')
        plt.subplots_adjust(left=0.4, bottom=None, right=None, top=None, wspace=1, hspace=None)
        plt.savefig('probabilities.png', bbox_inches='tight',transparent=True)
        global Ans,ix1,ix2,ix3,ix4,ix5,ix6,ix7,ix8,ix9,iy1,iy2,iy3,iy4,iy5,iy6,iy7,iy8,iy9

        Ans=labels[0]         
                
start=False
count=0
show=0
while done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = False
        # This block is executed once for each MOUSEBUTTONDOWN event.
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 1 is the left mouse button, 2 is middle, 3 is right.
            if event.button ==1:
                if button.collidepoint(event.pos):
                    show=0
                    count=0
                    ix1,iy1=200,250
                    ix2,iy2=250,250
                    ix3,iy3=300,250
                    ix4,iy4=200,300
                    ix5,iy5=250,300
                    ix6,iy6=300,300
                    ix7,iy7=200,350
                    ix8,iy8=250,350
                    ix9,iy9=300,350
                    start=True              
                    Upload()
                                    
    
    if(start==True and count<30):
        count=count+1
    if(flag==0):
        bgrnd()
        bu(testimage,ix1,ix2,ix3,ix4,ix5,ix6,ix7,ix8,ix9,iy1,iy2,iy3,iy4,iy5,iy6,iy7,iy8,iy9)
        
        if(ix1<401 and move==True):
            if(count>=30):
                start=False
                
                ix1,iy1=ix1+24,iy1-25
                ix2,iy2=ix2+19,iy2-17
                ix3,iy3=ix3+13,iy3-10
                
                ix4,iy4=ix4+24,iy4-7
                ix5,iy5=ix5+19,iy5-0
                ix6,iy6=ix6+13,iy6+9
                
                ix7,iy7=ix7+24,iy7+10
                ix8,iy8=ix8+19,iy8+19
                ix9,iy9=ix9+14,iy9+26
            #connect1(600,RED)
            #connect2(750,RED)
            #connect3(900,RED)
            neurons1(BLUE)
            neurons2(BLUE)
            neurons3(BLUE)
            neurons4(BLUE)                      

        else:
        
            connect1(600,RED)
            connect2(750,RED)
            connect3(900,RED)
            neurons1(BLUE)
            neurons2(BLUE)
            neurons3(BLUE)
            neurons4(BLUE)                      
                                  
            changecolor=True
            
            
            if(ll1+15<=600):
                ll1=ll1+15
                connect1(ll1,WHITE)    
            elif(ll2+15<=750):
                ll2=ll2+15
                connect2(ll2,WHITE)
                
            elif(ll3+15<=900):
                ll3=ll3+15
                connect3(ll3,WHITE)
            else:
                ll1=450
                ll2=600
                ll3=750
                show=1
                
            neurons1(BLUE)
            neurons2(BLUE)
            neurons3(BLUE)
            neurons4(BLUE)                      

            if(show==1):
                showresult(Ans)
                        
                                
        
        
                
        #background
    else:
        bgrnd()
        #global ix1,ix2,ix3,ix4,ix5,ix6,ix7,ix8,ix9,iy1,iy2,iy3,iy4,iy5,iy6,iy7,iy8,iy9
    
    pygame.display.update()        
 
pygame.quit()

