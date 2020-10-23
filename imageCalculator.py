# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:26:25 2019

@author: Madmax
"""

from tensorflow.keras.models import model_from_json
json_file = open('modelnew.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('modelnew.h5')
#%%
import cv2
import numpy as np
import math
import cmath
#CATEGORIES=["+","-","times","div","!","(",")","0","1","2","3","4","5","6","7","8","9","sqrt","=","pi","i","j","k"]
#CATEGORIES=["+","-","*","/","!","(",")","0","1","2","3","4","5","6","7","8","9","sqrt","=","pi","i","j","k"]
CATEGORIES=["+","-","*","/","(",")","0","1","2","3","4","5","6","7","8","9","=","k"]
#%%
def preprocess(X_in,w,h):
    #X_=np.array(mat)
    print(str(w)+" "+str(len(X_in[0])))
    mat=[[1.0 for i in range(w+20)] for j in range(h+20)]
    mat=np.array(mat)
    for i in range(10,h+10):
        mat[i][10:10+w]=X_in[i][:];
    
    mat=np.array(mat)
    cv2.imshow('frame',mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#%%parametr
th=120
ar=100
#%%
kernel = np.array([[0,0,1,0,0],
                  [0,1,1,1,0],
                  [1,1,1,1,1],
                  [0,1,1,1,0],
                  [0,0,1,0,0]],np.uint8)

kernel1 = np.array([[1,1,1,1,1],
                    [1,1,0,1,1],
                    [1,0,0,0,1],
                    [1,1,0,1,1],
                    [1,1,1,1,1]],np.uint8)

kernel2 = np.array([[1,1,0,1,1],
                    [1,1,0,1,1],
                    [0,0,0,0,0],
                    [1,1,0,1,1],
                    [1,1,0,1,1]],np.uint8)

chotakernel = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]],
                        np.uint8)
chotakernel1 = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]],
                        np.uint8)


img = cv2.imread('Test1001.jpeg',0)
test =img
_,img =cv2.threshold(img,th,255,cv2.THRESH_BINARY)
print(str(img.shape[0])+" "+str(img.shape[1]))
cv2.imshow('frame',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
#test = np.array(None,np.uint8)
if(img.shape[0]<500 or img.shape[1]<500):
    
    img = cv2.erode(img,chotakernel,iterations = 1)
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #img = cv2.dilate(img,kernel1,iterations = 3)
    #img = cv2.GaussianBlur(img,(2,2),0)
    img = cv2.resize(img,(int(img.shape[1]*800/img.shape[0]),800))
    img = cv2.dilate(img,chotakernel,iterations = 1)
    #img = cv2.dilate(img,kernel1,iterations = 1)
    img = cv2.erode(img,chotakernel,iterations = 2)
    test= img
    
    cv2.imshow('Testif',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    arr=100
    print('if')

    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.dilate(img,kernel1,iterations = 2)
    #img = cv2.GaussianBlur(img,(3,3),0)
else:
    print('else')
    img = cv2.dilate(img,kernel,iterations = 2)
    img = cv2.erode(img,kernel,iterations = 3)
    img = cv2.dilate(img,kernel1,iterations = 2)
    img = cv2.dilate(img,kernel2,iterations = 2)
    
    
    #img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.erode(img,kernel1,iterations = 4)
    img = cv2.erode(img,kernel2,iterations = 1)
    img = cv2.resize(img,(3*img.shape[0],3*img.shape[1]))



    #_,test =cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    cv2.imshow('Test',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test = cv2.erode(test,kernel,iterations = 3)#####
    test = cv2.dilate(test,kernel,iterations = 1)#####
    
    test = cv2.dilate(test,kernel2,iterations = 1)
    test = cv2.dilate(test,chotakernel1,iterations = 1)
    cv2.imshow('Test',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test = cv2.resize(test,(int(img.shape[0]/3),int(img.shape[1]/3)))

cv2.imshow('Test',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape[0])
print(img.shape[1])
#img =cv2.resize(img,(45,45)
#img = cv2.resize(img,(4*img.shape[0],4*img.shape[1]))
#
#
#
#_,test =cv2.threshold(img,th,255,cv2.THRESH_BINARY)
#test = cv2.erode(test,kernel,iterations = 1)#####
#test = cv2.dilate(test,kernel,iterations = 1)#####
#
#test = cv2.dilate(test,kernel2,iterations = 1)
#test = cv2.resize(test,(int(img.shape[0]/4),int(img.shape[1]/4)))
#%%
_,contours, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    print(cv2.contourArea(c))
#%%
#cv2.drawContours(img,contours,-1,(0,0,255),2)
#cv2.imshow('frame',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#x=cv2.contourArea(contours[0])
#%%
font = cv2.FONT_HERSHEY_SIMPLEX
index = 0
xbox=[]
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    
    if(cv2.contourArea(cnt)>ar  and (x>2 or y>2)):
        #cv2.rectangle(test,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.putText(test,str(cv2.contourArea(cnt)),(x,y-10), font, 0.5,(0,0,0),2)
        #test[y:y+h,x-2:x+w+2]=cv2.dilate(test[y:y+h,x-2:x+w+2],kernel1,iterations=3)
        xbox.append((x,y,w,h))
cv2.putText(test,str(len(xbox)),(20,20), font, 0.5,(0,0,0),2)
cv2.imshow('frame',test)
cv2.waitKey(0)
cv2.destroyAllWindows() 
xbox.sort()
#%%
S = "" 
index=0
prevX,prevY,prevW,prevH = xbox[0]
prevX=prevX-5
prevY=prevY-5
prevW=0
prevH=0
exp = False
for (x,y,w,h) in xbox :
    if(x>prevX+prevW/2): ##if(x>prevX+prevW/2):
            
            
        if(float(w)/float(h)>1.5 ):###div and sub
            test[y:y+h,x-2:x+w+2]=cv2.erode(test[y:y+h,x-2:x+w+2],kernel1,iterations=1)
            test[y:y+h,x-2:x+w+2]=cv2.dilate(test[y:y+h,x-2:x+w+2],kernel1,iterations=1)
            Xin = test[y-100:y+h-3+100,x-2:x+w+2]/255.0
            #Xin = test[y-50:y+h-3+50,x-2:x+w+2]/255.0
            #Xin = preprocess(Xin,w+4,h+97)
            Xin = cv2.resize(Xin,(45,45))
            Xin = np.reshape(Xin,(-1,45,45,1))
            if((prevY+prevH/2)>(y+h)  and x!= xbox[0][0] ):
                if (exp==False):
                    S=S+'**('
                    exp = True
                
            elif((prevY+prevH)<(y+h/2)  and x!= xbox[0][0]):
                if(exp==True):
                    S=S+')'
                    exp=False
        
            prevX,prevY,prevW,prevH = x-2,y-30,w+4,h+60
            #cv2.imwrite(str(index)+".jpg",test[y-30:y+h-3+30,x-10:x+w-3+10])
            p=np.argmax(model.predict(Xin))
            print(p)
            S=S+CATEGORIES[p]
            cv2.rectangle(test,(x-10,y-100),(x+w-3+10,y+h-3+100),(0,255,0),2)
            cv2.putText(test,CATEGORIES[p],(x-20+int(w/2),y-40), font, 0.5,(0,0,0),2)
            index=index+1
            
        elif(float(h)/float(w)>2.5):##one
            print('else')
            test[y:y+h,x-2:x+w+2]=cv2.erode(test[y:y+h,x-2:x+w+2],chotakernel,iterations=1)
            test[y:y+h,x-2:x+w+2]=cv2.dilate(test[y:y+h,x-2:x+w+2],chotakernel,iterations=1)
            Xin = test[y-8:y+h+4,x-15:x+w+15]/255.0
            #Xin = test[y:y+h,x:x+w]/255.0
            #Xin = preprocess(Xin,w+16,h+8)####
            Xin = cv2.resize(Xin,(45,45))
            Xin = np.reshape(Xin,(-1,45,45,1))
            if((prevY+prevH/2)>(y+h)  and x!= xbox[0][0] ):
                if (exp==False):
                    S=S+'**('
                    exp = True
                
            elif((prevY+prevH)<(y+h/2)  and x!= xbox[0][0]):
                if(exp==True):
                    S=S+')'
                    exp=False

            prevX,prevY,prevW,prevH = x-5,y-5,w+10,h+8
            #cv2.imwrite(str(index)+".jpg",test[y-5:y+h+3,x-20:x+w+20])
            p=np.argmax(model.predict(Xin))
            print(p)
            S=S+CATEGORIES[p]
            #S=S+'1'
            cv2.rectangle(test,(x-15,y-8),(x+w+15,y+h+4),(0,255,0),2)
            cv2.putText(test,CATEGORIES[p],(x-10+int(w/2),y-10), font, 0.5,(0,0,0),2)
            index=index+1
            
    
        else:
            
            #test[y:y+h,x-2:x+w+2]=cv2.erode(test[y:y+h,x-2:x+w+2],chotakernel,iterations=1)
            test[y:y+h,x-2:x+w+2]=cv2.dilate(test[y:y+h,x-2:x+w+2],chotakernel1,iterations=1)
            Xin = test[y-10:y+h+8,x-5:x+w+5]/255.0
            #Xin = preprocess(Xin,w,h)####
            Xin = cv2.resize(Xin,(45,45))
            Xin = np.reshape(Xin,(-1,45,45,1))
            #cv2.imwrite(str(index)+".jpg",feed_dict={x_image:Xin})
            if((prevY+prevH/2)>(y+h)  and x!= xbox[0][0] ):
                if (exp==False):
                    S=S+'**('
                    exp = True
                
            elif((prevY+prevH)<(y+h/2)  and x!= xbox[0][0]):
                if(exp==True):
                    S=S+')'
                    exp=False

            p=np.argmax(model.predict(Xin))
            print(p)
            prevX,prevY,prevW,prevH = x,y,w,h
            S=S+CATEGORIES[p]
            cv2.rectangle(test,(x-5,y-10),(x+w+5,y+h+8),(0,255,0),2)
            cv2.putText(test,CATEGORIES[p],(x-10+int(w/2),y-10), font, 0.5,(0,0,0),2)
            index=index+1
        
#index=index+1
cv2.imshow('frame',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
if(exp==True):
    S=S+')'
print(eval(S))
# (k,l)=xbox[0];
#%%
x,_,w,_ = xbox[len(xbox)-1]
_,y,_,h = xbox[0]
#ans = str(eval())

#%%

def solve(s):
    root1="";root2="";
    a=0;b=0;c=0;
    if(s.find('k')>0):
        a=eval(s[0:s.find('k')]);
    else:
        a=1;
    s=s[s.find('k')+6:];
    print(s);
    if(s.find('k')>1):
        b=eval(s[0:s.find('k')]);
    elif(s.find('k')==-1):
        b=0;
    else:
        if(s[0]=='+' ):
            b=1;
        else:
            b=-1;
    s=s[s.find('k')+1:];
    print(b);
    if(s.find('=') > 1):
        c=eval(s[0:s.find('=')]);
    elif(s.find('=')==0):
        c=0;
    else:
        c=eval(s);
    print(c);
    if(b**2 - 4*a*c >= 0):
        root1=str( (-b+ math.sqrt(b**2 - 4*a*c))/(2.0 *a));
        root2=str( (-b- math.sqrt(b**2 - 4*a*c))/(2.0 *a));
    else:
        root1=str( (-b+ cmath.sqrt(b**2 - 4*a*c))/(2.0 *a));
        root2=str( (-b- cmath.sqrt(b**2 - 4*a*c))/(2.0 *a));
    
    cv2.putText(test,'k='+str(root1)+','+str(root2),(0,y+int(h+20)), font,1.5,(0,0,0),2)
    cv2.imshow('frame',test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(S.find('k')!=-1):
    solve(S);
    
else:
    cv2.putText(test,'='+str(eval(S)),(20,y+int(2*h)), font,1.5,(0,0,0),2)
    cv2.imshow('frame',cv2.resize(test,(int(test.shape[1]/2),int(test.shape[0]/2))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()