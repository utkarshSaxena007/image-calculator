import cv2
import numpy as np
s={}
#img=cv2.imread('eq1.png')
lby = np.array([0,0,0])
uby = np.array([0,0,0])
def emptyFunction():
    pass
    
def filterYellow(): 
      
    # blackwindow having 3 color chanels 
    img = cv2.imread('test2.jpg')#np.zeros((512, 512, 3), np.uint8)
    #cv2.imshow('image',img)
    img = cv2.resize(img,(500,500))
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    global lby
    global uby
    
    windowName ="Open CV Color Palette"
    
      
    # window name 
    cv2.namedWindow(windowName)  
       
    # there trackbars which have the name 
    # of trackbars min and max value  
    cv2.createTrackbar('hl', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('sl', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('vl', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('hh', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('sh', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('vh', windowName, 0, 255, emptyFunction) 
       
    # Used to open the window 
    # till press the ESC key 
    while(True): 
        cv2.imshow(windowName, img) 
          
        if cv2.waitKey(1) == 27: 
            break
          
        # values of blue, green, red 
        hl = cv2.getTrackbarPos('hl', windowName) 
        sl = cv2.getTrackbarPos('sl', windowName) 
        vl = cv2.getTrackbarPos('vl', windowName)
        
        hh = cv2.getTrackbarPos('hh', windowName) 
        sh = cv2.getTrackbarPos('sh', windowName) 
        vh = cv2.getTrackbarPos('vh', windowName)
        
        #lby = np.array([hl,sl,vl])
        #uby = np.array([hh,sh,vh])
        lby = np.array([0,0,0])
        uby = np.array([107,107,109])
        mask=cv2.inRange(img,lby,uby)
        ret,mask = cv2.threshold(mask,150,255,cv2.THRESH_BINARY)
           
        cv2.imshow('Det',mask)
        cv2.imshow('Resize',img) 
           
    cv2.destroyAllWindows()
    return mask
grey=filterYellow()
size=np.shape(grey)
print(size)
kernel = np.ones((5,5),np.uint8)
im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    print(x)
    s[x]=cnt
p=sorted(s)
i=0
count=0
for cnt in range(0,len(p)):
    k=s.get(p[cnt])
    x,y,w,h=cv2.boundingRect(k)
    if cv2.contourArea(k)>50 and cv2.contourArea(k)<2000:
        #cv2.rectangle(thresh1,(x,y),(x+w,y+h),(0,255,0),2)
        #print(x,y,w,h)
        #if w>50 and h>50:
        im=grey[y:y+h,x:x+w]
        #im=grey[y-10:y+h+10,x-10:x+w+10]
        #im=grey[y:y+h,x:x+w]
        cv2.imwrite('image'+str(i)+'.jpg',im)
        #cv2.waitKey(2000)
        i=i+1
        count+=1
cv2.imshow('image',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()












import glob
#l=np.empty((50,100),dtype=object)
l=[]
h=[]
lis=[]
img = cv2.imread('digits.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
#print(np.shape(cells))
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

y=x[:,:,3:17,3:17]
for i in range(0,50):
    for j in range(0,100):
        z=y[i,j]
        z=cv2.resize(z,(20,20))
        l.append(z)
d=np.array(l)
d=np.reshape(d,(50,100,20,20))
#print(l[0])
#print(np.shape(d))
# Now we prepare train_data and test_data.
train = d[:,:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
images=glob.glob('C:/Users/Cool Utkarsh/Desktop/python_scripts/data/+/*.jpg')
kernel = np.ones((5,5),np.uint8)
for fname in images:
    p=cv2.imread(fname,0)
    p=255-p[:,:]
    p=cv2.resize(p,(200,200))
    p=cv2.dilate(p,kernel,iterations=2)
    p=cv2.resize(p,(20,20))
    p=p.reshape(-1,400).astype(np.float32)
    train=np.append(train,p,axis=0)
print(np.shape(train))
#test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(11)
train_labels = np.repeat(k,500)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
for k in range(0,count):
    image=cv2.imread('image'+str(k)+'.jpg')
    cv2.imshow('image'+str(k)+'.jpg',image)
    image=cv2.resize(image,(20,20))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    test = image.reshape(-1,400).astype(np.float32)
    ret,result,neighbours,dist = knn.findNearest(test,k=63)
    lis.append(result[0][0])
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
#matches = result==test_labels
#correct = np.count_nonzero(matches)
#accuracy = correct*100.0/result.size
print( lis )
string=''
for i in lis:
    i = str(int(i))
    if(i=='10'):
        i='+'
    string+=i
print(eval(string))

