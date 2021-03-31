import cv2
import numpy as np
import copy
#events = [i for i in dir(cv2) if 'EVENT' in i]
#print (events)
try:
    from draw_utils_ign import put_texts
except:
    from .draw_utils_ign import put_texts

def put_text(image,txt,x0=50,y0=50):

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (x0, y0) 
    # fontScale 
    fontScale = 1   
    # Blue color in BGR 
    color = (0, 255, 0) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    image = cv2.putText(image, txt, org, font,fontScale, color, thickness, cv2.LINE_AA) 
    return image

class MousePts:
    def __init__(self,img=None, windowname='image'):
        self.windowname = windowname
        if img is not None:
            self.img1 = img.copy()
            self.img = self.img1.copy()
            cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
            cv2.imshow(windowname,img)
        self.curr_pt = []
        self.point   = []

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x,y])
            #print(self.point)
            cv2.circle(self.img,(x,y),3,(0,255,0),2)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x,y]
            #print(self.point)
    
    def draw(self,closed=False):
        if len(self.point)>0:
            polys=copy.deepcopy(self.point)
            polys.append(self.curr_pt)
            self.img = cv2.polylines(img=self.img, pts=np.array([polys]), isClosed=closed, color=(255,0,0), thickness=2)

    def drawRect(self):
        if len(self.point)>0:
            start_point = (self.point[0][0],self.point[0][1]) 
            end_point = (self.curr_pt[0],self.curr_pt[1]) 
            color = (255, 0, 0) 
            thickness = 2
            self.img = cv2.rectangle(self.img, start_point, end_point, color, thickness) 


    def getpt(self,count=1,img=None,plotline=True):
        if img is not None:
            self.img = img
            self.img1 = self.img.copy()
        else:
            self.img = self.img1.copy()
        
        txt = 'Click on '+str(count)+' points'
        self.img1 = put_text(self.img1,txt)
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname,self.img)
        cv2.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        k=0
        while(1):
            self.img = self.img1.copy()
            self.draw()
            cv2.imshow(self.windowname,self.img)
            
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point)>=count:
                self.img = self.img1.copy()
                self.draw(closed=True)
                cv2.imshow(self.windowname,self.img)
                break
            #print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args : None)
        #cv2.destroyAllWindows()
        return self.point, self.img
    
    def sanity_check(self, frame, startpt,endpt, x_minimum, x_maximum):
        ix, iy= startpt
        x, y= endpt
        sanity_flag=True
        if (ix < x_minimum) or (x<x_minimum) or (ix>x_maximum) or (x>x_maximum) or (iy < 0) or (y<0) or (iy>frame.shape[0]) or (y>frame.shape[0]):
            sanity_flag=False
        return sanity_flag
        
    def selectRect(self,img,windowname='RectSelect', boxList=None, x_min=None, x_max=None):

        self.img = img
        self.img1 = self.img.copy()
        self.windowname = windowname

        txt = 'Click on corner points, Drag to select, r to reset, Enter to finish, Esc to quit'

        self.img1 = put_texts(self.img1, [[txt]], offsetval=(50, img.shape[0] - 80))
        if len(boxList)>0:
            for box in boxList:
                x1, y1 = box[:2]
                width, height = box[2:]
                x2, y2 = x1+width, y1+height
                cv2.rectangle(self.img1, (x1, y1), (x2, y2), (0, 255, 0), thickness=6)

        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname,self.img)
        cv2.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        p1=[]
        p2=[]
        k=0
        while(1):
            self.img = self.img1.copy()
            self.drawRect()
            cv2.imshow(self.windowname,self.img)
            
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                return [],[]
            if k == ord('r'):
                self.point = []
            if k == 27:
                break
            
            if len(self.point)>1 and k==13:
            # if len(self.point)==4:# and k==13:
                self.img = self.img1.copy()
                self.drawRect()
                cv2.imshow(self.windowname,self.img)
                p1, p2=self.point[0], self.curr_pt
                #import pdb;pdb.set_trace()
                x1,y1=min(p1[0],p2[0]),min(p1[1],p2[1])
                x2,y2=max(p1[0],p2[0]),max(p1[1],p2[1])
                p1=(x1,y1)
                p2=(x2,y2)
                if x_min is not None and x_max is not None:
                    sanity_check_flag = self.sanity_check(self.img, p1, p2, x_min, x_max)
                    if p1!=p2 and sanity_check_flag==True:
                        break
                    else:
                        print('Please draw inside image only')
                else:
                    break
            
            #print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args : None)
        #cv2.destroyAllWindows()
        return p1, p2

         
if __name__=='__main__':
    img = np.zeros((512,512,3), np.uint8)
    windowname = 'image'
    # coordinateStore = MousePts(img, windowname)

    # pts,img = MousePts(img, windowname).getpt(3)
    # print(pts)
        
    # pts,img = MousePts(img, windowname).getpt(3,img)
    # print(pts)
    mouse_obj=MousePts(windowname=windowname)
    pts1,pts2=mouse_obj.selectRect(img,windowname)
    print("pts1,pts2",pts1,pts2)
    cv2.imshow(windowname,img)
    cv2.waitKey(0)
    

