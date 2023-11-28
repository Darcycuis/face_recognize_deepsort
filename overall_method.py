import cv2,os,natsort

        
class Image_Capture:
    def __init__(self,source) -> None:
        super().__init__()
        self.source=os.path.dirname(source) if (source.endswith(".jpg") or source.endswith(".png")) else source
        if os.path.isdir(self.source):
            self.source_type="imgs" 
        elif source.isdigit():
            self.source_type="camera"
            self.source=int(self.source)
        elif source.startswith("rtsp") or source.startswith("rtmp"):
            self.source_type="camera"
        else:
            self.source_type="video"
        self.index=0
        self.ret=True

        if self.source_type == "imgs":
            if (source.endswith(".jpg") or source.endswith(".png")):
                self.img_List=[os.path.basename(source)]
            else:
                self.img_List=natsort.natsorted(os.listdir(source))
            _,img,_=self.read()
            self.index-=1
            self.shape=img.shape
        else:
            self.cap=cv2.VideoCapture(self.source)
        
    def read(self):
        if self.source_type == "imgs":
            img=cv2.imread(os.path.join(self.source,self.img_List[self.index]))
            ret = True if hasattr(img, 'shape') else False
            self.index+=1
            self.ret=ret
            return ret,img,self.img_List[self.index-1]
        elif self.source_type == "camera":
            ret,img=self.cap.read()
            self.index+=1
            self.ret=ret
            return ret,img,"frame_{}.jpg".format(self.index)
        else:
            ret,img=self.cap.read()
            self.ret=ret
            return ret,img
            
    def get(self,i=0):
        if self.source_type == "imgs":
            if i==1:
                return self.index
            if i==7:
                return len(self.img_List)
            if i==4:
                return self.shape[0]
            if i==3:
                return self.shape[1]
            
        elif self.source_type == "camera":
            return self.index if i==1 else int(self.cap.get(i))
        
        else:
            return int(self.cap.get(i))
    
    def get_index(self):
        return self.get(1)
    
    def get_length(self):
        return self.get(7)
    
    def get_height(self):
        return self.get(4)
    
    def get_width(self):
        return self.get(3)
    
    def ifcontinue(self):
        if self.source_type == "imgs":
            return (self.index < len(self.img_List)) and self.ret
        else:
            return (self.cap.get(1) < self.cap.get(7) or self.cap.get(7) <= 0) and self.ret

    def release(self):
        if self.source_type == "imgs":
            pass
        else:
            self.cap.release()
