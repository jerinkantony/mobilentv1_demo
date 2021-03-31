import numpy as np
#np.random.bit_generator = np.random._bit_generator
import cv2
import os
import imutils
import os.path as osp
import random
import sys

def display_image(img, waitkey=30, name='image', destroyflag=False):
    debug_flag = True #False to disable display
    if debug_flag is True:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        #import pdb; pdb.set_trace()
        img_ = img.copy()
        
        img_ = img_.astype('uint8')
        
        cv2.imshow(name, img_)
        k = cv2.waitKey(waitkey)
        if destroyflag:
            cv2.destroyWindow(name)
        if k == 27:
            sys.exit()
    else:
        pass

class Imgaug():
    def __init__(self, mode='sample_opencv_aug'):
        print('Available augment methods are "simple_albumentaion", "strong_albumentaion", "sample_aug", "strong_aug"')
        self.mode = mode
        self.min_max = None
        if mode in ['simple_albumentaion', 'strong_albumentaion']:
            from albumentations import (
            Compose, HorizontalFlip, CLAHE, HueSaturationValue,
                RandomBrightness, RandomContrast, RandomGamma,
                ToFloat, ShiftScaleRotate, IAAPerspective, CLAHE, RandomRotate90,
                Transpose, Blur, OpticalDistortion, GridDistortion,
                IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,IAASharpen, IAAEmboss, Flip, OneOf,RandomSunFlare, 
            RandomFog
            )
        if mode == 'simple_albumentaion':
            self.simple_albu= Compose([
            #HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.5, p=0.5),
            ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.2, rotate_limit=10, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=.5),
            Blur(blur_limit=3, p=.5)
            ],p=.5)
            self.albu = self.simple_albu
        elif mode == 'strong_albumentaion':
            self.strong_albu= Compose([
            #HorizontalFlip(p=0.5),
            RandomContrast(limit=0.5, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            OneOf([
                    #IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.5),
            #RandomRotate90(p=0.5),
            #Flip(p=0.5),
            RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=6, src_color=(255, 255, 255), always_apply=False, p=0.5),
            OneOf([
                MotionBlur(p=.5),
                MedianBlur(blur_limit=3, p=0.5),
                Blur(blur_limit=3, p=0.5),
            ], p=0.5),
         
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.7, rotate_limit=20, p=.5),
            OneOf([
                OpticalDistortion(p=0.5),
                RandomFog(p=0.5),
                #GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.5),
            ], p=0.5),
            #OneOf([
             #   CLAHE(clip_limit=2),
                #IAASharpen(),
                #IAAEmboss()            
            #], p=0.5),
            HueSaturationValue(p=0.3),
            ], p=.6)
            self.albu = self.strong_albu
        elif mode == "sample_opencv_aug":
            self.albu = self.sample_opencv_aug

    def apply(self, img, filename=None):
        if filename:
            min_level, max_level = filename[-1]
            self.min_max = (min_level, max_level)
        if self.mode != "sample_opencv_aug":
            #import pdb;pdb.set_trace()
            img = self.albu(image=img)['image']
        else:
            img = self.albu(img)
        return img

    def apply_func(self):
        print('Available augmentations: simple_albumentaion, strong_albumentaion, sample_aug, strong_aug; Using: ')
        if self.mode=='simple_albumentaion':
            return self.simple_albumentaion
        elif self.mode=='strong_albumentaion':
            return self.strong_albumentaion
        
        else:
            sys.exit('Invalid augmentation option, pls use one of : simple_albumentaion, strong_albumentaion, sample_aug, strong_aug')

    def run_augment(self, img, seg):
        rn = random.randint(0, 3)
        if rn==0:
            img, seg = self.augment_pipeline(img, seg)
            return img, seg
        else:
            return img, seg
    
    def do_sometimes(self, n):
        if n < 1:
            return False
        if n==1:
            return True
        rn = random.randint(0, n)
        if rn==0:
            return True
        else:
            return False

    def sample_opencv_aug(self, img):
        if self.do_sometimes(2):
            if self.do_sometimes(3):
                img,_ = self.randomize_hsv(img)
            
            if self.do_sometimes(3):
                rn2 = random.randint(0, 3)
                if rn2==0:
                    img, _ = self.bileteralBlur(img)
                elif rn2==1:
                    img,_ = self.gausian_blur(img)
            
            if self.do_sometimes(5):
                img, _= self.transformation_image(img)
            
            #if self.do_sometimes(3):
            #    img, seg = self.contrast_stretch(img)
            
            #if self.do_sometimes(3):
            #    img, _ = self.vary_brightness(img)
            
            if self.do_sometimes(4):
                img, __ = self.rotate_image(img, deg=15)
        return img
    
    def custom_aug(self, img):

        img = self.strong_aug(image=img)['image']
        img = self.augment_pipeline2(img)
        return img

    def sample_aug(self, img):
        img = self.sample_img_aug(img)
        return img
    '''
    def add_fogeffect(self, image, seg=None):
        aug = iaa.Fog()
        image_aug = aug.augment_image(image)
        return image_aug, seg
    '''
    '''
    def add_Snowflakes(self, image, seg):
        flake_size = [(random.uniform(0.0001, 0.9), random.uniform(0.0001, 0.9))][0]
        aug = iaa.Snowflakes(flake_size=flake_size, speed=(0.001, 0.03))
        image_aug = aug.augment_image(image)
        return image_aug, seg
    '''
    def flip_image(self, image, seg, dir):#0:horizontal, 1: vertical, -1:both
        image = cv2.flip(image, dir)
        if not seg is None:
            seg = cv2.flip(seg, dir)
        return image, seg

    def padding_image(self, image,seg, topBorder=2,bottomBorder=2,leftBorder=2,rightBorder=2, color_of_border=[0,0,0]):
        image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
            rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
        return image, seg

    def add_light_with_gamma(self, image, seg=None, gamma=1.0):
        gamma = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        invGamma = 1.0 / gamma
        if self.min_max is not None:
            table = np.array([((i / float(self.min_max[1])) ** invGamma) * self.min_max[1]
                        for i in np.arange(0, 256)]).astype("uint8")
        else:
            table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        image=cv2.LUT(image, table)
        return image, seg

    def vary_brightness(self, image, seg=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.min_max is not None:
            # print('Intensity levels: ', self.min_max)
            increase = random.randrange(0, self.min_max[1])
            # print('increase: ', increase)
        else:
            increase = random.randrange(0, 10)
        v = image[:, :, 2]
        v = np.where(v <= 255 - increase, v + increase, 255)
        image[:, :, 2] = v
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, seg

    def gausian_blur(self, image, seg=None, blur=0.30):
        blur = random.choice([0.0, 0.15, 0.2, 0.5])
        image = cv2.GaussianBlur(image,(5,5), blur)
        if not seg is None:
            seg = cv2.GaussianBlur(seg, (5,5), blur)
        return image, seg

    def averageing_blur(self, image, seg= None, shift=4):
        shift = random.choice([2, 3])
        image=cv2.blur(image,(shift,shift))
        if not seg is None:
            seg=cv2.blur(seg,(shift,shift))
        return image, seg

    def median_blur(self, image, seg, shift=4):
        image=cv2.medianBlur(image,shift)
        if not seg is None:
            seg=cv2.medianBlur(seg,shift)
        return image, seg

    def bileteralBlur(self, image, seg = None, d=10,color=220,space=100):
        d = random.choice([1, 4, 7])
        space = random.choice([10, 40, 20])
        color = random.choice([50, 100, 175, 200])
        image = cv2.bilateralFilter(image, d, color,space)
        if not seg is None:
            seg = cv2.bilateralFilter(seg, d, color,space)
        return image, seg

    def sharpen_image(self, image, seg):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        return image, seg

    def emboss_image(self, image, seg):
        kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        image = cv2.filter2D(image, -1, kernel_emboss_1)+28
        return image, seg

    
    def addeptive_gaussian_noise(self, image, seg = None):
        h,s,v=cv2.split(image)
        s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        image=cv2.merge([h,s,v])
        return image, seg

    def salt_image(self, image, seg, p,a):
        noisy=image
        num_salt = np.ceil(a * image.size * p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        noisy[coords] = 1
        return image, seg

    def paper_image(self, image, seg, p,a):
        noisy=image
        num_pepper = np.ceil(a * image.size * (1. - p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        noisy[coords] = 0
        return image, seg

    def img_filter(self, img, seg=None):                   #Calculate image gradient (high contrast pixel)
        x=cv2.Sobel(img,cv2.CV_16S,1,0)
        y=cv2.Sobel(img,cv2.CV_16S,0,1)

        absx=cv2.convertScaleAbs(x)
        absy=cv2.convertScaleAbs(y)
        dist=cv2.addWeighted(absx,0.5,absy,0.5,0)
        return dist, seg

    def salt_and_paper_image(self, image, seg=None, p=0.001,a=0.009):
        
        a = random.choice([0.002, 0.001])
        p = random.choice([0.01, 0.02, 0.05, 0.1])
        noisy=image
        #salt
        num_salt = np.ceil(a * image.size * p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        noisy[tuple(coords)] = 1

        #paper
        num_pepper = np.ceil(a * image.size * (1. - p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        noisy[tuple(coords)] = 0
        return image, seg

    def contrast_image(self, image,seg=None, contrast=128):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, seg

    def edge_detect_canny_image(self, image,seg, th1,th2):
        image = cv2.Canny(image,th1,th2)
        return image, seg

    def grayscale_image(self, image, seg ):
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, seg
    
    def scale_image(self, image,fx,fy, seg=None):
        image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
        if seg is not None:
            seg = cv2.resize(seg,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
        return image, seg
    '''
    def add_cloudlayer(self, img, seg=None):
        aug = iaa.CloudLayer( intensity_mean=(70, 255),
            intensity_freq_exponent=(-2.5, -1.5),
            intensity_coarse_scale=(1, 7),
            alpha_min=(0.01, 0.99),
            alpha_multiplier=(0.3, 0.5),
            alpha_size_px_max=1,
            alpha_freq_exponent=(-8.0, -2.0),
            sparsity=0.1,
            density_multiplier=(0.8, 0.9),
            deterministic=False
            )

        image_aug = aug.augment_image(img)
        return image_aug, seg
    '''
    def translation_image(self, image, seg):
        x = random.randrange(0, 50, 30)
        y = random.randrange(0, 50, 30)
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (cols, rows))
        return image, seg
            
    def rotate_image(self, image, seg=None, deg=10):
        deg = random.randint(-deg, deg)
        # cols, rows = image.shape[:2]
        # M = cv2.getRotationMatrix2D((rows//2,cols//2), deg, 1.0)
        # image = cv2.warpAffine(image, M, (rows, cols), flags=cv2.INTER_NEAREST)
        image = imutils.rotate(image, deg)
        if seg is not None:
            seg = cv2.warpAffine(seg, M, (rows, cols), flags=cv2.INTER_CUBIC) #cv2.INTER_NEAREST)
        return image, seg

    def transformation_image(self, image, seg=None):
        cols, rows = image.shape[:2]
        trh = int(cols*0.2)
        trv = int(rows*0.2)
        src_r = rows-trv
        src_c = cols-trh
        def rny(n):
            return n +random.randint(-trh, trh)
        def rnx(n):
            return n +random.randint(-trv, trv)    
        src_pts=np.float32([[rnx(trh), rny(trv)], [rnx(src_c), rny(trv)], [rnx(src_c), rny(src_r)], [rnx(trv), rny(src_r)] ])
        dst_pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows] ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(image, M, (rows, cols))
        return image, seg

    def contrast_stretch(self, img, seg=None):
        xp = [0, 255]
        if self.min_max is not None:
            fp = [random.randint(0, self.min_max[0]), random.randint(self.min_max[1], 255)]
        else:
            fp = [90, 190]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img = cv2.LUT(img, table)
        return img, seg

    def randomize_hsv(self, img, seg = None, hv=3,sv=20,vv=20): #hue,saturation,value
            
        hv=random.randint(-hv,hv)
        sv=random.randint(-sv,sv)
        vv=random.randint(-vv,vv)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h = np.float_(h)
        s = np.float_(s)
        v = np.float_(v)
        
        h+=hv
        s+=sv
        v+=vv
        
        h= h%179
        
        h[h < 0] = 0

        s[s > 255] = 255
        s[s < 0] = 0

        v[v > 255] = 255
        v[v < 0] = 0

        h=h.astype(np.uint8)
        s=s.astype(np.uint8)
        v=v.astype(np.uint8)

        '''
        lim = 255 - sv
        s[s > lim] = 255
        s[s <= lim] += sv

        lim = 255 - vv
        v[v > lim] = 255
        v[v <= lim] += vv
        '''

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img, seg
    
    def strong_img_aug(self, img, seg=None):
        '''
        if self.do_sometimes(5):
        	img, seg = self.add_Snowflakes(img, seg)
        '''
        '''
        if self.do_sometimes(4):
        	img, seg = self.add_fogeffect(img, seg)
        '''
        '''
        if self.do_sometimes(1):
        	img, seg = self.add_cloudlayer(img, seg)
        '''
        '''
        if self.do_sometimes(2):
        	img, seg = self.flip_image(img, seg, 1)
        '''
        if self.do_sometimes(10):
            img, seg = self.padding_image(img, seg)
        if self.do_sometimes(10):
            img, seg = self.salt_and_paper_image(img, seg)
    
        if self.do_sometimes(3):
            img, seg = self.transformation_image(img)
        '''
        if self.do_sometimes(4):
            img, seg = self.randomize_hsv(img, seg)
            print('randomize_hsv')
        '''
        if self.do_sometimes(3):
            img, seg = self.translation_image(img, seg)
        if self.do_sometimes(6):
            rn2 = random.randint(0, 3)
            if rn2==0:
                img, seg = self.bileteralBlur(img, seg)
            elif rn2==1:
                img, seg = self.gausian_blur(img, seg)
            else:
                img, seg = self.averageing_blur(img, seg)
        return img
        
def try_n_times( fn , n , *args , **kargs):
    attempts = 0
    while attempts < n:
        try:
            return fn( *args , **kargs )
        except:
            attempts += 1
    return fn( *args , **kargs )

if __name__=='__main__':
    import glob
    import argparse
    path = osp.dirname(osp.abspath(__file__))
    sys.path.append(path)
    sys.path.append(os.path.join(path, '..', '..', 'ign_utils'))
    from img_utils import Get_Min_Max_intensity_level
    
    aug = Imgaug("sample_opencv_aug")
    # files = glob.glob('/home/ignitarium/projects/cnn_classifier/crops/*.png')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-imfolder', '--imagefolder', default=None, help="Image folder path")
    args = parser.parse_args()

    if args.imagefolder:
        files = glob.glob(osp.join(args.imagefolder, '*'))
        for imgp in files:
            print('Path: ', imgp)
            if imgp.endswith('.png') or  imgp.endswith('.jpg'):
                img1 = cv2.imread(imgp)
                min_val, max_val = Get_Min_Max_intensity_level(img1)
                img2 = img1.copy()
                cv2.imshow("Org Img", img2)
                img_aug = aug.apply(img2, [(min_val, max_val)])
                display_image(img_aug, waitkey=0, name='AugOut')
        
