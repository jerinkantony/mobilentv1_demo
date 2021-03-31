import cv2


def determine_text_size(height):
    fraction_val=1/800
    new_text_ht= int(height*fraction_val)
    if new_text_ht<1:
        new_text_ht=1
    return new_text_ht

def put_texts(img, test_tuple_list=[], txt_thickness=3, txt_color=(255, 255, 255), default_align=None, offsetval=0):
    '''
    Given an image, list of texts, display texts on image
    '''
    ht, wd = img.shape[:2]
    l_ct = 1 
    r_ct = 1 
    align_left = None
    text_height=determine_text_size(ht)
    side_margin = 50
    font = cv2.FONT_HERSHEY_SIMPLEX 
    if not len(test_tuple_list):
        return img
    
    for index, txt_tuple in enumerate(test_tuple_list):
        if isinstance(txt_tuple, str):
                text = txt_tuple
                if default_align is None:
                    align_left = True
                if txt_color is None:
                    txt_color = (255, 255, 255)
        
        elif isinstance(txt_tuple, bool):
            text = 'Oclusion {}'.format(txt_tuple)
        
        elif len(txt_tuple)==3:
                text, txt_color, align_left = txt_tuple 
        
        elif len(txt_tuple)==0:
            break     
        
        else:
            text = txt_tuple[0]
            if default_align is None:
                align_left = True
            if txt_color is None:
                txt_color = (255, 255, 255)
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_height, thickness=txt_thickness)
        
        if align_left:
            y_ = 50*(l_ct)
            if offsetval:
                y_+=int(offsetval[1])
                left_gap = int(offsetval[0])
            else:
                left_gap=side_margin  
            l_ct+=1
        else:
            y_ = 50*r_ct
            if offsetval:
                y_+=int(offsetval[1])
                left_gap=int(offsetval[0])
            else:
                left_gap = wd-textSize[0][0]-side_margin
            r_ct+=1
        cv2.putText(img, text, (left_gap, y_), font, text_height, txt_color, txt_thickness, cv2.LINE_AA) 
    return img