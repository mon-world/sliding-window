import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 선 selective_search, 후 sliding window
def selective_search(image, method="fast"): 


    """
    image : 선택한 이미지
    method : fast 방식 혹은 slow 방식 선택
    """
   
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
   
    if method == "fast":
       ss.switchToSelectiveSearchFast()
    else:
       ss.switchToSelectiveSearchQuality()
   
    rects = ss.process()
   
    return rects # position 



# selective_search 함수를 통해 나온 튜플 값(네모 박스)에 sliding_window를 수행한다.

def sliding_window(img, model_address):

    """
    img : 선택한 이미지
    model : 기학습된 모델 주소
    """

    # 이미지 자르기
    input_shape = (224,224,3)
    pil_image = cv2.resize(img, dsize = input_shape)

    width = pil_image.shape[1]
    height = pil_image.shape[0]

    crop_w = 20
    crop_h = 20

    move_length = 5

    empty_img = np.zeros((height, width, 3), np.uint8)

    pil_image_xy = selective_search(pil_image, method = 'fast')
    box_num = pil_image_xy.shape[0] # pil_image의 x y w h 묶음 갯수


    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    crop_Image = opencvImage.copy()

    return_list = []
    sum_img_list = []

    for i in range(box_num):
      x = pil_image_xy[i][0]
      y = pil_image_xy[i][1]
      w = pil_image_xy[i][2]
      h = pil_image_xy[i][3]

      range_w = int((w - crop_w) / move_length)
      range_h = int((h - crop_h) / move_length)

      if range_w < 0 and range_h < 0 :
        continue

      for num_w in range(range_w):
        for num_h in range(range_h):
        
          img=crop_Image[move_length * num_h : move_length * num_h + crop_h, move_length * num_w : move_length * num_w + crop_w].copy()
          zeros_img = empty_img.copy()
          zeros_img[move_length * num_h : move_length * num_h + crop_h, move_length * num_w : move_length * num_w + crop_w] = img # crop된 부분을 제외한 나머지가 공백인 이미지

          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          zeros_img = cv2.cvtColor(zeros_img, cv2.COLOR_BGR2RGB)
          im_pil = Image.fromarray(img)
          im_sum = Image.fromarray(zeros_img)

          return_list.append(im_pil)
          sum_img_list.append(im_sum)
  
    # 자른 이미지들 예측하고 분류하기
    # 주소를 불러와 예측을 수행한 다음, crop된 부분을 제외한 나머지가 공백인 이미지를 리스트에 저장한다.
    # 3가지 분류를 가정하였고, fourth_list는 1,2,3에 해당되지 않는 이미지가 저장된다.

    model = load_model(model_address)

    first_list = []
    second_list = []


    for i in range(len(return_list)) :
      prediction = model.predict(return_list[i])
      if np.argmax(i) == 0 : 
        first_list.append(sum_img_list[i])
      elif np.argmax(i) == 1 : 
        second_list.append(sum_img_list[i])
      else : 
          continue

    # 같은 예측끼리 이미지 합치기
    # 0으로 이루어진 배열을 만들고, 그 배열을 업데이트 하는 방식
    # 배열의 한 픽셀에서 그 값보다 더 큰 값이 들어오면, 배열에 포함되지 않은 이미지 픽셀 값이므로, 업데이트 해준다

    first_img_sum = np.zeros((height, width, 3), np.uint8)
    for img in len(first_list) :
      first_list[img] = cv2.cvtColor(np.array(first_list[img]), cv2.COLOR_RGB2BGR)
      for i in range(width) :
        for j in range(height) :
          if  first_list[img][j][i].any() > first_img_sum[j][i].any()  : 
            first_img_sum[j][i] = first_list[img][j][i]
      first_list[img] = cv2.cvtColor(np.array(first_list[img]), cv2.COLOR_BG2RGB)
    first_img_sum = cv2.cvtColor(np.array(first_img_sum), cv2.COLOR_BG2RGB)


    second_img_sum = np.zeros((height, width, 3), np.uint8)
    for img in len(second_list) :
      second_list[img] = cv2.cvtColor(np.array(second_list[img]), cv2.COLOR_RGB2BGR)
      for i in range(width) :
        for j in range(height) :
          if  second_list[img][j][i].any() > second_img_sum[j][i].any()  : 
            second_img_sum[j][i] = second_list[img][j][i]
      second_list[img] = cv2.cvtColor(np.array(second_list[img]), cv2.COLOR_BG2RGB)
    second_img_sum = cv2.cvtColor(np.array(second_img_sum), cv2.COLOR_BG2RGB)





    print('return_list_len :',len(return_list))
    plt.imshow(first_img_sum)
    plt.imshow(second_img_sum)

    return return_list
