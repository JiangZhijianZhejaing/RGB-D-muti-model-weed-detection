import math
import time

import cv2
import joblib
import numpy as np

#设定常数作为要裁剪图像的依据
LEFT_TOP=103
LEFT_BOTTOM=615
RIGHT_TOP=383
RIGHT_BOTTOM=895
#高度的阈值
HEIGHT_THRESHOLD1=1800
HEIGHT_THRESHOLD2=30
#图像大小
BLOCK_NUM=8 #每边分成8块
BLOCK_SIZE=40
width,height=720,1028

# 1. 读取对应的rgb图像和深度图像,运行时间：0.02
def readFile(rgb_file_name,raw_file_name):
    '''
        读取raw和rgb图像
    '''
    img=cv2.imread(rgb_file_name)
    width,height,C=img.shape
    raw_img = np.fromfile(raw_file_name, dtype=np.uint16)
    # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
    raw_img = raw_img.reshape(width, height)
    raw=img[LEFT_TOP:LEFT_BOTTOM,RIGHT_TOP:RIGHT_BOTTOM,:]
    H,S,I = rgbtohsi(img[LEFT_TOP:LEFT_BOTTOM,RIGHT_TOP:RIGHT_BOTTOM,:])
    raw_img = raw_img[LEFT_TOP:LEFT_BOTTOM,RIGHT_TOP:RIGHT_BOTTOM]
    return H,S,I,raw_img

# 2. rgb图像转为HSI，运行时间0.07
def rgbtohsi(RgbImg):
    '''
        输入图像返回归一化的H,S,I分量
    '''
    B, G, R = cv2.split(RgbImg)
    # 归一化到[0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    # 计算光强为V
    I = (B + G + R) / 3.0  # 防止三通道相加溢出
    # I = ((B + G + R) / 3.0*255).astype(np.uint8)  # 防止三通道相加溢出
    #计算色调H
    num=0.5*(2*R-G-B)
    den=np.sqrt((R-G)*(R-G)+(R-B)*(G-B))
    theta = np.arccos(np.true_divide(num, (den+np.finfo(np.float32).eps))).astype(np.float32)
    H=theta
    Index=np.where(B>G)
    H[Index[0],Index[1]]=(2*np.pi-theta[Index[0],Index[1]])
    H=H/(2*np.pi)
    # H= (H * 255).astype(np.uint8)
    #计算饱和度为S
    S=1-3*np.true_divide(np.minimum(np.minimum(R,G),B),(R+G+B+np.finfo(np.float32).eps))
    # S=(S*255).astype(np.uint8)
    Index = np.where(S ==0 )
    H[Index[0], Index[1]]=0
    # return cv2.merge([I,S, H]) #因为python的BGR的显示问题如果要显示实际请使用此方法
    return H,S,I



# 3. 获取符合要求的图块坐标,运行时间：0.007
def getAim(raw_img):
    '''
        获取土壤占比较小的图像块信息
    '''
    raw_img[np.where(raw_img>HEIGHT_THRESHOLD1)]=0
    maxHight=np.max(raw_img)
    #获取实际的离地高度值
    raw_img=maxHight-raw_img
    #高度小于HEIGHT_THRESHOLD2的部分,我们认为是土壤
    raw_img[np.where(raw_img<HEIGHT_THRESHOLD2)]=0
    #目标区域坐标和高度均值
    aimList=[]
    #自上而下扫描区域，自左向又扫描区域
    for i in range(BLOCK_NUM):
        for j in range(BLOCK_NUM):
            newdata=raw_img[j:j+BLOCK_SIZE,i:i+BLOCK_SIZE]
            #获取土壤信息,如果土壤面积大于块区域的60%则认为是土壤,这一步就是获取土壤像素的值
            num=len(np.where(newdata==0)[0])
            if num>BLOCK_SIZE*BLOCK_SIZE*0.6:
                continue
            s=np.sum(newdata)
            aimList.append([j*BLOCK_SIZE,i*BLOCK_SIZE,RIGHT_TOP+int(BLOCK_SIZE/2)+i*BLOCK_SIZE,width-(LEFT_TOP+int(BLOCK_SIZE/2)+j*BLOCK_SIZE),int(s/(BLOCK_SIZE*BLOCK_SIZE-num))])
    return np.array(aimList)

# 4. 将HSI的三个分量量化为数值方便计算共生矩阵,进行深度判定depth of the decisionm,处理每个图片的特征要0.03秒
def GetAllFeatures(aim_list,H,S,I):
    features=np.zeros((len(aim_list),16),dtype=np.float32)
    for i in range(len(aim_list)):
        H2=H[aim_list[i,0]:aim_list[i,0]+BLOCK_SIZE,aim_list[i,1]:aim_list[i,1]+BLOCK_SIZE].copy()
        S2=S[aim_list[i,0]:aim_list[i,0]+BLOCK_SIZE,aim_list[i,1]:aim_list[i,1]+BLOCK_SIZE].copy()
        I2=I[aim_list[i,0]:aim_list[i,0]+BLOCK_SIZE,aim_list[i,1]:aim_list[i,1]+BLOCK_SIZE].copy()
        H2[np.where(H2 <= 0.1)] = 1
        H2[np.where(H2 <= 0.2)] = 2
        H2[np.where(H2 <= 0.3)] = 3
        H2[np.where(H2 <= 0.4)] = 4
        H2[np.where(H2 <= 0.5)] = 5
        H2[np.where(H2 <= 0.7)] = 6
        H2[np.where(H2 <= 0.9)] = 7
        H2[np.where(H2 < 1.0)] = 8
        S2[np.where(S2 <= 0.2)] = 1
        S2[np.where(S2 <= 0.3)] = 2
        S2[np.where(S2 <= 0.4)] = 3
        S2[np.where(S2 <= 0.5)] = 4
        S2[np.where(S2 <= 0.6)] = 5
        S2[np.where(S2 <= 0.8)] = 6
        S2[np.where(S2 <= 0.9)] = 7
        S2[np.where(S2 < 1.0)] = 8
        I2[np.where(I2 <= 0.2)] = 1
        I2[np.where(I2 <= 0.3)] = 2
        I2[np.where(I2 <= 0.4)] = 3
        I2[np.where(I2 <= 0.5)] = 4
        I2[np.where(I2 <= 0.6)] = 5
        I2[np.where(I2 <= 0.8)] = 6
        I2[np.where(I2 <= 0.9)] = 7
        I2[np.where(I2 < 1.0)] =  8
        CCM_HS,CCM_HI,CCM_SI=getColorMatrix(H2,S2,I2)#拼接得深度判定结果，拼接颜色共生矩阵
        features[i,0],features[i,1],features[i,2],features[i,3],features[i,4]=getCoFeature(CCM_HS)
        features[i,5],features[i,6],features[i,7],features[i,8],features[i,9]=getCoFeature(CCM_HI)
        features[i,10],features[i,11],features[i,12],features[i,13],features[i,14]=getCoFeature(CCM_SI)
        features[i, 15]=aim_list[i,4]
    return features


#获取颜色共生矩阵 ,单个颜色共生矩阵的时间:1.2272491455078125
def getColorMatrix(H,S,I):
    width,height=H.shape
    N=8 #自定义的深度
    # CCM_HS=np.zeros((N,N),dtype=np.uint8)
    # CCM_HI=np.zeros((N,N),dtype=np.uint8)
    # CCM_SI=np.zeros((N,N),dtype=np.uint8)
    # for i in range(1,N+1):
    #     for j in range(1,N+1):
    #         for m in range(width):
    #             for n in range(height):
    #                 if H[m,n]==i-1 and S[m,n]==j-1:
    #                     CCM_HS[i-1, j - 1] = CCM_HS[i - 1, j - 1] + 1
    #                 if S[m,n]==i-1 and I[m,n]==j-1:
    #                     CCM_SI[i - 1, j - 1] = CCM_SI[i - 1, j - 1] + 1
    #                 if H[m,n]==i-1 and I[m,n]==j-1:
    #                     CCM_HI[i - 1, j - 1] = CCM_HI[i - 1, j - 1] + 1
    # return CCM_HS,CCM_HI,CCM_SI
    CCM_HS0=np.zeros((N,N),dtype=np.uint32)
    CCM_HI0=np.zeros((N,N),dtype=np.uint32)
    CCM_SI0=np.zeros((N,N),dtype=np.uint32)
    for m in range(1,width):
        for n in range(1,height):
            CCM_HS0[int(H[m,n])-1, int(S[m, n])-1] += 1
            CCM_SI0[int(S[m,n])-1, int(I[m,n])-1] += 1
            CCM_HI0[int(H[m,n])-1, int(I[m,n])-1] += 1
    return CCM_HS0,CCM_HI0,CCM_SI0


#获取特征
def getCoFeature(CCM):
    '''
        计算纹理特征:对比度、角二阶矩
    '''
    M,N=CCM.shape
    # 1.归一化操作
    P=CCM.copy()
    R=np.sum(P)
    if R !=0:
        p=P/R
    else:
        p=P
    # 获取特征-对比度 f1
    f1=0
    for k in range(1,M):
        f11=0
        for i in range(M-k-1):
            f11+=p[i,i+k]+p[i+k,i]
        f1=f1+math.pow(k,2)*f11
    # 获取特征-角二阶矩 f2
    p1=np.power(p,2)
    f2=np.sum(p1)
    # 获取特征-相关系数 f3
    f3=np.nansum(np.corrcoef(p))#非nan的元素相加
    # 获取特征-逆差矩 f4 熵 f5
    f4=0.0
    for i in range(M):
        for j in range(N):
            f4=f4+(1.0 / (1 + math.pow(i-j,2)))* p[i, j]
    p_log_p=np.log(p+np.finfo(np.float32).eps)
    f5=np.sum(p*p_log_p)
    return f1,f2,f3,f4,f5

# 6. 最终分类代码
def ClassifyAim(aim_list,features):
    scaler=joblib.load('model/DataNormalization.m')
    features=scaler.fit_transform(features)
    SVC=joblib.load('model/SVCModel.m')
    y_predict=np.array(SVC.predict(features[:,:]))
    # y_predict=np.array(SVC.predict(features))
    index=np.where(y_predict==1)
    result_list=aim_list[index[0],:]
    # print('')
    return result_list





def ShowResult(img,result_list):
    width, height, channels = BLOCK_SIZE,BLOCK_SIZE,3
    index=1
    f = open('../ShootingResult.txt', "w+")
    f.write('下标\tx坐标\ty坐标\tz坐标\t杂草面积\n')
    for result in result_list:
        # compute the center of the contour
        x = result[1]+RIGHT_TOP
        y = result[0]+LEFT_TOP
        xy = "(%d,%d)" % (result[2], result[3])
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), -1)
        cX,cY=result[2]-20, 720 - result[3]
        cv2.putText(img, xy, (cX,cY), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), thickness=1)
        # size = cv2.contourArea(cnt)
        f.write(str(index) + '\t' + str(cX) + '\t' + str(cY) + '\t'
                + str(result[4]) + '\t' + str(BLOCK_SIZE*BLOCK_SIZE) + '\n')
        index += 1
    f.close()
    cv2.imshow("Weed0", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

# 数据处理部分————获取对应的杂草和小麦的特征和高度信息，保存在文件夹下用于训练
def getData(otherdirList='WeedData/other',weeddirList='WeedData/weed'):
    f = open('DatasWithHeight.txt', 'w+')
    for i in range(448):
        filename=otherdirList+'/'+str(i+1)+'.png'
        rawfilename=otherdirList+'/'+str(i+1)+'.raw'
        #获取图片特征
        print(filename)
        img = cv2.imread(filename)
        width, height, C = img.shape
        raw_img = np.fromfile(rawfilename, dtype=np.uint16)
        raw_img = raw_img.reshape(width, height)
        raw_img[np.where(raw_img > HEIGHT_THRESHOLD1)] = 0
        maxHight = np.max(raw_img)
        # 获取实际的离地高度值
        raw_img = maxHight - raw_img
        # 去掉高度小于40的部分
        raw_img[np.where(raw_img < HEIGHT_THRESHOLD2)] = 0
        num = len(np.where(raw_img == 0)[0])
        high = int(np.sum(raw_img)/(64*64-num))
        H,S,I=rgbtohsi(img)
        H2 = H.copy()
        S2 = S.copy()
        I2 = I.copy()
        H2[np.where(H2 <= 0.1)] = 1
        H2[np.where(H2 <= 0.2)] = 2
        H2[np.where(H2 <= 0.3)] = 3
        H2[np.where(H2 <= 0.4)] = 4
        H2[np.where(H2 <= 0.5)] = 5
        H2[np.where(H2 <= 0.7)] = 6
        H2[np.where(H2 <= 0.9)] = 7
        H2[np.where(H2 < 1.0)] = 8
        S2[np.where(S2 <= 0.1)] = 1
        S2[np.where(S2 <= 0.2)] = 2
        S2[np.where(S2 <= 0.3)] = 3
        S2[np.where(S2 <= 0.4)] = 4
        S2[np.where(S2 <= 0.5)] = 5
        S2[np.where(S2 <= 0.7)] = 6
        S2[np.where(S2 <= 0.9)] = 7
        S2[np.where(S2 < 1.0)] = 8
        I2[np.where(I2 <= 0.1)] = 1
        I2[np.where(I2 <= 0.2)] = 2
        I2[np.where(I2 <= 0.3)] = 3
        I2[np.where(I2 <= 0.4)] = 4
        I2[np.where(I2 <= 0.5)] = 5
        I2[np.where(I2 <= 0.7)] = 6
        I2[np.where(I2 <= 0.9)] = 7
        I2[np.where(I2 < 1.0)] = 8
        CCM_HS, CCM_HI, CCM_SI = getColorMatrix(H2,S2,I2)
        f1,f2,f3,f4,f5= getCoFeature(CCM_HS)
        f.write(str(round(f1,2)) + ' ' + str(round(f2,2)) + ' ' + str(round(f3,2)) + ' ' + str(round(f4,2)) + ' ' + str(round(f5,2)) + ' ')
        f1, f2, f3, f4, f5 = getCoFeature(CCM_HI)
        f.write(str(round(f1,2)) + ' ' + str(round(f2,2)) + ' ' + str(round(f3,2)) + ' ' + str(round(f4,2)) + ' ' + str(round(f5,2)) + ' ')
        f1, f2, f3, f4, f5 = getCoFeature(CCM_SI)
        f.write(str(round(f1,2)) + ' ' + str(round(f2,2)) + ' ' + str(round(f3,2)) + ' ' + str(round(f4,2)) + ' ' + str(round(f5,2))+' '+str(high)+' -1\n')
    for i in range(460):
        filename = weeddirList + '/'+str(i + 1) + '.png'
        rawfilename = weeddirList +'/'+ str(i + 1) + '.raw'
        print(filename)
        if i==220:
            print()
        # 获取图片特征
        img = cv2.imread(filename)
        width, height, C = img.shape
        raw_img = np.fromfile(rawfilename, dtype=np.uint16)
        raw_img = raw_img.reshape(width, height)
        raw_img[np.where(raw_img > HEIGHT_THRESHOLD1)] =0
        maxHight = np.max(raw_img)
        # 获取实际的离地高度值
        raw_img = maxHight - raw_img
        # 去掉高度小于40的部分
        raw_img[np.where(raw_img < HEIGHT_THRESHOLD2)] = 0
        num = len(np.where(raw_img == 0)[0])
        high2 = int(np.sum(raw_img) / (64 * 64 - num))
        H, S, I = rgbtohsi(img)
        H2 = H.copy()
        S2 = S.copy()
        I2 = I.copy()
        H2[np.where(H2 <= 0.1)] = 1
        H2[np.where(H2 <= 0.2)] = 2
        H2[np.where(H2 <= 0.3)] = 3
        H2[np.where(H2 <= 0.4)] = 4
        H2[np.where(H2 <= 0.5)] = 5
        H2[np.where(H2 <= 0.7)] = 6
        H2[np.where(H2 <= 0.9)] = 7
        H2[np.where(H2 < 1.0)] = 8
        S2[np.where(S2 <= 0.1)] = 1
        S2[np.where(S2 <= 0.2)] = 2
        S2[np.where(S2 <= 0.3)] = 3
        S2[np.where(S2 <= 0.4)] = 4
        S2[np.where(S2 <= 0.5)] = 5
        S2[np.where(S2 <= 0.7)] = 6
        S2[np.where(S2 <= 0.9)] = 7
        S2[np.where(S2 < 1.0)] = 8
        I2[np.where(I2 <= 0.1)] = 1
        I2[np.where(I2 <= 0.2)] = 2
        I2[np.where(I2 <= 0.3)] = 3
        I2[np.where(I2 <= 0.4)] = 4
        I2[np.where(I2 <= 0.5)] = 5
        I2[np.where(I2 <= 0.7)] = 6
        I2[np.where(I2 <= 0.9)] = 7
        I2[np.where(I2 < 1.0)] = 8
        CCM_HS, CCM_HI, CCM_SI = getColorMatrix(H2,S2,I2)
        f1, f2, f3, f4, f5 = getCoFeature(CCM_HS)
        f.write(
            str(round(f1, 2)) + ' ' + str(round(f2, 2)) + ' ' + str(round(f3, 2)) + ' ' + str(round(f4, 2)) + ' ' + str(
                round(f5, 2)) + ' ')

        f1, f2, f3, f4, f5 = getCoFeature(CCM_HI)
        f.write(
            str(round(f1, 2)) + ' ' + str(round(f2, 2)) + ' ' + str(round(f3, 2)) + ' ' + str(round(f4, 2)) + ' ' + str(
                round(f5, 2)) + ' ')

        f1, f2, f3, f4, f5 = getCoFeature(CCM_SI)
        f.write(
            str(round(f1, 2)) + ' ' + str(round(f2, 2)) + ' ' + str(round(f3, 2)) + ' ' + str(round(f4, 2)) + ' ' + str(
                round(f5, 2))+' '+str(high2) +  ' 1\n')
        # f.write(
        #     str(round(f1, 2)) + ' ' + str(round(f2, 2)) + ' ' + str(round(f3, 2)) + ' ' + str(round(f4, 2)) + ' ' + str(
        #         round(f5, 2)) +' '+str(height)+ ' 1\n')
    f.close()
    pass

# def deepLearning(rgb_file_name,raw_file_name):
#     from tensorflow.keras.models import load_model
#     import tensorflow as tf
#     img = cv2.imread(rgb_file_name)
#     width, height, C = img.shape
#     raw_img = np.fromfile(raw_file_name, dtype=np.uint16)
#     # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
#     raw_img = raw_img.reshape(width, height)
#     raw = img[LEFT_TOP:LEFT_BOTTOM, RIGHT_TOP:RIGHT_BOTTOM, :]
#     aim_list = getAim(raw_img)
#     model = load_model('model/ResNet50.h5')
#     testImg=np.zeros((len(aim_list),224,224,3),dtype=np.float32)
#     for index,aim in enumerate(aim_list):
#         imgNnknow=img[LEFT_TOP+aim[0]:LEFT_TOP+aim[0]+64,RIGHT_TOP+aim[1]:RIGHT_TOP+aim[1]+64,:]
#         # imgNnknow = tf.image.resize(imgNnknow, [224, 224])
#         testImg[index,:,:,:]= cv2.resize(imgNnknow, (224, 224), cv2.INTER_LINEAR)
#         result=model.predict(testImg[index,:,:,:])
#         print()

if __name__ == '__main__':
    start_time=time.time()
    rgb_file_name= r'../0311/1-1_Color.png'
    img=cv2.imread(rgb_file_name)
    raw_file_name= r'../0311/1-2_Depth.raw'
    cv2.imshow("img", img[LEFT_TOP:LEFT_BOTTOM,RIGHT_TOP:RIGHT_BOTTOM,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #获取对应的HSI和深度信息
    H,S,I,raw_img=readFile(rgb_file_name,raw_file_name)
    #结合深度信息对土壤影响不大的区域进行区分、识别
    aim_list=getAim(raw_img)
    width,height=H.shape
    #提取特征
    features=GetAllFeatures(aim_list,H,S,I)
    print('处理时间:',time.time()-start_time)
    #获取aim_list的分类结果
    result_list=ClassifyAim(aim_list,features)

    ShowResult(img,result_list)


    # getData()


    # start_time=time.time()
    # rgb_file_name=r'0311/1-1_Color.png'
    # img=cv2.imread(rgb_file_name)
    # raw_file_name=r'0311/1-2_Depth.raw'
    # # cv2.imshow("img", img[LEFT_TOP:LEFT_BOTTOM,RIGHT_TOP:RIGHT_BOTTOM,:])
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # #获取对应的HSI和深度信息
    # deepLearning(rgb_file_name,raw_file_name)
    # print()