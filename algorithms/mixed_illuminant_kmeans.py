from sklearn.cluster import KMeans
from benchmark import BaseBench
import numpy as np
from images import RGBImage, SpectralImage

def helper(t):
    if(t > 0.008856):
        return t ** (1/3)
    else:
        return 7.787 * t + 16/116

class MixedKmeans(BaseBench):
    NAME = 'mixed_kmeans'

    def output_segments(self):
        if(self.reflectance_map == None):
            self.reflectance_map = self.get_reflectance()
        
        height = np.shape(self.test_img.img_data)[0]
        width = np.shape(self.test_img.img_data)[1]
        
        for k in range(4):
            temp = np.zeros((height, width, 3))
            for i in range(height):
                for j in range(width):
                    if(self.segment[i][j] == k):
                        temp[i, j, :] = self.test_img.img_data[i, j, :]
            temp_temp = RGBImage.NewFromArray(temp)
            temp_temp.dump_file(f'dist/MixedKmeans_sgement_{k}.exr')
    
    def output_ks(self):
        if(self.reflectance_map == None):
            self.reflectance_map = self.get_reflectance()
        print("flag " + str(self.flag))
        for i in range(4):
            print("segment" + str(i) + ":")
            print("maxR: " + str(self.maxR[i]))
            print("maxG: " + str(self.maxG[i]))
            print("maxB: " + str(self.maxB[i]))

    def get_reflectance(self) -> RGBImage:
        
        if(self.reflectance_map != None):
            return self.reflectance_map

        height = np.shape(self.test_img.img_data)[0]
        width = np.shape(self.test_img.img_data)[1]

        #Get linear RGB of each pixel
        R = np.zeros((height, width))
        G = np.zeros((height, width))
        B = np.zeros((height, width))
    
        for i in range(height):
            for j in range(width):
                R[i][j] = self.test_img.img_data[i][j][0]
                G[i][j] = self.test_img.img_data[i][j][1]
                B[i][j] = self.test_img.img_data[i][j][2]

#---------------------------------------------------------------
        #Change linear RGB into CIE XYZ
        X = np.zeros((height, width))
        Y = np.zeros((height, width))
        Z = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                X[i][j] = 0.412453*R[i][j] + 0.357580*G[i][j] + 0.180423*B[i][j]
                Y[i][j] = 0.212671*R[i][j] + 0.715160*G[i][j] + 0.072169*B[i][j]
                Z[i][j] = 0.019334*R[i][j] + 0.119193*G[i][j] + 0.950227*B[i][j]

#---------------------------------------------------------------
        #Change CIE XYZ into CIE L*a*b*
        L = np.zeros((height, width))
        a = np.zeros((height, width))
        b = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                X[i][j] = X[i][j]/0.950456
                Z[i][j] = Z[i][j]/1.088754
                if(Y[i][j] > 0.008856):
                    L[i][j] = 116 * (Y[i][j] ** (1/3)) - 16
                else:
                    L[i][j] = 903.3 * Y[i][j]

        for i in range(height):
            for j in range(width):
                a[i][j] = 500 * (helper(X[i][j]) - helper(Y[i][j]))
                b[i][j] = 200 * (helper(Y[i][j]) - helper(Z[i][j]))

#---------------------------------------------------------------
        #Kmeans on the a*b* coordinate
        #Construct the dataset
        data = []
        for i in range(height):
            for j in range(width):
                data.append((a[i][j], b[i][j]))
        data = np.array(data)

        #Processing Kmeans, predict is the predicted type of the data point
        predict = KMeans(n_clusters = 4, random_state = 0).fit_predict(data)
#---------------------------------------------------------------
        #Calculate the NAAD for each segment
        temp = -1
        self.segment = np.zeros((height, width)) #Store the type of each pixel
        seg = []
        seg.append([]) #Store the pixels who belongs to the sgment 0 
        seg.append([]) #Segment 1
        seg.append([]) #Segment 2
        seg.append([]) #Segment 3

        for i in range(height):
            for j in range(width):
                temp = temp + 1
                self.segment[i][j] = predict[temp]
                seg[predict[temp]].append((i,j))

        #NAAD value for each segment
        NAAD_R = np.zeros(4)
        NAAD_G = np.zeros(4)
        NAAD_B = np.zeros(4)
        for i in range(4):
            N = len(seg[i])
            avg_R = 0
            avg_G = 0
            avg_B = 0
            for j in range(N):
                avg_R = avg_R + R[seg[i][j][0]][seg[i][j][1]]
                avg_G = avg_G + G[seg[i][j][0]][seg[i][j][1]]
                avg_B = avg_B + B[seg[i][j][0]][seg[i][j][1]]
            avg_R = avg_R / N
            avg_G = avg_G / N
            avg_B = avg_B / N
            sum_R = 0
            sum_G = 0
            sum_B = 0
            for j in range(N):
                sum_R = sum_R + abs(avg_R - R[seg[i][j][0]][seg[i][j][1]])
                sum_G = sum_G + abs(avg_G - G[seg[i][j][0]][seg[i][j][1]])
                sum_B = sum_B + abs(avg_B - B[seg[i][j][0]][seg[i][j][1]])
            NAAD_R[i] = sum_R / (N * avg_R)
            NAAD_G[i] = sum_G / (N * avg_G)
            NAAD_B[i] = sum_B / (N * avg_B)
   #-----------------------------------------------------------
        #Based on the threshold, pick the proper segment (The paper said that it's 0.01)
        TR = 0.01
        TG = 0.01
        TB = 0.01
        flag = []
        for i in range(4):
            if(NAAD_R[i] >= TR and NAAD_G[i] >= TG and NAAD_B[i] >= TB):
                flag.append(1)
            else:
                flag.append(0)

        self.flag = flag

        maxR = np.zeros(4)
        maxG = np.zeros(4)
        maxB = np.zeros(4)
        for i in range(4):
            if(flag[i] == 1):
                N = len(seg[i])
                for j in range(N):
                    maxR[i] = max(maxR[i], R[seg[i][j][0]][seg[i][j][1]])
                    maxG[i] = max(maxG[i], G[seg[i][j][0]][seg[i][j][1]])
                    maxB[i] = max(maxB[i], B[seg[i][j][0]][seg[i][j][1]])
        
        KR = np.zeros(4)
        KG = np.zeros(4)
        KB = np.zeros(4)

        for i in range(4):
            KR[i] = 1.0 / maxR[i]
            KG[i] = 1.0 / maxG[i]
            KB[i] = 1.0 / maxB[i]

        self.maxR = maxR
        self.maxG = maxG
        self.maxB = maxB

        #Calculate the center of each segments
        center = []
        for i in range(4):
            center.append([0, 0])
            if(flag[i] == 1):
                N = len(seg[i])
                for j in range(N):
                    center[i][0] = center[i][0] + a[seg[i][j][0]][seg[i][j][1]]
                    center[i][1] = center[i][1] + b[seg[i][j][0]][seg[i][j][1]]
                center[i][0] = center[i][0] / N
                center[i][1] = center[i][1] / N

        #Calculate the Value of KC for each pixel
        p_KR = np.zeros((height, width))
        p_KG = np.zeros((height, width))
        p_KB = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                d = []
                dsum = 0
                for k in range(4):
                    if(flag[k] == 1):
                        d.append(((a[i][j] - center[k][0]) ** 2 + (b[i][j] - center[k][1]) ** 2) ** (1/2))
                        dsum = dsum + d[k]
                    else:
                        d.append(0)
                for k in range(4):
                    if(flag[k] == 1):
                        p_KR[i][j] = p_KR[i][j] + d[k]/dsum*KR[k]
                        p_KG[i][j] = p_KG[i][j] + d[k]/dsum*KG[k]
                        p_KB[i][j] = p_KB[i][j] + d[k]/dsum*KB[k]

        #Times the factor back
        for i in range(height):
            for j in range(width):
                R[i][j] = R[i][j] * p_KR[i][j]
                G[i][j] = G[i][j] * p_KG[i][j]
                B[i][j] = B[i][j] * p_KB[i][j]

        #Return the RGB for the modified picture
        reflectance = np.zeros((height, width, 3))
        reflectance[:, :, 0] = R[:, :]
        reflectance[:, :, 1] = G[:, :]
        reflectance[:, :, 2] = B[:, :]

        self.reflectance_map = RGBImage.NewFromArray(reflectance)
        
        return self.reflectance_map

