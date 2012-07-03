from scipy import *
from scipy.linalg import *
from scipy.special import *
from random import choice

from sift import *
from homography import *

def get_points(locs1, locs2, matchscores):
    '''
        Return the corresponding points in both the images
    '''
    plist = []
    t = min(len(locs1), len(locs2))
    for i in range(len(matchscores)):
        if (matchscores[i] > 0):
            y1 = int(locs1[i, 1])
            x1 = int(locs1[i, 0])

            y2 = int(locs2[int(matchscores[i]), 1])
            x2 = int(locs2[int(matchscores[i]), 0])

            plist.append([[x1,y1],[x2,y2]])
    return plist

def ransac(im1, im2, points_list, iters = 10 , error = 10, good_model_num = 5):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''
    
    rows,cols = im1.shape

    model_error = 255
    model_H = None

    for i in range(iters):
        consensus_set = []
        points_list_temp = copy(points_list).tolist()
        # Randomly select 3 points
        for j in range(3):
            temp = choice(points_list_temp)
            consensus_set.append(temp)
            points_list_temp.remove(temp)
        
        # Calculate the homography matrix from the 3 points
        
        fp0 = []
        fp1 = []
        fp2 = []
        
        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:
        
            fp0.append(line[0][0])
            fp1.append(line[0][1])
            fp2.append(1)
            
            tp0.append(line[1][0])
            tp1.append(line[1][1])
            tp2.append(1)
            
        fp = array([fp0, fp1, fp2])
        tp = array([tp0, tp1, tp2])
        
        H = Haffine_from_points(fp, tp)
                            
        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = array([x1, y1, 1]).reshape(3,1)
            B = array([x2, y2, 1]).reshape(3,1)
            
            out = B - dot(H, A)
            dist_err = hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)            
            

        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0]
                x1, y1 = p[1]
                
                A = array([x0, y0, 1]).reshape(3,1)
                B = array([x1, y1, 1]).reshape(3,1)
                
                out = B - dot(H, A)
                dist_err = hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if max(dists) < error:
                model_error = max(dists)
                model_H = H
                        
    return model_H
    
if __name__ == "__main__":
    import Image
    try:
        os.mkdir("temp")
    except OSError:
        pass
    im1 = Image.open('1.jpg').convert('L')
    im2 = Image.open('2.jpg').convert('L')
    im1.save('temp/1.pgm')
    im2.save('temp/2.pgm')
    im1 = asarray(im1)
    im2 = asarray(im2)
    process_image('temp/1.pgm', 'temp/1.key')
    process_image('temp/2.pgm', 'temp/2.key')
    key1 = read_features_from_file('temp/1.key')
    key2 = read_features_from_file('temp/2.key')
    score = match(key1[1], key2[1])
    plist = get_points(key1[0], key2[0], score)
    plot_matches(im1,im2,key1[0],key2[0],score)
    out = ransac(im1, im2, plist)
    print out
