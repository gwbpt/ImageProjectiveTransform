
import numpy as np
from numpy import pi, sin, cos

deg2rd = pi/180

# recherche la matrice de transformation de 2 systeme de coordonnees 2D

# matriceTransformationProjective =
# /xx yx x0\   /x\
# |xy yy y0| x |y|
# \x1 y1  1/   \1 /

# xt = x * xx + y * yx + x0  # yx = poids de y dans x
# yt = x * xy + y * yy + y0
# s  = x * x1 + y * y1 +  1

# xy = xt/s, yt/s # homogene -> 2D

UnitMat = np.array([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), ( 0.0 , 0.0 , 1.0)])

def normalizeHomogenXY(a):
    s = a[..., 2]
    return a / s

def normalizeHomogenMatrix(M):
    return M / M[2,2]

def projMatrix(xHoriz=None, yHoriz=None, zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0, debug=0):
    a = rotDeg * deg2rd
    cosa, sina = cos(a), sin(a)

    # xC, yC : img Center pos (pix unit) generally: xC=imgWidth/2,  yC=imgHeight/2
    xx = yy = zoom * cosa
    yx = zoom * sina
    xy = -yx
    x1 = y1 = 0.0 # no horizon effect
    dx = dy = 0.0

    x1 = 0.0 if xHoriz is None else 1/xHoriz
    y1 = 0.0 if yHoriz is None else 1/yHoriz

    #print("z0**2 + z1**2 : %.2f"%(z0**2 + z1**2))
    M = np.array([(xx, yx, dx), (xy, yy, dy), (x1, y1, 1.0)])
    if 1:
        Tr0 = np.array([(1.0, 0.0, -xC ), (0.0, 1.0, -yC ), ( 0.0 , 0.0 , 1.0)])
        Tr1 = np.array([(1.0, 0.0, +xC ), (0.0, 1.0, +yC ), ( 0.0 , 0.0 , 1.0)])
        M = Tr1 @ M @ Tr0
    if debug >= 2: print("M: shape:%s:\n%s"%(str(M.shape), M))

    Mn = normalizeHomogenMatrix(M)
    if debug >= 1: print("Mn: shape:%s:\n%s"%(str(Mn.shape), Mn))

    return Mn

def inverseHomgenMatrix(M):
    Mi = np.linalg.inv(M)
    #print("Mi: shape:%s:\n%s"%(str(Mi.shape), Mi))

    Mn = normalizeHomogenMatrix(Mi)
    #print("Mn: shape:%s:\n%s"%(str(Mn.shape), Mn))

    return Mn

def projPt(xy, projMat):
    xyp = xy @ projMat.T
    return xyp / xyp[2]

def projPts(xys, projMat):
    xyps = xys @ projMat.T
    s = xyps[:,2]
    return xyps / s[:,None]


def projImage(img, projMat, dstBbox, debug=2):

    srcH, srcW = img.shape

    x0Dst, x1Dst, y0Dst, y1Dst = dstBbox
    dstH, dstW = y1Dst - y0Dst, x1Dst - x0Dst
    if debug >= 1:
        print("dstBbox:", dstBbox)
        print("dstW: %d, dstH: %d"%(dstW, dstH))
        print("srcW: %d, srcH: %d"%(srcW, srcH))

    Mti = inverseHomgenMatrix(projMat) # inverse matrix

    xx, yx, dx = Mti[0, :]
    xy, yy, dy = Mti[1, :]
    x1, y1, _  = Mti[2, :]
    if debug >= 2:
        print("xx:%7.3f, yx%7.3f, dx%7.3f"%(xx, yx, dx))
        print("xy:%7.3f, yy%7.3f, dy%7.3f"%(xy, yy, dy))

    # Transform coord of imgDst pixels (XT, YT) to coord in imgDst (XI, YI)
    XT, YT = np.meshgrid(range(x0Dst, x1Dst), range(y0Dst, y1Dst), sparse=False, indexing='xy') # XY coord in transformed img
    if debug >=3: print("XT:%s\n%s:\nYT:%s:\n%s"%(XT.shape, XT[:10, :12], YT.shape, YT[:10, :12]))

    Xh = XT * xx + YT * yx + dx   #
    Yh = XT * xy + YT * yy + dy   #
    Sh = XT * x1 + YT * y1 + 1.0  # third of homogeneous coordinates

    if debug >=3:
        print("Xh: shape:%s\n%s"%(Xh.shape, Xh[0:-1:8, 0:-1:32]))
        print("Yh: shape:%s\n%s"%(Yh.shape, Yh[0:-1:8, 0:-1:32]))
        print("Sh: shape:%s\n%s"%(Sh.shape, Sh[0:-1:8, 0:-1:32]))

    XI = (Xh / Sh).round().astype(int)
    YI = (Yh / Sh).round().astype(int)

    imgOk   = np.logical_and(np.logical_and(XI >= 0, XI < srcW), np.logical_and(YI >= 0, YI < srcH))# .astype(np.uint8)*255
    imgMask = np.logical_not(imgOk)
    if debug >= 2: print("imgMask: shape:%s\n%s"%(imgMask.shape, imgMask[0:-1:8, 0:-1:32]))

    # Get pixels within image boundaries
    #indices = np.where((x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height))

    XI = np.clip(XI, 0, srcW-1)
    YI = np.clip(YI, 0, srcH-1)
    if debug >= 2:
        print("XI: shape:%s\n%s"%(XI.shape, XI[:10, :12]))
        print("YI: shape:%s\n%s"%(YI.shape, YI[:10, :12]))

    imgProj = np.zeros((dstH, dstW))          # usefull image size
    imgProj[YT-y0Dst, XT-x0Dst] = img[YI, XI] # recadre
    imgProj[imgMask] = 127
    """
    tmp = img[YI, XI]
    if debug >= 0: print("tmp: shape:%s\n%s"%(tmp.shape, tmp[:10, :12]))
    imgProj[YT-y0Dst, XT-x0Dst] = tmp
    if debug >= 2: print("imgProj: shape:%s\n%s"%(imgProj.shape, imgProj[:10, :12]))
    """

    dstXY0 = x0Dst, y0Dst # position of destImg
    return imgProj, imgMask, dstXY0

#===========================================================

if __name__ == '__main__':

    def bbox(xys):
        min_xy, max_xy = xys.min(axis=0), xys.max(axis=0)
        #print("min_xy:%s, max_xy:%s"%(min_xy, max_xy))

        min_xy, max_xy = np.floor(min_xy).astype(np.int16), np.ceil(max_xy).astype(np.int16)
        # print("min_xy:%s, max_xy:%s"%(min_xy, max_xy))

        return min_xy[0], max_xy[0], min_xy[1], max_xy[1] #bbox


    def imgReducedBy2powN(img, n=1):
        for t in range(n):
            h, w = img.shape[0], img.shape[1]
            h = (h//2)*2 # even
            w = (w//2)*2 # even
            imgu16 = img.astype(np.uint16)
            imgu16 = (imgu16[0:h:2, 0:w:2, ...] + imgu16[1:h:2, 0:w:2, ...] + imgu16[0:h:2, 1:w:2, ...] + imgu16[1:h:2, 1:w:2, ...])/4
            img = imgu16.astype(np.uint8)
        return img

    def format_axes(ax, w=4, h=3, title="no title"):
        #ax.margins(0.2)
        #ax.set_axis_off()
        ax.set_aspect(1.0)
        #ax.set_xlim(0, w)
        #ax.set_ylim(0, h)
        ax.invert_yaxis()
        #ax.set_autoscale_on(False)
        ax.set_title(title)

    def gridOfPoints(w=8, h=6, step=1, centred=False):
        #w, h = 2*dw + 1, 2*dh + 1

        dw, dh = (w//2, h//2) if centred else (0, 0)

        #nPts = h * w
        #grid = np.zeros((nPts, 3))

        ys = tuple(range(0, h, step))
        xs = tuple(range(0, w, step))

        gridH, gridW = len(ys), len(xs)
        print("gridH: %d, gridW: %d"%(gridH, gridW))
        xys = list()
        for y in ys:
            i0 = y * w
            y0 = y-dh
            row = list()
            for x in xs:
                xys.append((x-dw, y0, 1))
        grid = np.array(xys, dtype=np.int16)
        #print("grid: shape:%s\n%s"%(grid.shape, grid[:50]))

        return grid

    #------------------------------------------------

    import imageio as io

    srcRGB = io.imread("../../../Images/Numpy.png") #, flatten=True)
    img = (np.sum(srcRGB, axis=2) / 3).round().astype(np.uint8)
    print("img: shape:%s"%str(img.shape))

    img = imgReducedBy2powN(img, 1)
    imgH, imgW = img.shape

    if 1:
        w, h = imgW, imgH
    else:
        dw, dh = 64, 32
        w, h = 2*dw + 1, 2*dh + 1
    mire = np.ones((h, w), dtype=np.uint8)*255
    mire[-1, -1] = 0

    imgCenterCorners = np.array([(imgW/2, imgH/2, 1.0), (0, 0, 1), (w-1, 0, 1), (w-1, h-1, 1), (0, h-1, 1), (0, 0, 1)], dtype=np.float)
    print("imgCenterCorners: shape:%s:\n%s"%(str(imgCenterCorners.shape), imgCenterCorners))

    gridp = gridOfPoints(w, h, 8)
    #print("gridp: shape:%s:\n%s"%(str(gridp.shape), gridp))

    import matplotlib.pylab as plt


    def plotProj(ax, title, M=UnitMat, frame=False, ptsGrid=False, debug=0):

        imgCenterCornersProj = projPts(imgCenterCorners, M)

        imgProjBbox = bbox(imgCenterCornersProj[1:5])
        if debug >= 1: print("imgProjBbox:", imgProjBbox)

        imgProj, imgMask, dstXY0 = projImage(img, M, dstBbox=imgProjBbox)
        gridProj = projPts(gridp, M)

        format_axes(ax, w, h, title)

        if 0 and dstXY0 == (0, 0):
            extent = None
        else:
            left , top    = dstXY0
            print("plotImage at left=x0:%d, top=y0:%d"%(left , top))
            dstH , dstW   = imgProj.shape
            right, bottom = left + dstW, top + dstH
            extent = left, right, bottom, top
        print("extent: left, right, bottom, top: ", extent)

        ax.imshow(imgProj, cmap='gray', interpolation='none', extent=extent)
        if ptsGrid: ax.plot(gridProj[:,0], gridProj[:,1], 'g.' )
        if frame:
            ax.plot(imgCenterCornersProj[0 ,0], imgCenterCornersProj[0 ,1], 'r.') # center
            ax.plot(imgCenterCornersProj[1:,0], imgCenterCornersProj[1:,1], '-r') # frame


    plt.figure(figsize=(14, 6))
    ax = plt.subplot(2,3,2)
    plotProj(ax, "Flat Image")
    if 1: # test
        print("----------------------------- Test ---------------------------------")
        ax = plt.subplot(2,3,6)
        #M = projMatrix(rotDeg=170, zoom=0.7, xC=w/2, yC=h/2) # yHoriz=-10*h, , zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0)
        M = projMatrix(xHoriz=-2*w, xC=w/2, yC=h/2) # , yHoriz=None, zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0)
        plotProj(ax, "Test", M)

    if 1: # 3 projections
        ax = plt.subplot(2,3,1)
        # vertical wall proj with horizon at left x -12.0 pix
        M = projMatrix(xHoriz=-2*w, xC=w/2, yC=h/2) # , yHoriz=None, zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0)
        plotProj(ax, "Left wall", M)

        ax = plt.subplot(2,3,3)
        # vertical wall proj with horizon at right x +12.0 pix
        M = projMatrix(xHoriz=+2*w, xC=w/2, yC=h/2) # , yHoriz=None, zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0)
        plotProj(ax, "Right wall", M)

        ax = plt.subplot(2,3,5)
        # horizontal ground proj with horizon at top  y  -5.0 pix
        M = projMatrix(yHoriz=-2*h, xC=w/2, yC=h/2) # , zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0)
        plotProj(ax, "Ground", M)

    plt.show()


