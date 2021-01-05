
import numpy as np
from numpy import pi, sin, cos, sqrt

deg2rd = pi/180
rd2deg = 180/pi

UnitMat = np.array([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), ( 0.0 , 0.0 , 1.0)])

def normalizeHomogenXY(a):
    s = a[..., 2]
    return a / s

def normalizeHomogenMatrix(M):
    return M / M[2,2]


def gridLine(w=640, h=480, n=4, margeHW=(3, 4)):
    mh, mw = margeHW

    x0, x1 = mw, w-mw
    y0, y1 = mh, h-mh

    dx = (x1 - x0) / n
    dy = (y1 - y0) / n

    gxys = list() # of xy
    for i in range(n+1): # horizontal lines
        y = y0 + dy * i
        xa, xb = (x0, x1) if i % 2 == 0 else (x1, x0)
        gxys.append((xa, y))
        gxys.append((xb, y))

    for i in range(n+1): # vertical   lines
        x = x0 + dx * i
        ya, yb = (y0, y1) if i % 2 != 0 else (y1, y0)
        gxys.append((x, ya))
        gxys.append((x, yb))

    return gxys


def gridImg(w=640, h=480, n=4, margeHW=(3, 4)):
    mh, mw = margeHW
    #print("gridImg wxh:%dx%d"%(w, h))
    img = np.ones((h, w)) * 0.7 # light gray background

    x0, x1 = mw, w-mw
    y0, y1 = mh, h-mh
    #print("x0:%d, x1:%d, y0:%d, y1:%d"%(x0, x1, y0, y1))

    img[y0:y1, x0:x1] = 1.0

    gw, gh = x1 - x0, y1 - y0
    dx = gw / n
    dy = gh / n

    e = 1

    for i in range(n+1): # horizontal lines
        y = y0 + int(dy * i)
        #print("y:%3d, x0:%3d:x1:%3d"%(y, x0, x1))
        img[y-e:y+e+1, x0:x1] = 0

    for i in range(n+1): # vertical   lines
        x = x0 + int(dx * i)
        #print("x:%3d, y0:%3d:y1:%3d"%(x, y0, y1))
        img[y0:y1, x-e:x+e+1] = 0

    #print("gridImg.shape:%s"%str(img.shape))
    return img


def gridLineImg(xys, n=4, marge=0.02):
    zoom = 1/(1 - 2*marge) # left and right

    x0, y0 = xys[0,:2]
    x1, y1 = xys[2,:2]
    dx = (x1 - x0) / n
    dy = (y1 - y0) / n

    imgH, imgW = (y1 - y0) * zoom, (x1 - x0) * zoom
    img = np.ones((int(np.ceil(imgH)), int(np.ceil(imgW))))

    gxys = list() # of xy
    for i in range(n+1): # horizontal lines
        y = y0 + dy * i
        img[int(y), int(x0):int(x1)+1] = 0
        xa, xb = (x0, x1) if i % 2 == 0 else (x1, x0)
        gxys.append((xa, y))
        gxys.append((xb, y))

    for i in range(n+1): # vertical   lines
        x = x0 + dx * i
        img[int(y0):int(y1)+1, int(x)] = 0
        ya, yb = (y0, y1) if i % 2 != 0 else (y1, y0)
        gxys.append((x, ya))
        gxys.append((x, yb))

    print("grid_xys     :%s"%gxys)
    print("gridImg.shape:%s"%str(img.shape))
    return gxys, img


def mat2str(M, imDiag):
    xx, yx, dx = M[0, :]
    xy, yy, dy = M[1, :]
    xs, ys, s1 = M[2, :]
    assert s1 == 1.0
    zoom = np.sqrt(xx*yy - xy*yx)
    a_rd = np.arctan2(yx, xx)
    return "zoom:%5.2f, rot:%6.1f, dxy:(%4.0f, %4.0f), xs:%5.2f, ys:%5.2f"%(zoom, a_rd*rd2deg, dx, dy, xs*imDiag, ys*imDiag)

def getHomogenTransformMatrix(xys, xyrs, coefss=None, debug=0):
    if debug >=1: print("getHomogenTransformMatrix:\nxys :\n%s\nxyrs:\n%s"%(xys, xyrs))
    assert xyrs.shape == xys.shape, "xyrs.shape:%s != xys.shape:%s"%(xyrs.shape, xys.shape)
    nPts, w = xys.shape

    nu  = 8 # xx yx dx xy yy dy xs, ys unknows
    neq = nPts * 2 # equations

    assert neq >= nu

    # M(1,nu) * V(nu, neq) = Vr(neq)
    # solve M = Vr / V
    # solve X = b / a

    V    = np.zeros((nu, neq))
    Vr   = np.zeros((neq, 1 ))
    if debug>= 1: print("V.shape:%s"%str(V.shape))

    #coefsX:  {xx: x, yx: y, dx: 1, xy: 0, yy: 0, dy: 0, xs: -x*xr, ys: -xr*y, 1: -xr}
    #coefsY:  {xx: 0, yx: 0, dx: 0, xy: x, yy: y, dy: 1, xs: -x*yr, ys: -y*yr, 1: -yr}
    i = 0
    for iPt in range(nPts):
        if debug>= 2: print("point%d: %-14s -> %-14s"%(iPt, xys[iPt], xyrs[iPt]))
        vxy  = xys [iPt]
        vxyr = xyrs[iPt]

        for j in range(2): # for x and y
            #coefs = coefss[j]
            dj = j * 3
            V[dj:dj+3, i] =  xys[iPt]  # xx, yx, dx
            V[  6    , i] = -vxy[0] * vxyr[j] # xs: -x*xr
            V[  7    , i] = -vxy[1] * vxyr[j] # ys: -y*xr
            Vr[i] = xyrs[iPt, j] # xr or yr
            i += 1

    if debug>= 1:
        print("V :\n%s"%V)
        print("Vr:\n%s"%Vr)

    #M = Vr / V
    if 1:
        mCoefs = np.linalg.solve(V.T, Vr)
    else:
        mCoefs = np.linalg.lstsq(V.T, Vr, rcond=None)[0] # V @ M = Vr
    if debug>= 1: print("mCoefs.shape:%s:\n%s"%(str(mCoefs.shape), mCoefs))


    cxx, cyx, cdx, cxy, cyy, cdy, cxs, cys  = list(mCoefs[:, 0])
    if debug>= 2: print("Coefs:", cxx, cyx, cdx, cxy, cyy, cdy, cxs, cys)

    M = np.array(((cxx, cyx, cdx), (cxy, cyy, cdy), (cxs, cys, 1)))
    if debug>= 1: print("M.shape:%s:\n%s"%(str(M.shape), M))

    return M


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
    n, d = xys.shape
    if d == 2: # transform in homogen
        xyhs = np.ones((n,3))
        xyhs[:,:2] = xys
    else: xyhs = xys
    xyps = xyhs @ projMat.T
    s = xyps[:,2]
    return xyps / s[:,None]


def projectImage(img, projMat, dstBbox, debug=0):
    srcH, srcW = img.shape[:2]
    nColors = None if len(img.shape) < 3 else img.shape[2]

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

    imProjShape = (dstH, dstW) if nColors is None else (dstH, dstW, nColors)
    imgProj = np.zeros(imProjShape, dtype=np.uint8) # usefull image size
    if debug >=1: print("shapes: imgProj:%s %s <- imgProj:%s %s"%(str(imgProj.shape), imgProj.dtype, str(img.shape), img.dtype))
    imgProj[YT-y0Dst, XT-x0Dst, ...] = img[YI, XI, ...] # recadre
    imgProj[imgMask] = 127
    """
    tmp = img[YI, XI]
    if debug >= 0: print("tmp: shape:%s\n%s"%(tmp.shape, tmp[:10, :12]))
    imgProj[YT-y0Dst, XT-x0Dst] = tmp
    if debug >= 2: print("imgProj: shape:%s\n%s"%(imgProj.shape, imgProj[:10, :12]))
    """

    dstXY0 = x0Dst, y0Dst # position of destImg
    return imgProj, imgMask, dstXY0


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

def closestPtIdxDist(xy, line):
    x, y = xy
    xs = line.get_xdata()
    ys = line.get_ydata()
    iOfMin = -1
    for i in range(len(xs)):
        dx, dy = xs[i] - x, ys[i] - y
        d2 = dx**2 + dy**2
        if iOfMin < 0 or d2 < d2min:
            d2min = d2
            iOfMin = i
            dxyMin = dx, dy
    return iOfMin, dxyMin, sqrt(d2min)

import matplotlib.pylab as plt

#-----------------------------------------------------------------------

class ImageView:
    def __init__(self, ax, img=None, nColors=1, title="", frame=True):
        self.ax      = ax
        self.img     = img
        self.nColors = nColors
        self.imgM    = UnitMat
        self.M       = UnitMat
        self.title   = title
        self.frame   = frame

        self.Mstr    = ""
        self.imgMstr = ""
        #print("self.centerCorners: shape:%s:\n%s"%(str(self.centerCorners.shape), self.centerCorners))

        self.initialPlot()

    def initialPlot(self):
        margeHW = margeH, margeW = 20, 25
        if self.img is None: self.img = gridImg(h=401, w=501, margeHW=margeHW)
        self.imgH, self.imgW = self.img.shape[:2]
        self.imDiag = np.sqrt(self.imgH * self.imgW) # size of square with same area
        self.imgFrame = np.array([(0, 0, 1), (self.imgW, 0, 1), (self.imgW, self.imgH, 1), (0, self.imgH, 1)], dtype=np.float)
        dw, dh, x0, x1, y0, y1 = self.imgW/2, self.imgH/2, margeW, self.imgW-margeW, margeH, self.imgH-margeH
        self.centerCorners = np.array([(dw, dh, 1.0), (x0, y0, 1), (x1, y0, 1), (x1, y1, 1), (x0, y1, 1), (x0, y0, 1)], dtype=np.float)

        format_axes(self.ax, self.imgW, self.imgH, self.title + " Image")

        self.initialCenterCorners = self.centerCorners.copy()
        self.initialCenter        = self.centerCorners[0,:].copy()
        self.gridLin_xys  = np.array(gridLine(h=self.imgH, w=self.imgW, margeHW=margeHW))
        self.gridLin_xyrs = self.gridLin_xys.copy()
        self.imgShow = self.ax.imshow(self.img, cmap='gray', interpolation='none')
        if self.frame:
            self.gridLin     , = self.ax.plot(self.gridLin_xyrs [ : ,0], self.gridLin_xyrs [ : ,1],'--b')
            self.quadrilatere, = self.ax.plot(self.centerCorners[1:6,0], self.centerCorners[1:6,1], '-r') # close
            self.handles     , = self.ax.plot(self.centerCorners[0:5,0], self.centerCorners[0:5,1], 'go', picker=10) # frame
            self.centerPt    , = self.ax.plot(self.centerCorners[0  ,0], self.centerCorners[0  ,1], 'r.') # center

    def updateGraphicOverlay(self):
        if self.frame:
            self.gridLin     .set_data(self.gridLin_xyrs [ : ,0], self.gridLin_xyrs [ : ,1])
            self.quadrilatere.set_data(self.centerCorners[1:6,0], self.centerCorners[1:6,1])
            self.handles     .set_data(self.centerCorners[0:5,0], self.centerCorners[0:5,1])
            self.centerPt    .set_data(self.center_xyr[0], self.center_xyr[1])

    def updateTransfMat(self, M, noCornerUpdate=False, debug=0):
        #print("updateTransfMat: noCornerUpdate:%s"%noCornerUpdate)
        if M is None : M = UnitMat
        if (M == self.M).all() :
            if debug >= 2: print("update %s : no change"%self.title)
            return
        #print("update %s.imgM:\nfrom:\n%s\nto:\n%s"%(self.title, self.imgM, M))
        self.Mstr = mat2str(M, self.imDiag)
        self.ax.set_title("%s:imgM:%s\ngridM:%s"%(self.title, self.imgMstr, self.Mstr))
        if debug >= 1: print("update %s.M:\nfrom:%s\nto  :%s"%(self.title, mat2str(self.M, self.imDiag), self.Mstr))
        self.M = M.copy()

        if not noCornerUpdate:
            self.centerCorners = projPts(self.initialCenterCorners, self.M)
            assert self.centerCorners.shape == (6, 3), "centerCorners.shape:%s != (6, 3)"%str(self.centerCorners.shape)

        self.gridLin_xyrs = projPts(self.gridLin_xys, self.M)
        self.center_xyr   = projPt(self.initialCenter, self.M)
        #print("center_xyr: %s"%self.center_xyr)
        #print("centerCornersUpdated => newM:\n%s\ngridLin_xyrs:\n%s"%(self.M.round(3), self.gridLin_xyrs))
        self.updateGraphicOverlay()



    def centerCornersUpdated(self): # and overlay matrix
        #print("centerCornersUpdated:\ninitialCenterCorners:\n%s\npresentCorners:\n%s"%(self.initialCenterCorners, self.centerCorners[1:5, :]))
        M = getHomogenTransformMatrix(self.initialCenterCorners[1:5, :], self.centerCorners[1:5, :], debug=0)
        self.updateTransfMat(M, noCornerUpdate=True)

    def updateImage(self, M, debug=0):
        if M is None : M = UnitMat
        if 0 and (M == self.imgM).all() :
            if debug >= 2: print("update %s.imgM : no change"%self.title)
            return
        self.imgMstr = mat2str(self.imgM, self.imDiag)
        #print("update %s.imgM:\nfrom:\n%s\nto:\n%s"%(self.title, self.imgM, M))
        if debug >= 1: print("update %s.imgM:\nfrom:%s\nto  :%s"%(self.title, mat2str(self.imgM, self.imDiag),mat2str(M, self.imDiag)))

        imgProjBbox = bbox(self.imgFrame)
        if debug >= 1: print("%s imgProjBbox:%s"%(self.title, imgProjBbox))

        self.imgProj, imgMask, dstXY0 = projectImage(self.img, M, dstBbox=imgProjBbox)

        #format_axes(ax, self.imgW, self.imgH, title)
        self.ax.set_xlim(0, self.imgW)
        self.ax.set_ylim(self.imgH, 0)
        self.ax.set_aspect(1.0)


        if dstXY0 == (0, 0):
            extent = None
        else:
            left , top    = dstXY0
            if debug >= 2: print("plotImage at left=x0:%d, top=y0:%d"%(left , top))
            dstH , dstW   = self.imgProj.shape
            right, bottom = left + dstW, top + dstH
            extent = left, right, bottom, top
        if debug >= 1: print("%s img.extent: left, right, bottom, top: %s"%(self.title, extent))

        if 1:
            self.imgShow.set_data(self.imgProj)# segmentedimg
            #self.ax.draw()
        else:
            self.ax.imshow(self.imgProj, cmap='gray', interpolation='none', extent=extent)

        self.imgM = M
        if debug >= 2: print("updateImage done")

    def resetGrid(self):
        self.centerCorners = self.initialCenterCorners.copy()
        self.centerCornersUpdated()

    def updateClickedPos(self, handleIdx, xy):
        dxy = xy - self.centerCorners[handleIdx, :2]
        if handleIdx == 0: # central handle => move all handles
            self.centerCorners[:, :2] += dxy
        else: # move one handle
            self.centerCorners[handleIdx, :2] += dxy
            if handleIdx == 1: # last point(idx:-1) of closed polygone = first point(idx:1)
                self.centerCorners[5, :] =  self.centerCorners[1, :] # skip Center
        #print("updateClickedPos => self.centerCorners:\n%s"%self.centerCorners)
        assert self.centerCorners.shape == (6, 3), "centerCorners.shape:%s != (6, 3)"%str(self.centerCorners.shape)
        self.centerCornersUpdated()

#-----------------------------------------------------------------------------------------

from matplotlib.widgets import Button, CheckButtons

class GUI:
    def __init__(self, gray= False, reduceFactor=None):

        self.imgFile = "castel" # "../../../Images/ParlementDeHambourg0.jpg" # "Numpy.png"
        self.ext = ".jpg"
        img = io.imread(self.imgFile+self.ext)

        self.nColors = 1 if len(img.shape) < 3 else img.shape[2]
        if self.nColors and gray:
            img = (np.sum(img, axis=2) / 3).round().astype(np.uint8)

        print("img.shape:%s"%str(img.shape))

        if reduceFactor is not None:
            assert 0 <= reduceFactor <= 4, "reduceFactor 1,2,3,4 => img size reduction by :4, 16, 64, 256 "
            if reduceFactor > 0 :img = imgReducedBy2powN(img, reduceFactor)

        self.M = UnitMat

        self.fig = plt.figure(figsize=(9, 10))

        ax0 = plt.subplot(2,1,1)
        self.originView = ImageView(ax0, img, self.nColors, "Original")

        ax1 = plt.subplot(2,1,2)
        self.projView   = ImageView(ax1, img, self.nColors, "Tr")

        h, w = self.originView.img.shape[:2]
        M = projMatrix(yHoriz=1.0*w, zoom=1.2, xC=w/2, yC=h*0.8) # , yHoriz=None, zoom=1.0, rotDeg=0.0, xC=0.0, yC=0.0) # hambourg

        self.ind = -1
        self.pickedView = None

        self.callbackRunning = None # for stop error in callbacks

        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('button_press_event'  , self.onButPress  )
        self.fig.canvas.mpl_connect('button_release_event', self.onButRelease)
        self.fig.canvas.mpl_connect('motion_notify_event' , self.onMotion    )

        if 1: #create a Widget
            x, y, w, h = 0.02, 0.97, 0.10, 0.02
            if 1: #create a button
                self.axWidgets = list()
                self.buttons   = list()
                for i, txt in enumerate(("reset grid", "Save Image")):
                    axWidget = plt.axes([x, y-0.03*i, w, h])
                    #print("axWidget:", axWidget)
                    button = Button(axWidget, txt)
                    button.on_clicked(self.onWidget)
                    self.axWidgets.append(axWidget)
                    self.buttons.append(button)
            else: #create a check box
                self.checkBox = CheckButtons(axWidget, ['On',], [False,])

        plt.show()

    def adjustImgToGrid(self):
        invM = np.linalg.inv(self.originView.M)
        imgM = normalizeHomogenMatrix(self.projView.M @ invM)
        self.projView.updateImage(imgM)

    #def accrocheCurseur(self):

    def decrocheCurseur(self):
        self.ind = -1
        self.pickedTraj = None
        self.pickedLine = None

        self.fig.canvas.draw_idle()

    def onpick(self, pickEvt):
        if self.callbackRunning != None:
            print("onpick reenter callbackRunning %s ! => quit"%self.callbackRunning)
            quit()
        self.callbackRunning = "onpick"
        event = pickEvt.mouseevent
        xy = x, y = event.xdata,  event.ydata
        #print("\nonpick: '%s'"%pickEvt.name, ": xy:({0}, {1}) -> xydata:({2:5.3f}, {3:5.3f})".format(event.x,event.y, x, y))
        line = pickEvt.artist
        view = None
        if   line == self.originView.handles: view = self.originView
        elif line == self.projView  .handles: view = self.projView
        else:
            print("not clicked on other => do nothing")
        if view is not None:
            ind, dxy, dist = closestPtIdxDist(xy, line)
            assert 0 <= ind < 5, "0 <= ind: %d < 5 : False !"%ind # 5 handles
            if dist <= 10:
                self.pickedView = view
                self.ind = ind
                self.dxyClick = dxy
            #print("onpick handles: xy:(%d, %d)"%(event.x, event.y))

        self.callbackRunning = None # no exception occurs

    def onWidget(self, event):
        print("onWidget: xy:(%d, %d))"%(event.x, event.y), event)
        idx = None
        for i, ax in enumerate(self.axWidgets):
            if event.inaxes == ax:
                idx = i
                break
        if idx == 0:
            print("resetGrid")
            self.projView.resetGrid()
            self.adjustImgToGrid()
            self.fig.canvas.draw_idle()
        elif idx == 1:
            imgFile = self.imgFile + "_proj" + self.ext
            print("writing %s ..."%imgFile)
            io.imwrite(imgFile, self.projView.imgProj)
            print("done")


    def onButPress(self, event):
        #if event.inaxes==None: return
        #if event.button != 1: return
        if self.callbackRunning != None:
            print("onButPress reenter callbackRunning %s ! => quit"%self.callbackRunning)
            #quit()
        self.callbackRunning = "onButPress"

        print("onButPress: xy:(%d, %d))"%(event.x, event.y))

        self.callbackRunning = None # no exception occurs


    def onMotion(self, event):
        if self.callbackRunning != None:
            print("onMotion reenter callbackRunning %s ! => quit"%self.callbackRunning)
            quit()
        #print("onMotion: event:", event)
        if event.inaxes is None: return # --------------------------->

        self.callbackRunning = "onMotion"

        #print('onMotion: ind:%d,  xy:(%d, %d) -> xydata:(%5.3f, %5.3f)'%(self.ind, event.x,event.y, event.xdata,event.ydata))
        if event.button == 1 and self.ind >= 0: # move point
            x,y = event.xdata, event.ydata
            if self.pickedView is not None:
                #print("onMotion handles: xy:(%d, %d) found idx: %d"%(x, y, self.ind))
                dx, dy = self.dxyClick
                x += dx
                y += dy
                self.pickedView.updateClickedPos(self.ind, (x, y))
                if self.pickedView == self.originView: # report change on projView
                    self.projView.updateTransfMat(normalizeHomogenMatrix(self.originView.M @ self.projView.imgM))
                elif self.pickedView == self.projView: # Change transformation
                    """
                    invM = np.linalg.inv(self.originView.M)
                    imgM = normalizeHomogenMatrix(self.projView.M @ invM)
                    self.projView.updateImage(imgM)
                    """
                    #self.projView.updateImage(projPts(self.originView.centerCorners, self.M))
                    self.adjustImgToGrid()
                    self.stop = False

                self.fig.canvas.draw_idle()
            else:
                print("d:%.1f > 2.0 => decrochage curseur"%d)
                self.decrocheCurseur()

        self.callbackRunning = None # no exception occurs

    def onButRelease(self, event):
        if self.callbackRunning != None and not self.callbackRunning in ("onButPress", ) :
            print("onButRelease reenter callbackRunning %s ! => quit"%self.callbackRunning)
            quit()
        #print("onButRelease: event:", event)
        #if not showverts: return
        if event.button != 1: return
        if self.ind == -1 and self.pickedView == None : return
        self.callbackRunning = "onButRelease"
        #print('onButRelease: xy:({0}, {1}) -> xydata:({2:5.3f}, {3:5.3f})'.format(event.x, event.y, event.xdata, event.ydata))
        self.decrocheCurseur()

        self.callbackRunning = None # no exception occurs

#===========================================================

if __name__ == '__main__':

    gui = GUI(gray=False, reduceFactor=0)


