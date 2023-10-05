from PIL import Image, ImageDraw
from random import randint

def load_info(filename):
    file = open(filename)
    info = file.read().split('\n')[:-1]
    info = list(map(lambda x: list(map(float, x.split(';'))), info))
    file.close()
    return info

def get_mx(info):
    if isinstance(info[0], list):
        mx = 0
        for i in range(len(info)):
            for j in range(len(info[0])):
                if abs(info[i][j]) > mx:
                    mx = abs(info[i][j])
        return mx
    else:
        mx = 0
        for i in range(len(info)):
            if abs(info[i]) > mx:
                mx = abs(info[i])
        return mx

def contr(info,flt):
    mx = get_mx(info)
    res = []
    if isinstance(info[0], list):
        for i in range(len(info)):
            res.append([])
            for j in range(len(info[0])):
                cl = abs(info[i][j])/mx
                if cl > flt:
                    res[i].append(mx)
                else:
                    res[i].append(0)
    else:
        for i in range(len(info)):
            cl = abs(info[i])/mx
            if cl > flt:
                res.append(mx)
            else:
                res.append(0)
    return res

def se(info, flt):
    mx = get_mx(info)
    s,e = 0,0
    for i in range(len(info)):
        for j in range(len(info[0])):
            cl = abs(info[i][j])/mx
            if cl > flt and s == 0:
                s = i
            if cl > flt and s != 0:
                e = i
    return [s,e]

def imag(info, name='tst', picturesfolder='pictures/'):
    if isinstance(info[0], list):
        width, length = len(info), len(info[0])
        mx = get_mx(info)
        img = Image.new('RGB', (length, width), 'black')
        imgdr = ImageDraw.Draw(img)
        for i in range(width):
            for j in range(length):
                cl = int(abs(info[i][j])/mx*255)
                imgdr.point([j,i], fill=(cl,255-cl,255-cl))
        img.save(picturesfolder+name+'.png')
    else:
        cf = 100
        mx = get_mx(info)
        img = Image.new('RGB', (cf, len(info)), 'black')
        imgdr = ImageDraw.Draw(img)
        for i in range(len(info)):
            for j in range(cf):
                cl = int(abs(info[i])/mx*255)
                imgdr.point([j,i], fill=(cl,255-cl,255-cl))
        img.save(picturesfolder+name+'.png')

def abi(info):
    res = []
    for i in range(len(info)):
        res.append([])
        for j in range(len(info[0])):
            res[i].append(abs(info[i][j]))
    return res

def svrt1(info, filt, flt):
    res = []
    for i in range(len(info)-len(filt)):
        res.append([])
        for j in range(len(info[0])-len(filt[0])):
            sm = 0
            for k in range(len(filt)):
                for n in range(len(filt[0])):
                    sm =+ filt[k][n] * info[i+k][j+n]
            res[i].append(sm)
    if flt!=0:
        return contr(res,flt)
    else:
        return res

def svrt2(info, flt):
    res = []
    for i in range(len(info)-len(info[0])):
        sm = 0
        for j in range(len(info[0])):
            for k in range(len(info[0])):
                sm += info[j+i][k]
        res.append(sm)
    if flt!=0:
        return contr(res,flt)
    else:
        return res

def svrt3(info, rml, flt):
    res = []
    for i in range(len(info)-rml):
        sm = 0
        for j in range(rml):
            sm += info[i+j]
        res.append(sm)
    if flt!=0:
        return contr(res,flt)
    else:
        return res
    

fls = ['5598_13_10_27_1246-13_.csv','5598_13_10_27_1404-33_.csv','5598_14_04_32_1247-13_.csv','5598_14_04_32_1684-19_.csv']
FLT = 0.5 #фильтр для образки(0 - без обрезки; чем ближе к 1 тем сильнее обрезается)
FLT2 = 0.25 #фильтр для контрастной обработки
RML = 101 #длина "рамки" для свертки
SC = 10 # количество сверток
info = load_info(fls[0])
mx = get_mx(info)

imag(info,'orig')

s,e = se(info, FLT)
info = info[s:e+1]

imag(info,'cut')
#print(ram(info,RML,BLT))


flt1 = [[1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]]

"""s1 = svrt1(info,flt1)
imag(s1,'svrt1')

s2 = svrt1(s1,flt1)
imag(s2,'svrt2')

s3 = svrt2(s2)
imag(s3,'svrt3')"""

s0 = svrt2(info, FLT2)
imag(s0,'svrt0')

cur = s0[:]
for i in range(SC):
    print(i+1, 'свертка')
    si = svrt3(cur, RML, FLT2)
    imag(si, f'svrt{i+1}')
    cur = si[:]
