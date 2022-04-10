import os
from PIL import Image

# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = fileName.replace('.bmp','.jpg')
        print(newFileName)
        im = Image.open(file_path+"\\"+fileName)
        im.save(file_path+"\\"+newFileName)


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "del "+file_path+"\\*."+imageFormat
    os.system(command)


def main():
    file_path = "D:\\BaiduNetdiskDownload\\64_CASIA-FaceV5"
    for i in range(500):
        foldername = str(i).zfill(3)
        final_path = file_path+"\\"+foldername
        bmpToJpg(final_path)
        deleteImages(final_path, "bmp")



if __name__ == '__main__':
    main()
