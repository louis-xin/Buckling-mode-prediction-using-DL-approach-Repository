from PIL import Image

for i in range(1):
    k = i + 520
    image1 = Image.open('D:\Desktop\Abaqus-python\DL\Data\Buckling Mode\BucklingMode ({}).png'.format(k))
    # print(image1.size)
    
    croppedImage1=image1.crop((806,582,4762,4266))
    
    croppedImage1.save('D:\Desktop\Abaqus-python\DL\Data\Buckling Mode\BucklingMode ({}).png'.format(k))

i = 0
    
#%%

for i in range(1):
    k = i + 520
    image1 = Image.open('D:\Desktop\Abaqus-python\DL\Data\Imperfection\Imperfection ({}).png'.format(k))
    # print(image1.size)
    
    croppedImage1=image1.crop((806,582,4762,4266))
    
    croppedImage1.save('D:\Desktop\Abaqus-python\DL\Data\Imperfection\Imperfection ({}).png'.format(k))

i = 0

#%%
 
# for i in range(504):
#     k = i+1
#     image1 = Image.open('D:\Desktop\Abaqus-python\CollectedData\Data\L-D Plot\L-D plot ({}).png'.format(k))
#     # print(image1.size)
    
#     croppedImage1=image1.crop((806,582,4762,4266))
    
#     croppedImage1.save('D:\Desktop\Abaqus-python\CollectedData\Data\L-D Plot\L-D plot ({}).png'.format(k))

# i = 0

#%%
from PIL import Image

for i in range(520):
    k = i+1
    image1 = Image.open('D:\Desktop\Abaqus-python\DL\Data\Dimensions\Dim_{}.png'.format(k))
    # print(image1.size)
    
    croppedImage1=image1.crop((110,54,610,554))
    
    croppedImage1.save('D:\Desktop\Abaqus-python\DL\Data\Dimensions\Dim_{}.png'.format(k))

i = 0
    