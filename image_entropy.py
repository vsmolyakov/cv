import matplotlib.pyplot as plt

from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

if __name__ == "__main__":
    
    image = img_as_ubyte(data.camera())
    #image = img_as_ubyte(data.checkerboard())

    #image_entropy = entropy(image,disk(5))    
    image_entropy = entropy(image,disk(10))
    #image_entropy = entropy(image,disk(15))
    
    fig, (ax0, ax1) = plt.subplots(ncols=2,figsize=(10,4))
    
    img0 = ax0.imshow(image, cmap=plt.cm.gray)
    ax0.set_title('image')
    ax0.axis('off')
    fig.colorbar(img0,ax=ax0)
    
    img1 = ax1.imshow(image_entropy, cmap=plt.cm.jet)
    ax1.set_title('entropy')
    ax1.axis('off')
    fig.colorbar(img1,ax=ax1)
    
    plt.show()
    