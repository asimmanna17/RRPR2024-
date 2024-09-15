import torch
import os
import numpy as np
import random
import operator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)

from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from network import Encoder
from metrics import nDCG, aCG, mAPw


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                        'Pleural']

def imageToLabel(image_name):
    labels = image_name.split('_')[2].split('|')
    #print(labels)
    label = [1 if dis in labels else 0 for dis in classes]
    return np.array(label)


def similarityLabel(sorted_pool, q_name):
    sorted_image = [item[0] for item in sorted_pool]
    q_labels = imageToLabel(q_name)
    r_i = []
    for i, _ in enumerate(sorted_image):
        image_labels = imageToLabel(sorted_image[i])
        #print(image_labels)
        #print(q_labels)
        r_i.append(np.dot(image_labels, q_labels))
        #print(r_i)
    sorted_r_i = sorted(r_i,reverse=True)
    return r_i, sorted_r_i



#### Hyperparemetr Details ######
hash_code = 48
lambda1 = 1 
lambda2 = 1.5
numClasses = len(classes)

#model load######################

model = Encoder(numClasses,hash_code)

if torch.cuda.is_available():
    model.cuda()

model_name = f'JaccHash_{hash_code}_{lambda1}_{lambda2}.pkl'
dataStorePath = './Datastore/Models/'
#print(os.listdir(dataStorePath))
model_path = os.path.join(dataStorePath,model_name)
print(model_path)
model.load_state_dict(torch.load(model_path))

#print(model_path)
galleryfolderpath ='./Dataset/gallery'
queryfolderpath = './Dataset/query'
gallery_files = os.listdir(galleryfolderpath)
gallery_files = random.sample(gallery_files, len(gallery_files))
query_files = os.listdir(queryfolderpath)
query_files = random.sample(query_files, len(query_files))
print(len(gallery_files))
querynumber = len((query_files))
print(querynumber)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

gallery = {}
print("\n\n Building Gallery .... \n")
with torch.no_grad():
    # Process each gallery image
    for img in gallery_files:
        image_path = os.path.join(galleryfolderpath, img)

        # Load and transform the image
        image = np.load(image_path)
        # transfer to one channel
        if len(image.shape)!= 2:
            image = np.mean(image,axis=-1)

        image = Image.fromarray(image)
        tensor_image = transform(image).unsqueeze(0).cuda()

        # Pass the tensor through the  model
        x_e, h = model(tensor_image)

        # Store the result in the gallery dictionary
        gallery[img] = torch.sign(h)

        # Clean up
        del tensor_image
    print("\n Building Complete. \n")

    count = 0

    

    nDCG_list = []
    acg_list=[]
    w_map_list=[]

    #print(len(qNimage[0:100]))
    for q_name in query_files:
        count = count+1
        query_image_path = os.path.join(queryfolderpath, q_name)
        # Load and transform the image
        query_image = np.load(query_image_path)
        # transfer to one channel
        if len(query_image.shape)!= 2:
            query_image = np.mean(query_image,axis=-1)
        query_image = Image.fromarray(query_image)
        query_tensor_image = transform(query_image).unsqueeze(0).cuda()

        # Pass the tensor through the  model
        _, h_q = model(query_tensor_image)

        dist = {}
        for key, h1 in gallery.items():
            cos = F.cosine_similarity(h1, torch.sign(h_q), dim=1, eps=1e-6)
            dist[key] = F.relu((1-cos)*hash_code/2)

        print(count)    
        sorted_pool = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
        
        #ndcg
        r_i, sorted_r_i = similarityLabel(sorted_pool, q_name)
        #print(r_i, sorted_r_i)
        nDCG_value = nDCG(r_i, sorted_r_i)
        #print(nDCG_value)
        nDCG_list.append(nDCG_value)
        #ACG
        acg_value = aCG(r_i)
        #print(acg_value)
        acg_list.append(acg_value)
        #WMAP
        wMAP_value = mAPw(r_i)
        #print(wMAP_value)
        w_map_list.append(wMAP_value)

        if count % 10 == 0:
            print('nDCG:', sum(nDCG_list)/len(nDCG_list))
            print('ACG:', sum(acg_list)/len(acg_list))
            print('mAPW:', sum(w_map_list)/len(w_map_list))
    
print('Perfomance for hash code length:', hash_code)
print('nDCG:', sum(nDCG_list)/len(nDCG_list))
print('ACG:', sum(acg_list)/len(acg_list))
print('mAPw:', sum(w_map_list)/len(w_map_list))
print(model_name)

#print('alpha:', alpha)



