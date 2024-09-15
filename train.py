import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import  transforms
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR


from imageloader import customDataset
from network import Encoder


#Seed
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

#### Hyperparemetr Details ######
hash_code = 48
classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration','Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis','Pleural']
numClasses = len(classes)
batch_size = 512
learningRate = 0.0001
epochs = 200
lambda1 = 1
lambda2 = 1.5

distance_file_mname = f'distances_{hash_code}.pkl'
with open(os.path.join('./Datastore/Distances/', distance_file_mname), 'rb') as file:
    distG = pickle.load(file)

def AdaptiveHammingDistance(h, h1, labels, labels1):
    cos = F.cosine_similarity(h, h1, dim=1, eps=1e-6)
    cos_distH = F.relu((1-cos)*hash_code/2)

    sum_label = labels +labels1
    union_label = (sum_label >= 1).sum(dim=1, keepdim=False)
    intersection_label = (sum_label >= 2).sum(dim=1, keepdim=False)
    #relvance_score = torch.div(intersection_label,union_label)+1
    #print(relvance_score)
    predicted_distH = []
    for i,j in zip(union_label.tolist(),intersection_label.tolist()):
        predicted_distH.append(distG[i][j])
    g_distH = torch.tensor(predicted_distH)
    
    return cos_distH.cuda(), g_distH.cuda() #1st one is cosine hamming distance and 2nd one is synamic groundtruth hamming distance

### model uploading #######
encoder = Encoder(numClasses,hash_code)
if torch.cuda.is_available():
    encoder.cuda()
# data prepearing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]) 
trainpath = './Dataset/train/'

trainset = customDataset(trainpath, transform=transform, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=batch_size, num_workers=4,drop_last=True)
print(len(trainset))

### loss functions
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(encoder.parameters(), lr=learningRate, weight_decay= 0.005)
model_scheduler = StepLR(optimizer, step_size=40, gamma=0.4)


classifier_loss_dict = {}
distance_loss_dict = {}

for epoch in tqdm(range(epochs)):
    print ("Epoch:%d/%s."%(epoch+1,epochs))
    running_loss = 0.0 
    distance_running_loss = 0.0
    total_loss =0
    ranking_count_full = 0
    
    encoder.train()
    #remHash.train()
    for i, data in tqdm(enumerate(trainloader)):
        #print(len(data))
        #Data Intializing for trainning
        inputs,labels, index, inputs1, labels1 = data        
        inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
        inputs1, labels1 = Variable(inputs1).cuda(),Variable(labels1).cuda()
        
        
        
        # Initializing model gradients to zero
        # Data feed-forward through the network
        optimizer.zero_grad()#-------------
        output, h = encoder(inputs)
        #print(output)

        output1, h1 = encoder(inputs1)

        loss = criterion(output, labels) + criterion(output1, labels1)#-------- Equation 2
        running_loss += loss #--------------------------------------------------------------------------------------
    
        cos_distH, g_distH = AdaptiveHammingDistance(h, h1, labels, labels1)#1st one is cosine hamming distance and 2nd one is synamic groundtruth hamming distance
        #print(cos_distH)
        #print(g_distH)
        dist_loss = (torch.div(cos_distH-g_distH.float(), hash_code)).cosh().log().sum()
        #print(dist_loss)
        distance_running_loss += dist_loss

        total_loss =     lambda2*loss #lambda1*dist_loss
        total_loss.backward()

        optimizer.step()
        ranking_count_full += 1
        
     
    distance_loss_dict[epoch] = distance_running_loss.item()/(ranking_count_full*batch_size)
    classifier_loss_dict[epoch] = running_loss.item()/(ranking_count_full*batch_size) 


    model_scheduler.step()
    #print(f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

    '''print('Distnce Loss:', distance_loss_dict[epoch])
    print('Train classification Loss:', classifier_loss_dict[epoch])

    #Saving logs

    dataStorePath = './Datastore/Models/'

    distance_loss_path = os.path.join(dataStorePath, 'distance_loss.pkl')
    with open(distance_loss_path, 'wb') as handle:
        pickle.dump(distance_loss_dict, handle)
        print("Saving distance loss log to ", distance_loss_path)
    
    classification_loss_path = os.path.join(dataStorePath, 'classification_loss.pkl')
    with open(classification_loss_path, 'wb') as handle:
        pickle.dump(classifier_loss_dict, handle)
        print("Saving distance loss log to ", classification_loss_path)'''


    '''model_save_path = os.path.join(dataStorePath, f'JaccHash_{hash_code}_{lambda1}_{lambda2}.pkl')
    print(model_save_path)
    torch.save(encoder.state_dict(), model_save_path)
    print("Saving model to ", model_save_path)
    print('------------------model saved-----------------------')'''
    print('------------------------------------------------------------------------------------------------')
