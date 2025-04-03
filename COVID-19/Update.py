import torch
def update(input, indice, num):
    # tensor(2D) input, tensor indice, tensor(int) num
    j=0;
    for i in indice:
        input[j, i]=num;
        j=j+1;
    return input

def innerproduct(a,b):
    ## tensor (4D) :a, b


    batchsize, channels, width, height=a.shape;
    a = a.permute(0, 2, 3, 1);
    b=b.permute(0,2,3,1);
    out=torch.zeros((batchsize, width, height))
    for i in range(batchsize):
        for j in range(width):
            for k in range(height):
                out[i,j,k]=torch.dot(a[i,j,k,:],b[i,j,k,:]);
    return out
def outputfile(a):
    batchszie, w,h,=a.shape;
    for i in range(batchszie):
        with open("./featuremap.txt",'w+') as f:
            f.write(str(a[i])+'\n');

def regions_num_decision(vector, threshold):
    # tensor(1D) vector
    # float threshold
    ratio=0;
    i=0;indice=None;selectedregions=None;
    while ratio<threshold:
        i=i+1;
        selectedregions, indice=torch.topk(vector, k=i,dim=0,largest=True, sorted=True);
        ratio=torch.div(torch.sum(selectedregions),torch.sum(vector)).data.item();
    return i,indice,selectedregions
def probability_to_score(probability_vector, threshold):
    # ndarray probability_vector (2D);
    # flaot threshold
    batchsize, length=probability_vector.shape;
    label=[];
    for i in range(batchsize):
        if probability_vector[i,0]>threshold:
            label.append(0);
        else:
            label.append(1);
    return torch.tensor(label,dtype=torch.int64);






