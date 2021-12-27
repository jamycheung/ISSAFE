import os
from tqdm import tqdm
import numpy as np
from mypath import Path

'''
weights of cityscapes 
[ 3.5724095 , 13.22700867,  4.40570363, 36.22547981, 32.7002075 ,
28.2670433 , 45.26888958, 38.04701802,  5.5232027 , 31.79240815,
19.00008406, 25.89065425, 45.75516109,  9.08402964, 44.30799668,
42.57939575, 42.37072459, 47.32382175, 39.76718432]
'''
def calculate_weigths_labels(dataset, dataloader, num_classes, classes_weights_path=None):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # --- class weights
    if not classes_weights_path:
        classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret

def get_erf_weight():
    weight_erf = np.ones(19)
    weight_erf[0] = 2.8149201869965
    weight_erf[1] = 6.9850029945374
    weight_erf[2] = 3.7890393733978
    weight_erf[3] = 9.9428062438965
    weight_erf[4] = 9.7702074050903
    weight_erf[5] = 9.5110931396484
    weight_erf[6] = 10.311357498169
    weight_erf[7] = 10.026463508606
    weight_erf[8] = 4.6323022842407
    weight_erf[9] = 9.5608062744141
    weight_erf[10] = 7.8698215484619
    weight_erf[11] = 9.5168733596802
    weight_erf[12] = 10.373730659485
    weight_erf[13] = 6.6616044044495
    weight_erf[14] = 10.260489463806
    weight_erf[15] = 10.287888526917
    weight_erf[16] = 10.289801597595
    weight_erf[17] = 10.405355453491
    weight_erf[18] = 10.138095855713
    return weight_erf