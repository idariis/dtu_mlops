"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.img_files = list(Path(path_to_folder).glob("**/*.jpg"))
        self.transform = transform
        
    def __len__(self):
        # return number of images in dataset 
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.img_files[index], mode='r')
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans =  transforms.Compose([transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
                        ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)

    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        #train_features, train_labels = next(iter(dataloader))
        images = next(iter(dataloader))
        plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
        plt.show()
        #pass
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
    
    #fig = plt.figure()
    #plt.errorbar(args.num_workers, np.std(res))
    #plt.show()
