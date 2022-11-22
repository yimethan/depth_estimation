# Monocular depth estimation with 3D object detection

+ 4D cost volume
+ Supervised monocular depth estimation
+ Refine depth estimation with 3D object detection

## B1: Depth estimation of overall image

1. ResNet
2. Generate cost volume
   + Option 1. SSPCV-Net based pyramid cost volume (no semantic segmentation)
     + Option 1-1. with aggregation module after generating cost volume
     + Option 1-2. without aggregation module after generating cost volume
   
   + Option 2. GC-Net based cost volume

     + Option 2-1. with multiscale 3D conv after generating cost volume
     + Option 2-2. without multiscale 3D conv after generating cost volume

## B2: CenterNet based 3D object detection

1. Filter samples with bBox IoU < 0.5
2. Depth estimation of bBox

   + Option 1. Filter bBoxes of depth > threshold
   + Option 2. Normalize depth with center coord of bBox

## B3: B1 + B2 and final depth map

1. Concatenate results of last layers of each B1 and B2, in disparity axis
   + Option 1. Simply concatenate
   + Option 2. Concatenate alternately
2. Follow B1/2/Option 0-2
3. Softmin to generate final depth map