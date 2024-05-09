# Road-Segmentation
Model LINK - https://www.kaggle.com/models/abhigyanryan/road-segmentation <br>
Road_segmentation for Disaster mitigation using CGAN on Satellite imagery  

The Project focuses on road segmentation using a Conditional Generative Adversarial Network (CGAN) with the pix2pix architecture. We utilized satellite images along with their corresponding masks for training. To handle video data, we employed image stitching techniques to create a comprehensive representation. Upon completing the prediction process, the generated masks are applied to the stitched image, effectively segmenting the roads from the rest of the scene. This approach enables accurate road segmentation in satellite imagery, aiding in various applications such as urban planning, traffic analysis, and infrastructure development.

## Results

### Table 1: Accuracy

| Threshold | 0.1  | 0.15 | 0.2  | 0.25 |
|-----------|------|------|------|------|
| Case 1    | 19   | 34   | 49   | 100  |
| Case 2    | 52   | 83   | 95   | 100  |

### Table 2: IOU

| Threshold | 0.25 | 0.5  | 0.85 |
|-----------|------|------|------|
| Case 1    | 0.9976 | 0.9975 | 0.9973 |
| Case 2    | 0.90 | 0.996 | 0.997 |

<p align="center">
Screenshots<br>
<img width="839" alt="Screenshot 2024-05-09 at 12 23 48 AM" src="https://github.com/Abhigyan126/Road-Segmentation/assets/108809711/21a0862d-e150-4249-b798-cf33206fe442"><br>
<img width="839" alt="Screenshot 2024-05-09 at 12 24 17 AM" src="https://github.com/Abhigyan126/Road-Segmentation/assets/108809711/b2406c63-a2e0-46df-9418-960ffffe4b24"><br>
<img width="839" alt="Screenshot 2024-05-09 at 12 24 29 AM" src="https://github.com/Abhigyan126/Road-Segmentation/assets/108809711/9ef69c60-8b50-4092-a089-c7c91f3ee46a"><br>

</p>
