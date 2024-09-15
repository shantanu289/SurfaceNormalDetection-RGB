# SurfaceNormalDetection-RGB <br>

This repository replicates a DORN-like architecture for surface-normal detection using RGB images. <br>
It uses a ResNET-101 backbone. A trans-angular loss is used to determine the principal directions in the image.<br>
The datasets used are : NYU-v2, ScanNet <br>
The figure below compares the performance having trained on 10% of the ScanNet dataset frames against the ground truth. The leftmost images show the predicted principal directions.<br>

![res_fig_4](https://github.com/user-attachments/assets/6a97dbf0-6553-47dc-9dc5-2948af9cd92d)
![res_fig_1](https://github.com/user-attachments/assets/fd306c2e-cfd9-49da-965e-49c8ff2c4998)
