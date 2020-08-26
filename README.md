# **LiveFaceReco_RaspberryPi**

```
Face recognition and live estimation on Raspberry Pi 4B with average FPS around 20 and 2800+ faces loaded.
```

## **Update**

**`2020-08-26`**: add ncnn libs (ubunutu, arm64-v8a, armeabi-v7a, and RaspberryPi4B ) to include folder

## Introduction

The project implements **Face Recognition** and **Face Anti Spoofing** on **Raspberry pi** with the models transformed to **ncnn**. Besides, the whole project is designed as an **entrance guard system** by reading face images in the img folder and determining whether the input face is in the dataset by **Arcface**. The most interesting function is that it is capable to estimate whether the face getting from the camera is real **just relaying on the input image** instead of with the help of human body sensors or temperature sensors. As a result, it can avoid the situation of deceived by false faces, including printed paper photos, the display screen of electronic products, silicone masks, 3D human images, etc.


- Neural Network Inference

  [ncnn](https://github.com/Tencent/ncnn)

- Detection:

  [mtcnn](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)

- Recognition: 

  [MobileFaceNet](https://github.com/deepinsight/insightface/issues/214)

-  Anti-Spoofing:

  [Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

## Performance

![image](https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi/blob/master/demo/livedetect.gif)

![image](https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi/blob/master/demo/Mask1.gif)

![image](https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi/blob/master/demo/Mask2.gif)

- The program was run with 2859 faces in img folder, which is enough for a moderate entrance guard system.  

- There is only one image of me(not wearing a mask) in the database, and it is capable to recognize me when wearing a mask(not robust enough). The performance can be improved when retinaface is used as the Detector(TODO).

- The average FPS is around 20, and it successfully recognized me from the database. 


- The number in cyan indicates the score for face recognition, and the number in yellow shows the confidence of live estimation. 

---

## Dependency

- OpenCV >= 4.0.0 

---

## Preparation

- **OpenCV**

  Building OpenCV on Raspberry Pi might be slightly different from that in other linux systems. If you met some problem, you may find [this website](https://blog.csdn.net/weixin_43287964/article/details/101696036?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task) helpful.

- 
   **project_path:**

    set `project_path` in livefacereco.hpp into your own

-  **face database:**

    set ` record_face=true` in livefacereco.hpp to add your face to database, you can rename it in img folder.

---

## Run

make sure you have changed  `project_path` to your own

```shell
mkdir build
cd build
cmake ..
make -j4
./LiveFaceReco
```

---

## Adjustable Parameters

1. **largest_face_only:** only detects the largest face
2. **record_face:** add face to database
3. **distance_threshold:** avoid recognize face which is far away (default 90)
4. **face_thre:** threshold for Recognition (default 0.40)
5. **true_thre:** threshold for Anti Spoofing (default 0.89)
6. **jump:** jump some frames to accelerate
7. **input_width:** set input width (recommend 320)
8. **input_height:** set input height (recommend 240)
9. **output_width:** set output width (recommend 320)
10. **output_height:** set input height (recommend 240)
11. **project_path:** set to your own path

------

## TODO LIST

- [ ] Implement RetinaFace as the detector
- [ ] optimize FPS when output frame is large

---

## Citation

> ```
> @inproceedings{deng2018arcface,
> title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
> author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
> booktitle={CVPR},
> year={2019}
> }
> 
> @inproceedings{deng2019retinaface,
> title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
> author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
> booktitle={arxiv},
> year={2019}
> }
> 
> @inproceedings{ncnn,
> title={ncnn https://github.com/ElegantGod/ncnn},
> author={ElegantGod},
> }
> 
> @inproceedings{Face-Recognition-Cpp,
> title={Face-Recognition-Cpp https://github.com/markson14/Face-Recognition-Cpp},
> author={markson14},
> }
> 
> @inproceedings{insightface_ncnn,
> title={insightface_ncnn https://github.com/KangKangLoveCat/insightface_ncnn},
> author={KangKangLoveCat},
> }
> 
> @inproceedings{Silent-Face-Anti-Spoofing,
> title={Silent-Face-Anti-Spoofing https://github.com/minivision-ai/Silent-Face-Anti-Spoofing},
> author={minivision-ai},
> }
> ```
>
> 







