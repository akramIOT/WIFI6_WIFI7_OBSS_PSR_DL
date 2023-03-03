
While the PSR  (parameterized spatial reuse  with with coordinated beamforming/null steering) framework allows for a larger spatial reuse, two fundamental challenges have been identified within the 802.11be  WG  forum:

* Devices taking advantage of a spatial reuse /SR opportunity must lower their transmit power to limit the interference generated. In some cases  this translates into a reduced throughput. In other cases  devices cannot even access spatial reuse opportunities as their maximum allowed transmit power is insufficient to reach their receive.  The  focus of this  DL  project  with  Torch and  PyTorch  is  to  simulate, study the effects of  ACI, CCI  on throughput.
* Devices taking advantage of a spatial reuse opportunity are unaware—and have no control over—the interference perceived by their respective receivers  on Rx  side. This  would affect  effective  throughput  in some  HD  WLAN  RF  conditions.
![Illustration-of-the-PSR-framework](https://user-images.githubusercontent.com/21118209/222838584-901a4090-97c8-42db-8a82-0b445d63bdb5.png)


* ![More interference](https://user-images.githubusercontent.com/21118209/186952591-018bb1bd-98a9-4abb-a111-a2d2fbfc52a1.jpeg)


Kindly  refer  these  Publications for further  detailed  study   https://deepai.org/publication/ieee-802-11be-wi-fi-7-strikes-back
https://www.researchgate.net/publication/343546727_IEEE_80211be_Wi-Fi_7_Strikes_Back
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9194746

Kindly also  refer this Project  by  ITU AI/ML Challenge https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-013-ATARI  

DL PROJECT  ENVIRONMENT:
========================
1) Check if  you  have  installed the PyTorch library: https://pytorch.org/get-started/locally/   or  else  install  the latest versions as per  #2 below
2) pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html  (Pip based) or  
3) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  (Conda based)
4) Install matplotlib



DATA:  The input  dataset used in this project is  geenrated  by using  the Komondor open source  Tool,kindly use that https://github.com/wn-upf/Komondor
Reference  link: https://ieeexplore.ieee.org/document/8734225

Dataset:
========
The dataset used during in this project can be downloaded in the following link: https://zenodo.org/record/4106127#.Ykxw3PexXmg
