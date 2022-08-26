
While the PSR framework allows for a larger spatial reuse, two fundamental challenges have been identified within the 802.11be  WG  forum:

* Devices taking advantage of a spatial reuse /SR opportunity must lower their transmit power to limit the interference generated. In some cases  this translates into a reduced throughput. In other cases  devices cannot even access spatial reuse opportunities as their maximum allowed transmit power is insufficient to reach their receive.  The  focus of this  DL  project  with  Torch and  PyTorch  is  to  simulate, study the effects of  ACI, CCI  on throughput.
* Devices taking advantage of a spatial reuse opportunity are unaware—and have no control over—the interference perceived by their respective receivers  on Rx  side. This  would affect  effective  throughput  in some  HD  WLAN  RF  conditions.

Kindly  refer  this  Publication for further  detailed  study   https://deepai.org/publication/ieee-802-11be-wi-fi-7-strikes-back


DL PROJECT  ENVIRONMENT:
========================
1) Check if  you  have  installed the PyTorch library: https://pytorch.org/get-started/locally/   or  else  install  the latest versions as per  #2 below
2) pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html  (Pip based) or  
3) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  (Conda based)
4) Install matplotlib
