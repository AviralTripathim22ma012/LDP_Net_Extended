# LDP_Net_Extended
command to run:
~~~bash
!python /content/LDP-Net/test_1.py --n_support 1 --seed 1111 --current_data_path /content/2750 --current_class 10 --test_n_eposide 600 --model_path /content/LDP-Net/checkpoint/100.tar --use_random_crops
~~~

dataset:

~~~bash
!wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
!unzip /content/EuroSAT.zip
~~~