all: mjsynth-download mjsynth-tfrecord train

demo: train

mjsynth-download: mjsynth-wget mjsynth-unpack 

mjsynth-wget:
	mkdir -p data
	cd data ; \
	wget http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz

mjsynth-unpack:
	mkdir -p data/images
# strip leading mnt/ramdisk/max/90kDICT32px/
	tar xzvf data/mjsynth.tar.gz \
    --strip=4 \
    -C data/images

mjsynth-tfrecord:
	mkdir -p data/train data/val data/test 
	cd src ; python mjsynth-tfrecord.py

train:
	cd src ; python train.py # use --help for options

monitor:
	tensorboard --logdir=data/model --port=8008

test:
	cd src ; python test.py # use --help for options

val: captcha validate

validate:
	cd src ;  python validate.py < paths_to_images.txt 

captcha:
	cd data/val ; find . -name "*.jpg" -exec rm {} ";"
	cd CaptchaGenerator ; python val_english_captcha.py 
	cd CaptchaGenerator ; python val_math_captcha.py
	
val_t:
	cd src ; python validate_3.py < final_val.txt
	
val_all: captcha validate val_t