# TF2-Fast-Neural-Style
Implementation of Fast Neural Style in TensorFlow 2 


### Installation
```
git clone https://github.com/lambdal/TF2-Fast-Neural-Style.git
cd TF2-Fast-Neural-Style
virtualenv venv-tf2
. venv-tf2/bin/activate
pip install tf-nightly-gpu-2.0-preview==2.0.0.dev20190611
pip install Pillow

python download_data.py \
--data_url=https://s3-us-west-2.amazonaws.com/lambdalabs-files/mscoco_fns.tar.gz \
--data_dir=~/demo/data
```

### Run the demo

__Train__
```
python fast_neural_style.py train \
--style_image_path=style_image/gothic.jpg \
--train_csv_path=/home/ubuntu/demo/data/mscoco_fns/train2014.csv \
--test_images_path=/home/ubuntu/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg,/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000301397.jpg \
--num_epochs=10 \
--decay_epoch=8,10
```

__Inference__
```
python fast_neural_style.py infer \
--style_image_path=style_image/gothic.jpg \
--test_images_path=/home/ubuntu/demo/data/mscoco_fns/train2014/COCO_train2014_000000003348.jpg,/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000138954.jpg,/home/ubuntu/demo/data/mscoco_fns/val2014/COCO_val2014_000000301397.jpg
```


<p>
<a href="README/gothic.jpg" target="_blank"><img src="README/gothic.jpg" height="240px" style="max-width:100%;"></a>
<a href="README/COCO_val2014_000000301397.jpg" target="_blank"><img src="README/COCO_val2014_000000301397.jpg" height="240px" style="max-width:100%;"></a>
<a href="README/output_COCO_val2014_000000301397.jpg" target="_blank"><img src="README/output_COCO_val2014_000000301397.jpg" height="240px" style="max-width:100%;"></a>
<a href="README/wave_crop.jpg" target="_blank"><img src="README/wave_crop.jpg" height="200px" style="max-width:100%;"></a>
<a href="README/chicago.jpg" target="_blank"><img src="README/chicago.jpg" height="200px" style="max-width:100%;"></a>
<a href="README/output_chicago.jpg" target="_blank"><img src="README/output_chicago.jpg" height="200px" style="max-width:100%;"></a>
</p>