### Train

- Put pretrained model(ResNet 50 for ImageNet) in pretrained folder

- Command: CUDA_VISEBLE_DEVICES=0,1,... python segmentation/train.py

- If you want to only train segmentation branch, just add nothing

- Else if you want to train both segmentation and instance branch, just add --inst

- Add --resume when you want to resume from the latest checkpoint

### Test

- Put the model you want to test in final folder
- Command: python segmentation/test.py
- If you want to test on cpu, just do nothing
- Else if you want to test on gpu, you need to set use_cuda as True when initialize the model

### Visualization

- We alse support visualization to show a 4 * 4 images of result for user
- This means each row is for one image, and the columns represent orignal image, image with RGB after segmentation and the seperated digits masks(2 images)
- Command: python segmentation/visualization.py