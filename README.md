

# LSDiff: A Diffusion-Guided Level Set Method for Low-Contrast Lesion Boundary Segmentation
LSDiff is a diffusion-probabilistic-model–driven level set segmentation framework. The methodological details can be found in our paper: *LSDiff: A Diffusion-Guided Level Set Method for Low-Contrast Lesion Boundary Segmentation*.

## A Quick Overview 

| <img align="left" width="480" height="170" src="https://github.com/benjaminbarker0219/LSDiff/blob/main/LSDiff%20framework.png"> |
| :----------------------------------------------------------: |
|                          **LSDiff**                          |

## Example Cases
For training, run: ``python scripts/segmentation_train.py --data_name dataset name --data_dir input data direction --out_dir output data direction --image_size --num_channels --class_cond --num_res_blocks --num_heads --learn_sigma --use_scale_shift_norm --attention_resolutions --diffusion_steps --noise_schedule --rescale_learned_sigmas --rescale_timesteps --lr --batch_size``

## Suggestions for Hyperparameters and Training
To train a fine model, i.e., LSDiff in the paper, set the model hyperparameters as:
~~~
--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 
~~~
diffusion hyperparameters as:
~~~
--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False


