<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.


-->

# SuperGlue

## Overview

The SuperGlue model was proposed
in [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763) by Paul-Edouard Sarlin, Daniel
DeTone, Tomasz Malisiewicz and Andrew Rabinovich.

This model consists of matching two sets of interest points detected in an image. Paired with the 
[SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and 
estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following:

*This paper introduces SuperGlue, a neural network that matches two sets of local features by jointly finding correspondences 
and rejecting non-matchable points. Assignments are estimated by solving a differentiable optimal transport problem, whose costs 
are predicted by a graph neural network. We introduce a flexible context aggregation mechanism based on attention, enabling 
SuperGlue to reason about the underlying 3D scene and feature assignments jointly. Compared to traditional, hand-designed heuristics, 
our technique learns priors over geometric transformations and regularities of the 3D world through end-to-end training from image 
pairs. SuperGlue outperforms other learned approaches and achieves state-of-the-art results on the task of pose estimation in 
challenging real-world indoor and outdoor environments. The proposed method performs matching in real-time on a modern GPU and 
can be readily integrated into modern SfM or SLAM systems. The code and trained weights are publicly available at this [URL](https://github.com/magicleap/SuperGluePretrainedNetwork).*

## How to use

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched. 
The outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding 
matching scores. Due to the nature of SuperGlue, to output a dynamic number of matches, you will need to use the mask attribute to 
retrieve the respective information:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image1 = "https://github.com/cvg/LightGlue/blob/main/assets/sacre_coeur1.jpg?raw=true"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://github.com/cvg/LightGlue/blob/main/assets/sacre_coeur2.jpg?raw=true"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the respective image masks 
image0_mask, image1_mask = outputs.mask[0] > 0

image0_indices = torch.nonzero(image0_mask).squeeze()
image1_indices = torch.nonzero(image1_mask).squeeze()

image0_matches = outputs.matches[0, 0][image0_indices]
image1_matches = outputs.matches[0, 1][image1_indices]

image0_matching_scores = outputs.matching_scores[0, 0][image0_indices]
image1_matching_scores = outputs.matching_scores[0, 1][image1_indices]
```

You can then print the matched keypoints on a side-by-side image to visualize the result :

```python
import numpy as np
import matplotlib.pyplot as plt

# Create side by side image
input_data = inputs["pixel_values"]
height, width = input_data.shape[-2:]
matched_image = np.zeros((height, width * 2, 3))
matched_image[:, :width] = input_data.squeeze()[0].permute(1, 2, 0).cpu().numpy()
matched_image[:, width:] = input_data.squeeze()[1].permute(1, 2, 0).cpu().numpy()
matched_image = (matched_image * 255).astype(np.uint8)

# Get the respective image keypoints
image0_keypoints = outputs.keypoints[0, 0][image0_mask]
image1_keypoints = outputs.keypoints[0, 1][image0_matches[image0_matches > -1]]

# Draw matches
plt.imshow(matched_image)
plt.axis('off')

# Draw matches based on score threshold
for keypoint0, keypoint1, score in zip(image0_keypoints, image1_keypoints, image0_matching_scores):
    if score > 0.1:
        keypoint0_x, keypoint0_y = int(keypoint0[0]), int(keypoint0[1])
        keypoint1_x, keypoint1_y = int(keypoint1[0] + width), int(keypoint1[1])
        color = [score.item()] * 3  # Set color based on score
        plt.plot([keypoint0_x, keypoint1_x], [keypoint0_y, keypoint1_y], color=color, linewidth=1)

# Save the image
plt.savefig("matched_image.png")
plt.close()
```

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperGluePretrainedNetwork).

## SuperGlueConfig

[[autodoc]] SuperGlueConfig

## SuperGlueImageProcessor

[[autodoc]] SuperGlueImageProcessor

- preprocess

## SuperGlueForKeypointMatching

[[autodoc]] SuperGlueForKeypointMatching

- forward
