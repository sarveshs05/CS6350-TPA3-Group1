### **README.md**

---

# **TPA-3: YOLOv8++ for overlap object detection from cluttered indoor shots, invariant to sensor, lighting and affine transformation**

## **Overview**
This project implements a modified version of YOLOv8 (termed YOLOv8++) for detecting overlapping objects in cluttered indoor environments. The model is invariant to sensor types, lighting conditions, and affine transformations. 

The project introduces an additional head to the YOLOv8 architecture for improved detection. It uses a custom dataset augmented from a base dataset obtained via Roboflow.

## **Authors**
- **Debanjan Guha** (CE22B050)
- **Sarvesh Shanbhag** (CE22B103)

---

## **Data and Resources**
### **Directory Structure**
- **YOLOV8++ Drive Link**: [Drive Folder](https://drive.google.com/drive/folders/1z78FACvcam31CxOse2aGdolgLaJXjsnV?usp=sharing)
  - `stationaryaug.zip` (Augmented Dataset)
  - `ultralytics.zip` (Modified YOLOv8 Source Code)
  - `100epochsbest.pt` (Trained Model)
  - few test images and their output.
    
### **Custom Dataset**
- Augmented dataset in COCO format: `stationaryaug.zip`
- Original dataset source: [Roboflow Stationary Items Dataset](https://universe.roboflow.com/national-university-fast/stationary-items-dataset/dataset/8).
  - This is the base dataset which was later augmented. (stationaryaug.zip).

### **Ultralytics Source Code**
- Ultralytics github was cloned and necessary changes were made in its architecture to add an extra head. All the edits have been added to `ultralytics.zip`
- The model has been trained on custom dataset for 100 epochs using Kaggle GPU P100 (no need to train again as it might consume a lot of time) and the best.pt has been uploaded in the drive `100epochsbest.pt`.

major changes in ultralytics.zip -
1. the yaml file was edited and an extra head was added - ultralytics/ultralytics/cfg/models/v8/yolov8x-2xhead.yaml
2. ConcatHead Layer was added in ultralytics/ultralytics/nn/modules/conv.py

```
class ConcatHead(nn.Module):
    """Concatenation layer for Detect heads."""

    def __init__(self, nc1=80, nc2=1, ch=()):
        """Initializes the ConcatHead."""
        super().__init__()
        self.nc1 = nc1  # number of classes of head 1
        self.nc2 = nc2  # number of classes of head 2

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        # x is a list of length 2
        # Each element is either a tuple or just the decoded features
        # depending whether it's being exported.
        # First element of tuple are the decoded preds,
        # second element are feature maps for heatmap visualization

        if isinstance(x[0], tuple):
            preds1 = x[0][0]
            preds2 = x[1][0]
        elif isinstance(x[0], list): # when returned raw outputs
            # The shape is used for stride creation in tasks.py.
            # Feature maps will have to be decoded individually if used as they can't be merged.
            return [torch.cat((x0, x1), dim=1) for x0, x1 in zip(x[0], x[1])]
        else:
            preds1 = x[0]
            preds2 = x[1]

        # Concatenate the new head outputs as extra outputs

        # 1. Concatenate bbox outputs
        # Shape changes from [N, 4, 6300] to [N, 4, 12600]
        preds = torch.cat((preds1[:, :4, :], preds2[:, :4, :]), dim=2)

        # 2. Concatenate class outputs
        # Append preds 1 with empty outputs of size 6300
        shape = list(preds1.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]

        preds1_extended = torch.zeros(shape, device=preds1.device,
                                      dtype=preds1.dtype)
        preds1_extended[..., : preds1.shape[-1]] = preds1

        # Prepend preds 2 with empty outputs of size 6300
        shape = list(preds2.shape)
        shape[-1] = preds1.shape[-1] + preds2.shape[-1]

        preds2_extended = torch.zeros(shape, device=preds2.device,
                                      dtype=preds2.dtype)
        preds2_extended[..., preds2.shape[-1] :] = preds2

        # Arrange the class probabilities in order preds1, preds2. The
        # class indices of preds2 will therefore start after preds1
        preds = torch.cat((preds, preds1_extended[:, 4:, :]), dim=1)
        preds = torch.cat((preds, preds2_extended[:, 4:, :]), dim=1)

        if isinstance(x[0], tuple):
            return (preds, x[0][1])
        else:
            return preds
```
---


## **Usage**
1. **Unzipping Files**
   - Unzip the provided datasets (`stationaryaug.zip` and `ultralytics.zip`) and code:
   ```bash
   !unzip /content/drive/MyDrive/YOLOV8++/ultralytics.zip -d /content/
   !unzip /content/drive/MyDrive/YOLOV8++/stationaryaug.zip -d /content/
   ```

3. **Load Pretrained Model**
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8x.pt")  # Load a pretrained YOLOv8 model
   ```

4. **Finetune the Model** (Not necessary as the pre-trained model is provided.)
   - use the yaml file of stationary data
   ```python
   results = model.train(data='/content/stationary/data.yaml', freeze=22, epochs=100, imgsz=640)
   ```

6. **Load Finetuned Model**
   - use `100epochsbest.pt`
   ```python
   model = YOLO("/content/drive/MyDrive/YOLOV8++/100epochsbest.pt")
   ```

8. **Add Additional Head**
   - use yolo8x-2xhead.yaml file (updated yaml file)
   ```python
   state_dict = torch.load("yolov8xfin.pth")
   model_2 = YOLO('ultralytics/cfg/models/v8/yolov8x-2xhead.yaml', task="detect").load('yolov8x.pt')
   model_2.load_state_dict(state_dict, strict=False)
   ```

10. **Run Predictions**
   ```python
   result_merged = model_2.predict(test_image)[0]
   ```

11. **Visualize Results**
   ```python
   show_output([result_coco.plot(), result_custom.plot(), result_merged.plot()])
   ```

---

## **Custom Modifications**
### **Key Architectural Changes**
- **Additional Head**
  Added a `ConcatHead` layer in `ultralytics/ultralytics/nn/modules/conv.py`.
  ```python
  class ConcatHead(nn.Module):
      ...
  ```

- **Updated YAML Configuration**
  Modified the YAML file `ultralytics/ultralytics/cfg/models/v8/yolov8x-2xhead.yaml` to integrate the new head.

---

## **Results**
- The modified model successfully detects overlapping objects in cluttered environments, as demonstrated in the test images.
- Outputs include predictions from the base COCO model, the finetuned stationary model, and the merged YOLOv8++ model.

---


## **Troubleshooting**
- If issues arise, please do let us know.
- Ensure all dependencies are installed using the provided requirements file.

---

