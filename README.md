<p align="center">
    <br>
        <img src="assets/logo_no_shadow-with_text-blue_background.png" width="400"/>
    <br>
<p>

# Segments.ai Voxel51 Plugin

A plugin for Voxel51, integrating it with the annotation platform [Segments.ai](https://segments.ai)
## Demo video

https://github.com/user-attachments/assets/30831626-da06-47b9-b290-5996db8b4597

# Setup
## Plugin installation

```bash
fiftyone plugins download https://github.com/segments-ai/voxel51-plugin
```

## Configuration

To use this plugin, you need to provide your Segments.ai API key. This is done by setting it as an environment variable before launching the app:
```bash
export SEGMENTS_API_KEY=<YOUR API KEY HERE>
fiftyone launch app
```

# Dataset type compatibility

The following table shows the compatible [fityone mediatypes](https://docs.voxel51.com/user_guide/using_datasets.html#using-media-type) with the [Segments.ai label types](https://docs.segments.ai/reference/sample-and-label-types).

| Fiftyone media type | Compatible Segments.ai datasets              |
| --------------------| ---------------------------------------------|
| image               | segmentation-bitmap, vector, bbox, keypoints |
| pointcloud          | pointcloud-cuboid, pointcloud-vector         |
| 3d[^1]              | pointcloud-cuboid, pointcloud-vector         |
| video               | **Not supported**                            |
| group               | **Not supported**                            |

[^1]: request_annotations operator not yet supported for 3d mediatype. Please use the pointcloud mediatype if you need this feature.

# Operators
You can bring up the operators by pressing "`" and searching for their name. Below you can find the available operators.

## Select segments dataset

This operator selects the corresponding Segments.ai dataset for the current fiftyone dataset. This is needed for the other operators to interact with existing Segments.ai datasets. This operator will show a dropdown list of all your Segments.ai datasets with the corresponding data type. The selected Segments.ai dataset is stored internally in the fiftyone dataset object.

## Request annotations

With this operator you can, from the fiftyone app, create a new Segments.ai annotation dataset. You can either upload the whole fiftyone dataset, upload only the current view, or you can upload all of the selected samples.

Current limitations:
 - Fiftyone datasets with grouped media, and 3D scenes are not yet supported. If you want to upload 3D pointclouds, please use [point cloud datasets](https://docs.voxel51.com/user_guide/using_datasets.html#point-cloud-datasets)


## Fetch annotations

You can fetch annotations from a Segments.ai dataset using this operator. When you call this operator, you can select one of your datasets and one of its releases. It will then pull the annotations and display them within the fiftyone app. 

When pulling labels from Segments.ai, the operator will match fiftyone samples with Segments.ai samples using the segments UUID. These are automatically stored in the fiftyone sample object as an attribute, under the key "segments_uuid". If you have uploaded your dataset to Segments.ai using the `request_annotations` operator, this is done for you automatically! If not, you will have to provide this information yourself. This can be done as follows:

```python
import fiftyone as fo
dataset = fo.load_dataset("your_dataset")
for sample in dataset:
    # Somehow match the fo.Sample with the segments.ai sample
    sample["segments_uuid"] = "<UUID OF YOUR SAMPLE HERE>"
    sample.save()
```

Current limitations:
 - It's currently not possible to pull annotations for Segments.ai sequences.

## Add issue

Segments.ai issues are useful mechanisms for communicating problems in the labelling with you annotation team. With this operator, you can file an issue in a specific sample from within the fiftyone app. Simply select 1 sample and run this operator. You will be able to describe your issue in the text box, which will then be uploaded to Segments.ai! 
