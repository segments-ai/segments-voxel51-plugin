<p align="center">
    <br>
        <img src="assets/logo_no_shadow-with_text-blue_background.png" width="400"/>
    <br>
    🚀 <b>New: experimental support for multisensor sequences</b> 🚀
<p>

# Segments.ai Voxel51 Plugin

A plugin for [Voxel51](https://voxel51.com/), integrating it with the annotation platform [Segments.ai](https://segments.ai)
## Demo video

https://github.com/user-attachments/assets/6580e39d-8a4f-4d2f-bf4e-f434110dcd65

# Setup
## Plugin installation

```bash
fiftyone plugins download https://github.com/segments-ai/segments-voxel51-plugin
fiftyone plugins requirements @segmentsai/segmentsai --install
```

## Configuration

To use this plugin, you need to provide your Segments.ai API key. This is done by setting it as an environment variable before launching the app:
```bash
export SEGMENTS_API_KEY=<YOUR API KEY HERE>
fiftyone app launch
```

# Dataset type compatibility

The following table shows the compatible [fityone mediatypes](https://docs.voxel51.com/user_guide/using_datasets.html#using-media-type) with the [Segments.ai label types](https://docs.segments.ai/reference/sample-and-label-types).

| Fiftyone media type | Compatible Segments.ai datasets                                                              |
| --------------------| ---------------------------------------------------------------------------------------------|
| image               | segmentation-bitmap, vector, bbox, keypoints                                                 |
| pointcloud          | pointcloud-cuboid, pointcloud-vector, pointcloud-cuboid-sequence, pointcloud-vector-sequence |
| 3d[^1]              | pointcloud-cuboid, pointcloud-vector                                                         |
| video               | **Not supported**                                                                            |
| group               | multisensor-sequence, pointcloud-vector-sequence                                             |

[^1]: request_annotations operator not yet supported for 3d mediatype. Please use the pointcloud mediatype if you need this feature.

When using `*-sequence` datasets, this plugin expects you to use the [dynamic grouping](https://voxel51.com/blog/dynamic-groups-fiftyone-tips-and-tricks-sep-8-2023/) feature of fiftyone. This will allow you to group fiftyone samples into sequences.

# Operators
You can bring up the operators by pressing "`" and searching for their name. Below you can find the available operators.

## Select segments dataset

This operator selects the corresponding Segments.ai dataset for the current fiftyone dataset. This is needed for the other operators to interact with existing Segments.ai datasets. This operator will show a dropdown list of all your Segments.ai datasets with the corresponding data type. The selected Segments.ai dataset is stored internally in the fiftyone dataset object.

## Request annotations

With this operator you can, from the fiftyone app, create a new Segments.ai annotation dataset. You can either upload the whole fiftyone dataset, upload only the current view, or you can upload all of the selected samples. This operator will either upload your data to the segments.ai AWS bucket if the samples `filepath` is a local path. If it refers to a cloud storage location (`s3://` or `gs://`), it will only send that reference to Segments.ai. Alternatively you can specify the `segments_filepath` metadata field in the fiftyone sample (see [this section](#sample-metadata)).

Current limitations:
 - Fiftyone datasets with 3D scenes are not yet supported. If you want to upload 3D pointclouds, please use [point cloud datasets](https://docs.voxel51.com/user_guide/using_datasets.html#point-cloud-datasets)
 - You can't create a  `multisensor-sequence` Segments.ai dataset using this operator. Instead, you can create a dataset using the segments.ai web interface and use the plugin to setup your annotation task. 


## Fetch annotations

You can fetch annotations from a Segments.ai dataset using this operator. When you call this operator, you can select one of your datasets and one of its releases. It will then pull the annotations and display them within the fiftyone app. You can read more on how samples are matched to eachother across the two platforms [here](#matching-segmentsai-samples-with-fiftyone-samples).


Current limitations:
 - It's currently not possible to fetch annotations for Segments.ai sequences (except `multisensor-sequences`).
 - For `multisensor-sequences`: you can only fetch annotations for the pointcloud annotation task. Importing image annotations from these datasets is not yet implemented.


## Add issue

Segments.ai issues are useful mechanisms for communicating problems in the labelling with you annotation team. With this operator, you can file an issue in a specific sample from within the fiftyone app. Simply select 1 sample and run this operator. You will be able to describe your issue in the text box, which will then be uploaded to Segments.ai! 

# Matching Segments.ai samples with fiftyone samples

When pulling labels from Segments.ai, the operator will match fiftyone samples with Segments.ai samples in one of two ways. Which mechanism is used is based on if you are using a sequence Segments.ai dataset or not. 

## Single-frame datasets

In case you are using a non-sequence Segments.ai dataset, the matching mechanism simply looks up the corresponding fiftyone sample based on the segments sample UUID.  These are automatically stored in the fiftyone sample object as an attribute, under the key "segments_uuid". If you have uploaded your dataset to Segments.ai using the `request_annotations` operator, this is done for you automatically! If not, you will have to provide this information yourself. This can be done as follows:

```python
import fiftyone as fo
dataset = fo.load_dataset("your_dataset")
for sample in dataset.iter_samples(autosave=True,progress=True):
    # Somehow match the fo.Sample with the segments.ai sample
    sample["segments_uuid"] = "<UUID OF YOUR SAMPLE HERE>"
```

Alternatively, this example uses FiftyOne's [set_values()][] method to perform a bulk update:

[set_values()]: https://docs.voxel51.com/api/fiftyone.core.dataset.html?highlight=set_values#fiftyone.core.dataset.Dataset.set_values

```python
import fiftyone as fo
dataset = fo.load_dataset("your_dataset")
dataset.set_values("segments_uuid",your_values_map,key_field=your_key)
```

## Sequence datasets

For sequence datasets, matching a fiftyone sample and Segments.ai sample is a bit more complex, as a Segments.ai sample contains multiple frames and data types, while fiftyone samples always contain 1 frame. Instead of matching only UUID, we instead match samples on a tuple of 3 values: 
 - `segments_uuid` - `str`: the UUID of the sample on Segments.ai
 - `segments_frame_idx` - `int`: the frame number of the data point on Segments.ai (zero indexed)
 - `segments_sensor_name` - `str`: the sensor name of this data point on Segments.ai

These 3 are all stored as fields in the fiftyone sample.

If you have created your sample using the `request_annotations` operator, then the plugin will have populated these fields if fiftyone for you.

# Sample metadata

To optimally use the Segments.ai platform, you can provide information such as calibration parameters and egomotion transformations to your fiftyone samples before creating a sample on Segments.ai. Below you can find an overview of the different metadata fields this plugin interacts with.

Here is an example showing how to specify the intrinsic matrix of an image sample:

```python
import fiftyone as fo
dataset = fo.load_dataset("your_dataset")
for sample in dataset.iter_samples(autosave=True,progress=True):
    # Somehow match the fo.Sample with the segments.ai sample
    sample.metadata.intrinsic_matrix = [[721.5377, 0.0, 609.5593], [0.0, 721.5377, 172.854], [0.0, 0.0, 1.0]]
```

## For any sample

| Field                    | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| Sample.segments_filepath | The cloud storage location where this data sample is stored. If defined, the plugin will send this URL to Segments.ai when using the `request_annotations` operator, instead of using the default `filepath` field.|

## For image samples

| Field                                   | Description                                                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Sample.metadata.intrinsic_matrix        | A nested list with 3x3 elements representing the camera intrinsic matrix.                                            |
| Sample.metadata.extrinsics_position     | Dictionary containing the translation component of the camera extrinsics. Expects keys `x, y, z`                     |
| Sample.metadata.extrinsics_rotation     | Dictionary containing the rotation component of the camera extrinsics as a quaternion. Expects keys `qx, qy, qz, qw` |
| Sample.metadata.distortion_model        | String representing which distortion model the camera uses.                                                          |
| Sample.metadata.distortion_coefficients | The coefficients of the camera model                                                                                 |
| Sample.metadata.camera_convention       | The camera convention used for the calibration. Should be either “OpenGL” or “OpenCV”                                |

# For lidar samples

| Field                    | Description                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Sample.metadata.position | the position of the lidar sensor. Should be a dictionary containing `x, y` and `z`.                       |
| Sample.metadata.heading  | The rotation of the lidar sensor as a quaternion. Should be a dictionary containing `qx, qy, qz` and `qw` |
