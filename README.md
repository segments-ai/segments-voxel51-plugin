# Segments.ai Voxel51 Plugin

A WIP plugin for Voxel51, integrating it with the annotation platform [segments.ai](https://segments.ai)

# Setup
## Plugin installation

**TODO**

## Configuration

To use this plugin, you need to provide your segments.ai API key. This is done by setting it as an environment variable before launching the app:
```bash
export SEGMENTS_API_KEY=<YOUR API KEY HERE>
fiftyone launch app
```

# Operators
You can bring up the operators by pressing "`" and searching for their name. Below you can find the available operators.

## Request annotations

With this operator you can, from the fiftyone app, create a new segments.ai annotation dataset. You can either upload the whole fiftyone dataset, upload only the current view, or you can upload all of the selected samples.

Current limitations:
 - Fiftyone datasets with grouped media, and 3D scenes are not yet supported. If you want to upload 3D pointclouds, please use [point cloud datasets](https://docs.voxel51.com/user_guide/using_datasets.html#point-cloud-datasets)

## Fetch annotations

You can fetch annotations from a segments.ai dataset using this operator. When you call this operator, you can select one of your datasets and one of its releases. It will then pull the annotations and display them within the fiftyone app. 

When pulling the labels from segments.ai, it will match them with the fiftyone samples by comparing their filenames. In the case of 3D scenes, the pcd filename is not stored in the sample metadata. When using 3D scenes, you should store the filename in the sample attributes under the key "segments_pc_filename". For example:

```python
import fiftyone as fo
dataset = fo.load_dataset("your_dataset")
for sample in dataset:
    sample["segments_pc_filename"] = "<FILENAME-OF-THIS-SAMPLES-POINTCLOUD-HERE>"
    sample.save()
```

Current limitations:
 - It's currently not possible to pull annotations for segments.ai sequences.

## Add issue

Segments.ai issues are useful mechanisms for communicating problems in the labelling with you annotation team. With this operator, you can file an issue in a specific sample from within the fiftyone app. Simply select 1 sample and run this operator. You will be able to select your segments.ai dataset and write the description of the problem.