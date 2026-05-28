# Benchmark dataset

This folder is intended to hold the **495-image gold-standard annotation CSV** used to benchmark HNI-VLM.

## Status

- ⏳ Annotations CSV (`annotations.csv`) — to be added.
- ⏳ Image URLs / paths — to be added.

## File format (planned)

`annotations.csv` will follow this schema:

| column | description |
|---|---|
| `post_id`                   | Unique Flickr post identifier |
| `url`                       | Public image URL |
| `primary_category_H`        | Human consensus label: setting / subject / partner |
| `social_context_H`          | solo / group / NA |
| `Activity_Intensity_H`      | sedentary / moderate / vigorous / NA |
| `photographer_engagement_H` | documented / implied |

The `_H` suffix denotes human (gold-standard) labels. Model predictions go in matching `_A` columns when running the evaluator.

## Annotation protocol

- Two independent annotators applied the taxonomy defined in `docs/framework.md`.
- Inter-annotator agreement (Cohen's κ): primary=0.853, engagement=0.860, social=0.684, intensity=0.569.
- Disagreements were reconciled through joint review against the written protocol.

## Licensing & redistribution

Flickr images themselves are subject to each photographer's license. This repository only stores URLs and consensus labels, not the image bytes. Researchers should re-download images using the URLs and respect the original license of each photo.
