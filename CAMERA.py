import ast
import os
from typing import Optional

import datasets as ds
import pandas as pd

_CITATION = """\
@inproceedings{mita-et-al:nlp2023,
    author =    "三田 雅人 and 村上 聡一朗 and 張 培楠",
    title =	    "広告文生成タスクの規定とベンチマーク構築",
    booktitle = "言語処理学会 第29回年次大会",
    year =      2023,
}
"""

_DESCRIPTION = """\
CAMERA (CyberAgent Multimodal Evaluation for Ad Text GeneRAtion) is the Japanese ad text generation dataset.
"""

_HOMEPAGE = "https://github.com/CyberAgentAILab/camera"

_LICENSE = """\
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
"""

_URLS = {
    "without-lp-images": "https://storage.googleapis.com/camera-public/camera-v1-minimal.tar.gz",
    "with-lp-images": "https://storage.googleapis.com/camera-public/camera-v1.tar.gz",
}


class CameraDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="without-lp-images",
            version=VERSION,
            description="The CAMERA dataset w/o LP images (ver.1.0.0 | 126.2 MiB)",
        ),
        ds.BuilderConfig(
            name="with-lp-images",
            version=VERSION,
            description="The CAMERA dataset w/ LP images (ver.1.0.0 | 61.5 GiB)",
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "asset_id": ds.Value("int64"),
                "kw": ds.Value("string"),
                "lp_meta_description": ds.Value("string"),
                "title_org": ds.Value("string"),
                "title_ne1": ds.Value("string"),
                "title_ne2": ds.Value("string"),
                "title_ne3": ds.Value("string"),
                "domain": ds.Value("string"),
                "parsed_full_text_annotation": ds.Sequence(
                    {
                        "text": ds.Value("string"),
                        "xmax": ds.Value("int64"),
                        "xmin": ds.Value("int64"),
                        "ymax": ds.Value("int64"),
                        "ymin": ds.Value("int64"),
                    }
                ),
            }
        )

        if self.config.name == "with-lp-images":
            features["lp_image"] = ds.Image()

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        base_dir = dl_manager.download_and_extract(_URLS[self.config.name])
        lp_image_dir: Optional[str] = None

        if self.config.name == "without-lp-images":
            camera_dir_name = f"camera-v{self.VERSION.major}-minimal"
        elif self.config.name == "with-lp-images":
            camera_dir_name = f"camera-v{self.VERSION.major}"
            lp_image_dir = os.path.join(base_dir, camera_dir_name, "lp-screenshot")
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

        tng_path = os.path.join(base_dir, camera_dir_name, "train.csv")
        dev_path = os.path.join(base_dir, camera_dir_name, "dev.csv")
        tst_path = os.path.join(base_dir, camera_dir_name, "test.csv")

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"file_path": tng_path, "lp_image_dir": lp_image_dir},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"file_path": dev_path, "lp_image_dir": lp_image_dir},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={"file_path": tst_path, "lp_image_dir": lp_image_dir},
            ),
        ]

    def _generate_examples(self, file_path: str, lp_image_dir: Optional[str] = None):
        df = pd.read_csv(file_path)
        for i in range(len(df)):
            data_dict = df.iloc[i].to_dict()

            asset_id = data_dict["asset_id"]
            keywords = data_dict["kw"]
            lp_meta_description = data_dict["lp_meta_description"]
            domain = data_dict.get("domain", "")
            text_anns = ast.literal_eval(data_dict["parsed_full_text_annotation"])

            title_org = data_dict["title_org"]
            title_ne1 = data_dict.get("title_ne1", "")
            title_ne2 = data_dict.get("title_ne2", "")
            title_ne3 = data_dict.get("title_ne3", "")

            example_dict = {
                "asset_id": asset_id,
                "kw": keywords,
                "lp_meta_description": lp_meta_description,
                "title_org": title_org,
                "title_ne1": title_ne1,
                "title_ne2": title_ne2,
                "title_ne3": title_ne3,
                "domain": domain,
                "parsed_full_text_annotation": text_anns,
            }

            if self.config.name == "with-lp-images" and lp_image_dir is not None:
                lp_image_file_name = f"screen-1200-{asset_id}.png"
                example_dict["lp_image"] = os.path.join(
                    lp_image_dir, lp_image_file_name
                )

            yield i, example_dict
