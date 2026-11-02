from os import path
from functools import partial
from .cdf import CDF
from .base import DeepFakeDataModule


class SFFF(CDF):
    """SynthForensics dataset for finetuning and training from scratch — same structure as CDF (REAL/FAKE/csv_files)
    but supports train/val/test splits."""
    # Dataset folder name, set by SFFFDataModule: the metadata cache is named after it
    # (.cache/SFFF-<dataset>-<REAL|FAKE>.pkl), so each dataset (sfff_finetune, sfff_scratch) gets its own
    dataset_name = ''

    @classmethod
    def get_cache_dir(cls, *args):
        return super().get_cache_dir(cls.dataset_name, *args)


class SFFFDataModule(DeepFakeDataModule):
    def prepare_data(self):
        SFFF.dataset_name = path.basename(path.normpath(self.data_dir))
        SFFF.prepare_data(self.data_dir, self.vid_ext)

    def setup(self, stage: str):
        SFFF.dataset_name = path.basename(path.normpath(self.data_dir))
        data_cls = partial(
            SFFF,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            ratio=self.ratio,
            pack=self.pack,
        )

        if stage == "fit":
            self._train_dataset = data_cls(split="train", max_clips=self.max_clips)
            self._val_dataset   = data_cls(split="val",   max_clips=1)
        elif stage in ("valid", "validate"):
            self._val_dataset   = data_cls(split="val",   max_clips=1)
        elif stage == "test":
            self._test_dataset  = data_cls(split="test",  max_clips=1)
