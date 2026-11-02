"""
SynthForensics NSG feature dataset for NSG-VD.

Subclasses ScoreFeaturesDataset, overriding only _get_dataset() and
_setup_cache_paths() to work with the SynthForensics directory layout
instead of the GenVideo layout.
"""

import os
from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
from data.synthforensics_video_dataset import SynthForensicsVideoDataset


class SynthForensicsScoreFeaturesDataset(ScoreFeaturesDataset):
    """
    NSG feature dataset for SynthForensics videos.

    Args:
        video_dir:        flat directory containing .mp4 video files
        ids_file:         path to SF-*.txt split file (bare IDs, # = comments)
                          pass None to load all .mp4 files in video_dir
        filename_suffix:  appended to bare ID to form the filename stem
                          e.g. "_cogvideox" → "000_cogvideox.mp4"
                          pass "" for real videos (already named "{id}.mp4")
        frame_cache_dir:  directory where extracted frames are cached
        nsg_cache_dir:    directory where NSG features are cached
        label:            "fake" or "real"
        **kwargs:         forwarded to ScoreFeaturesDataset
                          (num_frames, diffuse_steps, feature_type, device, …)
    """

    def __init__(
        self,
        video_dir: str,
        ids_file: str,
        filename_suffix: str,
        frame_cache_dir: str,
        nsg_cache_dir: str,
        label: str = "fake",
        **kwargs,
    ):
        # Store SynthForensics-specific attrs BEFORE super().__init__,
        # because super().__init__ immediately calls _get_dataset() and
        # _setup_cache_paths() (which are overridden here).
        self._sf_video_dir = video_dir
        self._sf_ids_file = ids_file
        self._sf_filename_suffix = filename_suffix
        self._sf_frame_cache_dir = frame_cache_dir
        self._sf_nsg_cache_dir = nsg_cache_dir

        # Use a dummy generation_model that maps to the correct label via
        # get_label_from_generation_model():  "Pika" → "fake", "MSR-VTT" → "real"
        dummy_gen = "Pika" if label == "fake" else "MSR-VTT"

        super().__init__(
            data_path="/tmp",        # unused: _get_dataset() is overridden
            dataset_name="GenVideo", # unused: _get_dataset() is overridden
            generation_model=dummy_gen,
            ids_file=None,           # handled by SynthForensicsVideoDataset
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _get_dataset(self):
        return SynthForensicsVideoDataset(
            video_dir=self._sf_video_dir,
            frame_cache_dir=self._sf_frame_cache_dir,
            num_frames=self.num_frames,
            label=self.label,
            ids_file=self._sf_ids_file,
            filename_suffix=self._sf_filename_suffix,
            load_len=self.load_len,
        )

    def _setup_cache_paths(self):
        self.cache_path = self._sf_nsg_cache_dir
        self.input_dir = os.path.join(self.cache_path, "input")
        self.output_file = os.path.join(self.cache_path, "output.npy")
        os.makedirs(self.input_dir, exist_ok=True)
