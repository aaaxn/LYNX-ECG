import h5py
import numpy as np
import pandas as pd
import pytest
from pack_codetest import pack

pytestmark = pytest.mark.unit


def test_pack_keeps_each_label_under_its_own_name(tmp_path):
    release = tmp_path / "release"
    (release / "annotations").mkdir(parents=True)
    with h5py.File(release / "ecg_tracings.hdf5", "w") as h5:
        h5.create_dataset("tracings", data=np.zeros((3, 4096, 12)))
    gold = pd.DataFrame(
        {
            "1dAVb": [1, 0, 0],
            "RBBB": [0, 1, 0],
            "LBBB": [0, 0, 0],
            "SB": [0, 0, 0],
            "AF": [1, 1, 0],
            "ST": [0, 0, 1],
        }
    )
    gold.to_csv(release / "annotations" / "gold_standard.csv", index=False)
    pd.DataFrame({"age": [50, 60, 70], "sex": ["M", "F", "M"]}).to_csv(
        release / "attributes.csv", index=False
    )

    pack(release, tmp_path / "codetest_raw.hdf5")

    with h5py.File(tmp_path / "codetest_raw.hdf5", "r") as h5:
        names = [str(name) for name in h5.attrs["label_names"]]
        labels = h5["labels"][()]
        assert labels[:, names.index("AF")].tolist() == [1, 1, 0]
        assert labels[:, names.index("ST")].tolist() == [0, 0, 1]
        assert h5["exam_id"][()].tolist() == [0, 1, 2]
        assert h5["is_male"][()].tolist() == [1, 0, 1]
