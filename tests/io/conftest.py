"""Shared fixtures for xradar IO tests."""

from pathlib import Path

import pooch
import pytest

# ---------------------------------------------------------------------------
# Argentine BUFR test data — hosted on swnesbitt/open-radar-data
# ---------------------------------------------------------------------------

_BUFR_BASE_URL = "https://github.com/swnesbitt/open-radar-data/raw/main/data/"

_BUFR_REGISTRY = {
    # vol1: RMA1 scan_02 — 3 sweeps, 12 moments, 120 m bins (RELAMPAGO 00:04Z)
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_CM_20181111T000457Z.BUFR.gz":    "062c57cf69e148697a33998d4f50271a2ee5223cbd280e56422d1e40faa36064",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_DBZH_20181111T000457Z.BUFR.gz":  "2e68b91be2e75af74eb8b6a59984238f968f9810c14c50ecdf721cc46f5c31cc",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_DBZV_20181111T000457Z.BUFR.gz":  "be257842bb4fe519c34d7d8a6274deb9f50a9268f91f37534cebc6001f704da8",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_KDP_20181111T000457Z.BUFR.gz":   "ddda9a3baab788b8569e117fcc4d995cfd3a22e8af61933c200b3461854e28e8",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_PHIDP_20181111T000457Z.BUFR.gz": "02209957bd12ab45c59fbc76d68f3f0a96786d6163fef1e643d4db7d659bc220",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_RHOHV_20181111T000457Z.BUFR.gz": "d6ee23f13956da2aecde06d1fe805ab3af2c1d09451700a9f08597d2a3bcb7bd",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_TDR_20181111T000457Z.BUFR.gz":   "3c997f0a0c4ff4bc5f120cf826f5700c1352569dff16e86682c89b3d9b8ade02",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_TH_20181111T000457Z.BUFR.gz":    "b79cbc7cde78335ad3ec385d6428d295cec147962f42d36e246d95a740ee8c10",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_TV_20181111T000457Z.BUFR.gz":    "a8d1b53ad6b28c706e178cabd4bf0ed76aa925fe8318265f8ec5ce4680a0ebb0",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_VRAD_20181111T000457Z.BUFR.gz":  "a7cf2f8e34dc1c9709da9db9de9ac34ba70397886147958f7073fa31bbc1918d",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_WRAD_20181111T000457Z.BUFR.gz":  "6a61f921c5c2b68071a4a9af738386cfada1ace87cb98a790bffd2f4100b0a9a",
    "ARBUFR/bufr_relampago/00/0457/RMA1_0301_02_ZDR_20181111T000457Z.BUFR.gz":   "86ab6ad97ce570930a230bb214bb14a9dc9f14ee927869eba1c2e8ea6fe6ada6",
    # vol2: RMA1 scan_01 — 15 sweeps, 12 moments, 450 m bins (RELAMPAGO 00:06Z)
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_CM_20181111T000620Z.BUFR.gz":    "1e1c1f306ec76978085f49f0394742fff4caca1ea7290dee72a0eee5216486f7",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_DBZH_20181111T000620Z.BUFR.gz":  "41846a02d1c796440c994abcf3bfde6b43dfe59bc3604b7735303fd380d8316e",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_DBZV_20181111T000620Z.BUFR.gz":  "fcccdce102416c54bb47e743e40cac3d5ebb2ee911db32d79595af6747e7787a",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_KDP_20181111T000620Z.BUFR.gz":   "b830d9951cdf1bba7178b0e3fc4c41c561a0f639497edce62c467f5aa6d6a24a",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_PHIDP_20181111T000620Z.BUFR.gz": "c873570ffa91a558ba61eb3d72f780265e569b34325bdbc485014af4f5b78aea",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_RHOHV_20181111T000620Z.BUFR.gz": "bacbd2ccf9ec78bc10fcef2dbfc39d080cb34bced5802bb7c9ec23366fcfd526",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_TDR_20181111T000620Z.BUFR.gz":   "75f5d4865962864f2be147fbaaa7a752b2b5afa3dd4dae3818456b71afc74cc3",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_TH_20181111T000620Z.BUFR.gz":    "08e8506d1f507055dd7c15e1b037aa313be59a58f8271e6e68b4c2c3cf64a418",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_TV_20181111T000620Z.BUFR.gz":    "ebf787e866e72dbdab28a4a85416ef2132d06d2f9f4102cc94e1109b839b81fc",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_VRAD_20181111T000620Z.BUFR.gz":  "de6c2015cdd20c35da4f761f099d7aad1d4f05db6413af435d8dd5f395fb55eb",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_WRAD_20181111T000620Z.BUFR.gz":  "3b57cc4b4b2c1afba5dbafa1754ccfe1951104a6cd514302eb747fdb00ea133e",
    "ARBUFR/bufr_relampago/00/0620/RMA1_0301_01_ZDR_20181111T000620Z.BUFR.gz":   "4a612a71d0ebd89f75efe5a56e31a67eeb93c06fe56b24a6040ac48e00b98c75",
}

_BUFR_STORE = pooch.create(
    path=pooch.os_cache("xradar-argentina-bufr"),
    base_url=_BUFR_BASE_URL,
    registry=_BUFR_REGISTRY,
)


def _fetch_bufr_volume(vol_name: str) -> Path:
    """Download all BUFR files for one volume and return their local directory."""
    prefix = f"ARBUFR/bufr_relampago/00/{vol_name}/"
    keys = [k for k in _BUFR_REGISTRY if k.startswith(prefix)]
    paths = [Path(_BUFR_STORE.fetch(k)) for k in sorted(keys)]
    return paths[0].parent


@pytest.fixture(scope="session")
def bufr_vol1_dir() -> Path:
    """Local directory for vol1: scan_02, 3 sweeps, 120 m bins."""
    return _fetch_bufr_volume("0457")


@pytest.fixture(scope="session")
def bufr_vol2_dir() -> Path:
    """Local directory for vol2: scan_01, 15 sweeps, 450 m bins."""
    return _fetch_bufr_volume("0620")


@pytest.fixture(scope="session")
def bufr_dbzh_file(bufr_vol1_dir) -> Path:
    return next(bufr_vol1_dir.glob("*_DBZH_*.BUFR.gz"))


@pytest.fixture(scope="session")
def bufr_vradh_file(bufr_vol1_dir) -> Path:
    return next(bufr_vol1_dir.glob("*_VRAD_*.BUFR.gz"))
