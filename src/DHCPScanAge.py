import csv
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset


class DHCPScanAgeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        data_root="../dHCP",
        target="scan_age",
        hemisphere="left",
        surface="wm",
        features=("pos", "norm", "x"),
        split_files=("scan_age_train.txt", "scan_age_val.txt", "scan_age_test.txt"),
    ):
        self.data_root = Path(os.environ.get("DHCP_DATA_ROOT", data_root)).expanduser().resolve()
        self.target = target
        self.hemisphere = hemisphere
        self.surface = surface
        self.features = tuple(features)
        self.mesh_transform = T.Compose([
            T.NormalizeScale(),
            T.GenerateMeshNormals(),
            T.FaceToEdge(),
        ])

        valid_features = {"x", "pos", "norm"}
        if not self.features or not set(self.features) <= valid_features:
            raise ValueError(f"features must be a non-empty subset of {sorted(valid_features)}")
        if self.target != "scan_age":
            raise ValueError("DHCPScanAgeDataset only supports the 'scan_age' target")

        self.ids = self._read_ids(split_files)
        self.id_set = set(self.ids)
        self.labels = self._read_labels()
        missing_labels = [scan_id for scan_id in self.ids if scan_id not in self.labels]
        if missing_labels:
            raise ValueError(f"Missing {self.target!r} labels for scans: {missing_labels[:5]}")

        missing_assets = []
        for scan_id in self.ids:
            surface_path, feature_path = self._asset_paths(scan_id)
            if not surface_path.is_file() or ("x" in self.features and not feature_path.is_file()):
                missing_assets.append(scan_id)
        if missing_assets:
            raise FileNotFoundError(f"Missing surface or feature assets for scans: {missing_assets[:5]}")

        source_files = [self.data_root / "combined.tsv"]
        source_files.extend(self.data_root / split_file for split_file in split_files)
        for scan_id in self.ids:
            surface_path, feature_path = self._asset_paths(scan_id)
            source_files.append(surface_path)
            if "x" in self.features:
                source_files.append(feature_path)
        source_signature = [
            (str(path.relative_to(self.data_root)), path.stat().st_mtime_ns, path.stat().st_size)
            for path in source_files
        ]
        cache_identity = {
            "cache_version": 2,
            "data_root": str(self.data_root),
            "source_signature": source_signature,
            "target": self.target,
            "hemisphere": self.hemisphere,
            "surface": self.surface,
            "features": self.features,
        }
        cache_key = hashlib.sha1(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:12]
        self.processed_filename = f"scan_age_{cache_key}.pt"

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_filename]

    def download(self):
        pass

    def process(self):
        data_list = [self._load_graph(scan_id) for scan_id in self.ids]
        self.save(data_list, self.processed_paths[0])

    def _read_ids(self, split_files):
        scan_ids = []
        seen = set()
        for split_file in split_files:
            path = self.data_root / split_file
            if not path.is_file():
                raise FileNotFoundError(f"Scan ID list not found: {path}")
            with path.open() as file:
                for line in file:
                    scan_id = line.strip()
                    if scan_id and scan_id not in seen:
                        seen.add(scan_id)
                        scan_ids.append(scan_id)
        if not scan_ids:
            raise ValueError("No scan IDs were found in the configured split files")
        return scan_ids

    @staticmethod
    def _format_session_id(value):
        value = value.strip()
        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
        except ValueError:
            pass
        return value

    def _read_labels(self):
        metadata_path = self.data_root / "combined.tsv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata table not found: {metadata_path}")

        labels = {}
        with metadata_path.open(newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            required_columns = {"participant_id", "session_id", self.target}
            if not reader.fieldnames or not required_columns <= set(reader.fieldnames):
                raise ValueError(f"{metadata_path} must contain columns {sorted(required_columns)}")
            for row in reader:
                participant_id = row["participant_id"].strip()
                session_id = self._format_session_id(row["session_id"])
                scan_id = f"sub-{participant_id}_ses-{session_id}"
                if scan_id not in self.id_set:
                    continue
                if scan_id in labels:
                    raise ValueError(f"Duplicate metadata rows for scan {scan_id}")
                try:
                    label = float(row[self.target])
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid {self.target!r} label for scan {scan_id}") from None
                if not np.isfinite(label):
                    raise ValueError(f"Non-finite {self.target!r} label for scan {scan_id}")
                labels[scan_id] = label
        return labels

    def _asset_paths(self, scan_id):
        surface_path = self.data_root / "surfaces" / f"{scan_id}_{self.hemisphere}.{self.surface}.surf.gii"
        feature_path = self.data_root / "features" / f"{scan_id}_{self.hemisphere}.shape.gii"
        return surface_path, feature_path

    def _load_graph(self, scan_id):
        surface_path, feature_path = self._asset_paths(scan_id)
        pos, face = nib.load(str(surface_path)).agg_data()
        data = Data(
            pos=torch.as_tensor(np.asarray(pos), dtype=torch.float32),
            face=torch.as_tensor(np.asarray(face).T.copy(), dtype=torch.long),
            y=torch.tensor(self.labels[scan_id], dtype=torch.float32),
            scan_id=scan_id,
        )
        data = self.mesh_transform(data)

        feature_parts = []
        if "x" in self.features:
            raw_features = nib.load(str(feature_path)).agg_data()
            if isinstance(raw_features, np.ndarray):
                node_features = raw_features
            else:
                node_features = np.stack(raw_features, axis=1)
            if node_features.ndim == 1:
                node_features = node_features[:, None]
            feature_parts.append(torch.as_tensor(node_features, dtype=torch.float32))
        if "pos" in self.features:
            feature_parts.append(data.pos)
        if "norm" in self.features:
            feature_parts.append(data.norm)

        if not all(part.shape[0] == data.num_nodes for part in feature_parts):
            raise ValueError(f"Node feature count does not match mesh vertices for scan {scan_id}")
        data.x = torch.cat(feature_parts, dim=1)
        if not torch.isfinite(data.x).all() or not torch.isfinite(data.y):
            raise ValueError(f"Non-finite graph features or target for scan {scan_id}")
        if data.edge_index.numel() and (
            data.edge_index.min() < 0 or data.edge_index.max() >= data.num_nodes
        ):
            raise ValueError(f"Invalid edge index for scan {scan_id}")
        return data
