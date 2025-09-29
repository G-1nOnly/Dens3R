from enum import Enum

from hloc import extract_features, match_features

MatchingMode = Enum("MatchingMode", ["sequential", "exhaustive", "vocab_tree", "pairs_from_poses", "database"])

SfMMode = Enum("SfMMode", ["sfm", "triangulate"])

Extractor = Enum("Extractor", [*list(extract_features.confs.keys()), "loftr"])

Matcher = Enum("Matcher", [*list(match_features.confs.keys()), "loftr"])
