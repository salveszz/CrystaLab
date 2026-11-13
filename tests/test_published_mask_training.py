# FILE PURPOSE: Test source-label fidelity, whole-image refusal and task separation
# using generated temporary source pixels, never fabricated human review records.
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_published_masks as source_fixture
from models.segmentation import training_options
from vision.published_mask_training import prepare, audit_training
from vision.training_data import audit_segmentation
from vision.video import file_sha256
from evaluate_chimp_pilot import native_ious, source_matches


@pytest.fixture
def generated_source(tmp_path):
    case = source_fixture.PublishedMaskTests()
    case.setUp()
    try:
        yield case, tmp_path
    finally:
        case.tearDown()


def setup_pilot(case, root):
    case.import_index()
    protocol = dict(
        schema="crystalab_chimp_selected_mask_pilot_v1",
        task="published_selected_protein_instances",
        classes=["crystal", "drop"],
        pitch_class_mapping=None,
        source_manifest_sha256=file_sha256(case.output / "manifest.json"),
        source_index_sha256=file_sha256(case.output / "images.jsonl"),
        selection=dict(
            datasets=["a", "b"],
            train_per_dataset=3,
            validation_per_dataset=1,
            hash_prefix="fixture|",
        ),
        polygon_export=dict(minimum_native_iou=0.95),
        class_semantics="Generated author-selected-task fixture; not generic KDP negatives",
    )
    path = root / "pilot.json"
    path.write_text(json.dumps(protocol))
    return path, root / "pilot"


def test_original_holes_survive_as_truth_while_polygon_loss_is_recorded(
    generated_source,
):
    case, root = generated_source
    protocol, output = setup_pilot(case, root)
    result = prepare(case.output, protocol, output)
    assert result["eligible_by_split"] == dict(train=6, validation=2)
    records = json.loads((output / "records.json").read_text())
    hole = next(
        o
        for r in records
        for o in r["objects"]
        if o["conversion"]["filled_hole_pixels"]
    )
    from PIL import Image
    import numpy as np

    mask = np.asarray(Image.open(output / hole["mask_path"]))
    assert mask[6, 8] == 0
    assert hole["conversion"]["polygon_mask_iou"] >= 0.95
    assert result["actual_human_reviews_created"] == 0
    with pytest.raises(ValueError):
        audit_segmentation(output)


def test_disconnected_mask_excludes_entire_image_without_replacement(generated_source):
    case, root = generated_source
    import io
    import numpy as np

    payload = case.entries["a/masks/1.npz"]
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        masks, classes = archive["masks"], archive["class_labels"]
    masks[1, 20, 30] = True
    case.entries["a/masks/1.npz"] = source_fixture.npz_bytes(classes, masks)
    case.repack()
    protocol, output = setup_pilot(case, root)
    result = prepare(case.output, protocol, output)
    assert result["rejected_images"] == 1 and result["eligible_by_split"]["train"] == 5
    rejected = next(
        r
        for r in json.loads((output / "records.json").read_text())
        if not r["eligible"]
    )
    assert len(rejected["objects"]) == 2
    assert rejected["training_image_path"] is None and rejected["label_path"] is None
    assert (output / rejected["objects"][1]["mask_path"]).exists()


def test_rehashed_split_edit_still_fails_source_membership(generated_source):
    case, root = generated_source
    protocol, output = setup_pilot(case, root)
    prepare(case.output, protocol, output)
    records = json.loads((output / "records.json").read_text())
    records[0]["split"] = "validation"
    (output / "records.json").write_text(json.dumps(records))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["records.json"] = file_sha256(output / "records.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership"):
        audit_training(output)


def test_source_classes_cannot_be_renamed_as_seed_or_inclusion(generated_source):
    case, root = generated_source
    protocol, output = setup_pilot(case, root)
    content = json.loads(protocol.read_text())
    content["pitch_class_mapping"] = dict(crystal="seed", drop="inclusion_candidate")
    protocol.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="distinct"):
        prepare(case.output, protocol, output)


def test_unindexed_image_cannot_enter_the_training_directory(generated_source):
    case, root = generated_source
    protocol, output = setup_pilot(case, root)
    prepare(case.output, protocol, output)
    (output / "images/train/unindexed.jpg").write_bytes(case.entries["a/images/0.jpg"])
    with pytest.raises(ValueError, match="Unindexed"):
        audit_training(output)


def test_dense_mask_detection_limit_is_explicit_and_not_boolean():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/chimp_mask_pilot_train.json").read_text())
    assert training_options(config)["max_det"] == 512
    config["max_det"] = True
    with pytest.raises(ValueError, match="Maximum detections"):
        training_options(config)


def test_packed_native_iou_keeps_holes_tail_bits_and_class_boundaries():
    import numpy as np

    target = np.zeros((5, 7), np.uint8)
    target[1:4, 1:5] = 255
    target[2, 2] = 0
    prediction = target.copy()
    prediction[2, 2] = 255
    truth = [dict(class_id=0, mask=target)]
    predicted = [dict(class_id=1, mask=prediction), dict(class_id=0, mask=prediction)]
    np.testing.assert_allclose(native_ious(truth, predicted), [[11 / 12, 11 / 12]])
    matches = source_matches(truth, predicted, 0.5)
    assert len(matches) == 1 and matches[0]["prediction_index"] == 1
    assert matches[0]["iou"] == pytest.approx(11 / 12)


def test_no_predictions_preserves_missed_source_targets():
    import numpy as np

    truth = [dict(class_id=0, mask=np.ones((5, 7), np.uint8))]
    assert native_ious(truth, []).shape == (1, 0)
    assert source_matches(truth, [], 0.5) == []
