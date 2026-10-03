from cardisim.deepcardiosim import DEEPCARDIOSIM_REFERENCE, deepcardiosim_reference


def test_deepcardiosim_reference_is_pinned():
    reference = deepcardiosim_reference()
    assert reference.repository.endswith("ehsanngh/DeepCardioSim")
    assert reference.repository_ref.startswith("main@")
    assert len(reference.repository_ref.split("@", 1)[1]) == 40
    assert reference.license == "MIT"
    assert reference.paper_doi == "10.1038/s41746-026-02399-7"
    assert reference.dataset_url.endswith("17651628")
    assert reference == DEEPCARDIOSIM_REFERENCE


def test_reference_serialization_is_plain_data():
    payload = DEEPCARDIOSIM_REFERENCE.to_dict()
    assert set(payload) == {
        "name",
        "repository",
        "repository_ref",
        "license",
        "paper_doi",
        "dataset_url",
        "intended_task",
    }
    assert all(isinstance(value, str) for value in payload.values())
