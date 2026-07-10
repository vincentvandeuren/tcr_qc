"""Layout registry: key/name consistency and the derived tree."""
from tcr_io.structure import Layout, LAYOUT, Kind, render_tree


def test_keys_match_attribute_names():
    for name, art in LAYOUT.items():
        assert art.key == name
        assert getattr(Layout, name) is art


def test_render_tree_covers_every_artifact():
    tree = render_tree()
    # every declared path's leaf name appears in the derived tree -> can't silently drift
    for art in LAYOUT.values():
        leaf = art.path.rstrip("/").split("/")[-1]
        assert leaf in tree, f"{art.key} ({art.path}) missing from render_tree()"


def test_render_tree_marks_optional_and_generated():
    tree = render_tree()
    assert "manifest.json  # optional" in tree
    assert "operations/  # generated" in tree


def test_render_tree_shows_parameterised_repertoire_example():
    assert "sample_001.parquet" in render_tree(examples=True)
    assert "sample_001.parquet" not in render_tree(examples=False)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok:", _name)
