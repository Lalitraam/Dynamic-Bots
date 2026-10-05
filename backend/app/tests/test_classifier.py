import pytest
from app.services.classifier import parse_time_control, meets_min_time_control, classify_time_control
from app import config

def test_parse_time_control():
    assert parse_time_control("180+2") == (180, 2)
    assert parse_time_control("600") == (600, 0)
    assert parse_time_control("60+0") == (60, 0)
    assert parse_time_control("1+0") == (1, 0)   # literal 1 second base
    assert parse_time_control("3+2") == (3, 2)   # literal 3 seconds base
    assert parse_time_control("-") is None
    assert parse_time_control("") is None
    assert parse_time_control("?") is None
    assert parse_time_control("abc") is None
    assert parse_time_control("10+") is None
    assert parse_time_control("+5") is None
    assert parse_time_control("10+abc") is None

def test_meets_min_time_control():
    original_min = config.MIN_BASE_SECONDS
    config.MIN_BASE_SECONDS = 60

    try:
        assert meets_min_time_control("60+0") == True   # exactly at minimum
        assert meets_min_time_control("60+1") == True   # 60s base
        assert meets_min_time_control("120+1") == True  # 120s base
        assert meets_min_time_control("30+0") == False  # 30s base - below minimum
        assert meets_min_time_control("15+0") == False  # 15s base - below minimum
        assert meets_min_time_control("30+1") == False  # 30s base - below minimum
        assert meets_min_time_control("0+1") == False   # 0s base - below minimum
        assert meets_min_time_control("180+0") == True  # 180s base
        assert meets_min_time_control("180+2") == True  # 180s base
        assert meets_min_time_control("300+3") == True  # 300s base
        assert meets_min_time_control("480+0") == True  # 480s base
        assert meets_min_time_control("600+0") == True  # 600s base
        assert meets_min_time_control("900+10") == True # 900s base
        assert meets_min_time_control("1500+0") == True # 1500s base
        assert meets_min_time_control("-") == False
        assert meets_min_time_control("") == False
        assert meets_min_time_control("abc") == False
    finally:
        config.MIN_BASE_SECONDS = original_min

def test_classify_time_control():
    original_min = config.MIN_BASE_SECONDS
    original_cats = config.INCLUDED_CATEGORIES
    config.MIN_BASE_SECONDS = 60
    config.INCLUDED_CATEGORIES = {"bullet", "blitz", "rapid"}

    try:
        # Test accepted cases (should return the category, not "other")
        assert classify_time_control("60+0") == "bullet"   # 60s base -> bullet
        assert classify_time_control("60+1") == "bullet"   # 60s base -> bullet
        assert classify_time_control("120+1") == "bullet"  # 120s base -> bullet
        assert classify_time_control("180+0") == "blitz"   # 180s base -> blitz
        assert classify_time_control("180+2") == "blitz"   # 180s base -> blitz
        assert classify_time_control("300+3") == "blitz"   # 300s base -> blitz
        assert classify_time_control("480+0") == "rapid"   # 480s base -> rapid
        assert classify_time_control("600+0") == "rapid"   # 600s base -> rapid
        assert classify_time_control("900+10") == "rapid"  # 900s base -> rapid

        # Test rejected cases (should return "other")
        assert classify_time_control("1500+0") == "other"  # classical (excluded by default)
        assert classify_time_control("15+0") == "other"    # ultrabullet (excluded by default)
        assert classify_time_control("-") == "other"
        assert classify_time_control("") == "other"
        assert classify_time_control("abc") == "other"

        # classify_time_control categorizes without base floor logic
        assert classify_time_control("30+1") == "bullet"   # estimated 70s -> bullet
        assert classify_time_control("0+1") == "bullet"    # estimated 40s -> bullet

        # Test with different INCLUDED_CATEGORIES
        config.INCLUDED_CATEGORIES = {"classical"}
        assert classify_time_control("1500+0") == "classical"
        assert classify_time_control("60+0") == "other"   # bullet now excluded

        config.INCLUDED_CATEGORIES = {"ultrabullet"}
        assert classify_time_control("15+0") == "ultrabullet" # estimated 15 < 30 -> ultrabullet
        assert classify_time_control("30+0") == "other"       # estimated 30 is bullet (excluded)
        assert classify_time_control("60+0") == "other"       # bullet now excluded
    finally:
        config.MIN_BASE_SECONDS = original_min
        config.INCLUDED_CATEGORIES = original_cats

def test_config_model_tier():
    assert config.model_tier(100) == "rejected"
    assert config.model_tier(149) == "rejected"
    assert config.model_tier(150) == "small"
    assert config.model_tier(399) == "small"
    assert config.model_tier(400) == "standard"
    assert config.model_tier(799) == "standard"
    assert config.model_tier(800) == "full"
    assert config.model_tier(1200) == "full"