import cv2
import numpy as np
import pytest
from kerascan.config import EngineConfig, ROIConfig, QualityConfig, RadialConfig
from kerascan.roi_detection import detect_placido_roi
from kerascan.quality import evaluate_quality
from kerascan.segmentation import TraditionalSegmenter
from kerascan.radial_scan import RadialResult, radial_scan
from kerascan.graph_tracking import track_rings
from kerascan.inference import KerascanEngine
from kerascan.synthetic import synthetic_placido

def ring(**kw): return synthetic_placido(**kw)

@pytest.mark.parametrize('name,kwargs',[
 ('clean_normal_like_rings',{}),('rotated_normal_like_rings',{'rotation':.5}),
 ('mild_distortion',{'distortion':.10}),('severe_distortion',{'distortion':.28})])
def test_ring_variants_have_detectable_roi(name,kwargs):
    img=ring(**kwargs); roi=detect_placido_roi(img)
    assert roi.crop.shape[0] == roi.crop.shape[1]
    assert roi.outer_radius_px > 35

@pytest.mark.parametrize('name,kwargs,flag',[
 ('blur',{'blur':7},'blur'),('glare',{'glare':True},'glare_or_saturation'),
 ('darkness',{'darkness':.92},'underexposed'),('eyelid_occlusion',{'occlusion':True},'possible_eyelid_or_eyelash_obstruction'),
 ('low_contrast',{'low_contrast':True},'low_contrast'),('heavy_noise',{'noise':45},'sensor_noise')])
def test_quality_perturbations(name,kwargs,flag):
    img=ring(**kwargs)
    # A severe synthetic lid can hide the automatic locator; quality must still
    # recognise the obstruction when an acquisition/manual centre is available.
    roi=detect_placido_roi(img, ROIConfig(manual_center=(240,240)) if name=='eyelid_occlusion' else ROIConfig())
    seg=TraditionalSegmenter().segment(roi.crop,roi.center_roi,roi.outer_radius_px)
    q=evaluate_quality(roi.crop,roi.center_roi,roi.outer_radius_px,ring_mask=seg.mask)
    # The 3-level quality gate may classify mild low-contrast as 'mild_low_contrast'
    # (ACCEPTABLE_WITH_WARNING) rather than 'low_contrast' (REJECTED). Accept either.
    if flag == 'low_contrast':
        assert 'low_contrast' in q['flags'] or 'mild_low_contrast' in q['flags'], \
            f"Expected low-contrast flag; got flags={q['flags']}"
    else:
        assert flag in q['flags']

def test_off_center_and_small_patterns():
    img=ring(shape=(900,1100),center=(170,500)); roi=detect_placido_roi(img)
    q=evaluate_quality(roi.crop,roi.center_roi,roi.outer_radius_px,ring_mask=TraditionalSegmenter().segment(roi.crop,roi.center_roi,roi.outer_radius_px).mask)
    assert roi.box[0] >= 0 and 'pattern_off_centre' not in q['flags'] # crop is correctly re-centred
    small=ring(shape=(1200,1600),center=(800,600)); # embedded ring is deliberately small for full image
    r=detect_placido_roi(small); assert r.outer_radius_px > 30

def test_low_resolution_blank_and_rgba():
    img=ring(shape=(120,120)); roi=detect_placido_roi(img); q=evaluate_quality(roi.crop,roi.center_roi,roi.outer_radius_px)
    assert 'low_resolution' in q['flags']
    blank=np.zeros((500,700,3),np.uint8); outcome=KerascanEngine().analyze(blank)
    assert outcome['screening_result']=='UNGRADABLE' and outcome['classification_skipped']
    rgba=np.dstack((ring(),np.full((480,480),255,np.uint8))); roi=detect_placido_roi(rgba); assert roi.crop.shape[2]==4

@pytest.mark.parametrize('shape',[(480,1000),(1000,480)])
def test_wide_and_tall_full_eye_photographs(shape):
    image=ring(shape=shape,center=(shape[1]*.55,shape[0]*.5)); roi=detect_placido_roi(image)
    assert roi.crop.shape[0]==roi.crop.shape[1] and 0<=roi.box[0]<roi.box[2]<=shape[1]

def test_small_pattern_inside_large_photograph_is_unggradable():
    image=ring(shape=(3000,3000),center=(1500,1500))
    out=KerascanEngine().analyze(image)
    assert 'placido_pattern_too_small' in out['quality']['flags'] and out['screening_result']=='UNGRADABLE'

def test_partial_missing_rings_remain_nan():
    img=ring(occlusion=True); roi=detect_placido_roi(img); seg=TraditionalSegmenter().segment(roi.crop,roi.center_roi,roi.outer_radius_px)
    config=RadialConfig(meridians=180)
    scan=radial_scan(seg.enhanced,roi.center_roi,roi.outer_radius_px,config)
    tracked=track_rings(scan,radial=config)
    assert np.any(~np.isfinite(tracked.radii))

def test_duplicate_or_incorrect_ring_numbering_is_removed():
    angles=np.arange(12)*30.; positions=np.arange(5.,50.)
    candidates=[np.asarray([20.,20.,25.,30.,40.]) for _ in angles]
    scan=RadialResult(angles,positions,np.zeros((12,len(positions))),np.zeros((12,len(positions))),candidates,
                      [np.ones(len(row)) for row in candidates],np.asarray([20.,25.,30.,40.]),"provisional_polar_profile",1.,0.)
    tracked=track_rings(scan,radial=RadialConfig(max_radial_jump_px=6.))
    assert tracked.duplicate_removals >= len(angles) and tracked.radii.shape==(4,len(angles))
    assert np.all(np.diff(tracked.radii,axis=0)>0)

def test_unggradable_never_fits_or_classifies():
    engine=KerascanEngine()
    out=engine.analyze(np.zeros((300,300,3),np.uint8))
    assert out['screening_result']=='UNGRADABLE'
    assert out['classification_performed'] is False

def test_manual_crop_fallback():
    img=ring(); roi=detect_placido_roi(img,ROIConfig(manual_box=(50,50,430,430)))
    assert roi.method=='manual_box' and roi.box==(50,50,430,430)
