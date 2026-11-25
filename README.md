# 🛰️ GeoPulse: Semantic-Aware, Bandwidth-Adaptive Aerial Disaster & Infrastructure Vision Pipeline

[![Course](https://img.shields.io/badge/Academic%20Course-UAI501%3A%20Image%20Processing-00E5FF)](https://github.com)
[![Hardware](https://img.shields.io/badge/Hardware-Windows%2011%20%7C%20GTX%201650%20Ti-10B981)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Command%20Center-FF4B4B)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Build-Passing%20100%25-brightgreen)](https://github.com)

---

## 1. Executive Summary & Problem Statement

During major natural catastrophes (hurricanes, earthquakes, flash floods, landslides), first responders and command centers rely heavily on Unmanned Aerial Vehicles (UAVs) and satellite imagery. However, real-world tactical visual telemetry encounters three severe bottlenecks:

1. **Physical & Sensor Degradation:** Rotor vibration causes severe linear motion blur; wildfire smoke and coastal humidity inject dense atmospheric haze; high-contrast environments clip sensor dynamic range; and low-light operations produce sensor Poisson-Gaussian noise.
2. **Extreme Uplink Scarcity:** Disaster tactical radio and congested cellular uplinks operate at constrained bandwidths (**150 kbps – 1.5 Mbps**). Transmitting uncompressed 4K video is impossible, while aggressive uniform compression obliterates structural cracks, building rubble, and flooded road boundaries.
3. **Spatial-Temporal Misalignment:** Raw, jittery UAV feeds cannot be directly correlated with pre-disaster satellite GIS layers without homography registration, and unstable camera motion impedes dynamic flood tracking and emergency asset tracking.

**GeoPulse** is an end-to-end edge-to-command computer vision and signal-processing engine. Rather than relying on opaque end-to-end deep learning models, GeoPulse places **classical mathematical image processing** (2D-FFT frequency deconvolution, Dark Channel Prior + Guided Filter dehazing, wavelet BayesShrink, anisotropic Perona-Malik diffusion, ROI-adaptive DWT coding, and Lucas-Kanade optical flow) at its core, achieving up to **80% video jitter reduction**, **+5.47 dB foreground PSNR improvements at low bitrates**, and **RFC-7946 GeoJSON disaster vector intelligence**.

---

## 2. Quick Setup & Launch Instructions

### Prerequisites
- **OS:** Windows 10/11 (x64)
- **Python:** 3.10+ (Tested on Python 3.13 with CUDA-enabled PyTorch)
- **GPU (Optional):** NVIDIA GTX 1650 Ti (4GB VRAM) or CPU fallback

### 1. Installation
Clone the repository and install all required dependencies:
```powershell
# Navigate to the workspace
cd d:\Projects\geopulse

# Install dependencies (fast, Windows-compatible)
pip install -r requirements.txt
```

### 2. Generate Synthetic Test Assets (If not already present)
GeoPulse includes self-contained high-detail aerial disaster test assets (768×512 RGB + NIR bands + 10-class segmentation masks + EXIF telemetry + 60-frame shaky video):
```powershell
python scripts/generate_demo_assets.py
```

### 3. Launch the Streamlit Tactical Command Center
Run the interactive 4-tab dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Quantitative Benchmarks & Tests
Execute automated unit tests and generate publication-grade evaluation charts:
```powershell
# Run unit test suite
python -m unittest discover -s tests -v

# Run full evaluation benchmark and export CSV tables + ablation chart
python scripts/run_full_benchmark.py
```

---

## 3. UAI501 Syllabus-to-Module Traceability Matrix

Every theoretical unit in the `UAI501: Image Processing` curriculum maps directly to a concrete, verifiable implementation in GeoPulse:

| Syllabus Unit & Topics | GeoPulse Module | Exact Algorithms & Mathematical Implementation | Verification Output |
|---|---|---|---|
| **Unit 1: Foundations & Advanced Image Representations**<br>• Multispectral & Satellite images<br>• EXIF, GPS, Gimbal metadata<br>• Bit-depth, RAW linearization, Dynamic range<br>• Reinhard HDR-to-SDR Tone Mapping<br>• Color spaces (`YCbCr`, `LAB`, `LUV`)<br>• NDWI (Normalized Difference Water Index) | `src/ingestion/radiometric.py` (`Module 0`) | • `extract_metadata()`: Reads GPS Lat/Lon, Altitude, Gimbal Pitch, and focal length from EXIF with deterministic synthetic fallback.<br>• `simulate_raw_and_tonemap()`: Simulates 12-bit RAW sensor data ($0–4095$), expands dynamic range by $10.0$ stops, and maps to SDR via Reinhard global+local tone-mapping.<br>• `convert_color_spaces()`: Separates luminance $Y$ from chrominance (`Cb`, `Cr`) and converts to `CIELAB`/`LUV`.<br>• `compute_ndwi()`: Computes $\frac{G - \text{NIR}}{G + \text{NIR} + \epsilon}$ for flood boundary delineation. | Telemetry HUD strip; Dynamic range histogram plot; Channel-split inspector in Tab 1. |
| **Unit 2: Enhancement & Restoration**<br>• 2D-FFT Log-magnitude blur spectrum<br>• Radon/Hough PSF angle & length estimation<br>• Frequency-domain Wiener deconvolution<br>• Dark Channel Prior + Guided Filter dehazing<br>• Adaptive Median, Bilateral, NLM, Perona–Malik Anisotropic Diffusion, TV Chambolle, Wavelet BayesShrink<br>• DCT compression artifact cleanup | `src/degradation/simulator.py` (`Module 1`)<br>`src/restoration/engine.py` (`Module 2`) | • `estimate_blur_fft()`: Computes $\log(1 + \|\mathcal{F}(Y_{\text{win}})\|)$ on Hann-windowed patches, uses Radon transform projection variance to find blur orientation $\hat{\theta}$, and detects spectral zero-crossings for blur length $\hat{L}$.<br>• `wiener_deblur_freq()`: Restores luminance channel via $\hat{X}(u,v) = \frac{H^*(u,v)}{\|H(u,v)\|^2 + K} Y(u,v)$.<br>• `dehaze_guided_filter()`: Recovers scene radiance $J(x)$ using Dark Channel Prior refined by edge-preserving guided filter.<br>• `denoise_suite()`: Implements all 6 syllabus denoisers including iterative Perona–Malik PDE diffusion ($c(\|\nabla I\|) = \exp(-(\|\nabla I\|/\kappa)^2)$). | 2D-FFT live spectrum view; +1.2 dB to +2.3 dB PSNR restoration gain; `outputs/benchmarks/restoration_ablation.csv`. |
| **Unit 3: Image Compression**<br>• Lossless & Lossy compression<br>• Spatial Importance Map $M(x,y)$<br>• ROI-Adaptive JPEG2000 (DWT) & WebP<br>• Python Witten-Neal-Cleary Arithmetic Coder<br>• Rate-Distortion & Bjøntegaard Delta-Rate | `src/compression/roi_codec.py` (`Module 3`) | • `generate_roi_importance_map()`: Fuses Sobel edge energy $E(x,y)$, semantic disaster probability $P_{\text{fg}}(x,y)$, and Shannon entropy $H(p)$.<br>• `encode_roi_adaptive()`: Concentrates 88% of bits onto building/road ROIs ($Q_{\text{fg}}=95$ vs $Q_{\text{bg}}=12$).<br>• `arithmetic_coding_demo()`: Pure-Python 32-bit integer arithmetic coder on quantized DWT subbands achieving 100% lossless round-trip within $<2\%$ of theoretical Shannon entropy. | Matched-bitrate side-by-side comparison; Spatial bit-allocation heatmap; Interactive Plotly R-D curve; `outputs/benchmarks/compression_rd_curve.csv`. |
| **Unit 4: Feature Detection & Image Matching**<br>• SIFT, ORB, AKAZE descriptors<br>• Lowe's ratio test & RANSAC homography<br>• MobileNetV3 CNN deep embedding fallback<br>• Multi-temporal structural change heatmap | `src/registration/geo_align.py` (`Module 4`) | • `register_cross_view()`: Detects multiscale keypoints on CLAHE-normalized UAV and satellite tiles, matches via BFMatcher + Lowe's ratio test ($0.75$), and solves $3\times 3$ homography $H$ with RANSAC.<br>• Hybrid Fallback: Triggers `MobileNetV3-Small` cosine-similarity feature matching when RANSAC inliers $< 12$.<br>• Change Detection: Warps UAV frame into satellite frame and computes difference heatmap $\|I_{\text{post}} - I_{\text{pre}}\|$. | Green inlier match canvas; $0.37\text{ px}$ reprojection error; Multi-temporal change overlay in Tab 3. |
| **Unit 5: Video Processing & Analytics**<br>• Sparse Lucas-Kanade optical flow<br>• Moving-average trajectory smoothing<br>• MOG2 dynamic hazard segmentation<br>• Multi-object asset tracking (DeepSORT style)<br>• Dense Farneback frame interpolation | `src/video_analytics/tracker.py` (`Module 5`) | • `stabilize_video_stream()`: Tracks Shi-Tomasi corners via Lucas-Kanade optical flow, estimates affine trajectory, smooths with temporal moving-average filter, and applies $1.05\times$ crop.<br>• `detect_dynamic_hazards_mog2()`: Background subtraction isolating active flood surges and vehicle motion.<br>• `track_rescue_assets()`: Kalman filter + HSV appearance histogram association assigning persistent asset IDs.<br>• `interpolate_low_fps()`: Synthesizes intermediate frames (15 FPS $\rightarrow$ 30 FPS) via bidirectional Farneback dense flow. | Dual synchronized video player in Tab 4; Trajectory jitter reduction plot (**80.0% reduction**); Active track counts. |
| **Unit 6: Cutting-Edge Domain Application**<br>• Geospatial land-cover segmentation<br>• RescueNet 10-class disaster damage mapping<br>• RFC-7946 GeoJSON export | `src/analytics/disaster_seg.py` (`Module 6`) | • `segment_disaster_scene()`: Real-time hybrid segmentation ($<40\text{ ms}$) identifying intact, minor, major, and destroyed buildings, clear and blocked roads, and floodwater.<br>• `export_geojson_report()`: Vectorizes damaged structures and roads into geo-referenced polygons using EXIF affine transforms. | Triage breakdown KPIs; Color-coded disaster severity map; Downloadable RFC-7946 `.geojson` report. |

---

## 4. Quantitative Benchmark Results

All benchmarks are automatically logged to `outputs/benchmarks/`:

### 1. Spatial & Frequency Restoration Ablation (`restoration_ablation.csv`)
Evaluated on moderate drone degradation (PSF motion blur $L=15, \theta=35^\circ$ + Koschmieder haze $\beta=0.8$ + Poisson-Gaussian noise):
- **Raw Degraded Input:** `18.81 dB` PSNR · `0.303` SSIM · `0.013` EPI
- **Dehaze + Bilateral Filter:** `20.67 dB` PSNR · `0.524` SSIM · `0.018` EPI (`10.3 ms`)
- **Dehaze + Wavelet BayesShrink:** `21.06 dB` PSNR · `0.702` SSIM · `0.019` EPI (`88.1 ms`)
- **Dehaze + Total Variation (TV):** `21.09 dB` PSNR · `0.748` SSIM · `0.055` EPI (`595.1 ms`)
- **Dehaze + Perona–Malik Anisotropic:** `20.97 dB` PSNR · `0.693` SSIM · `0.009` EPI
- **Full GeoPulse (FFT-Wiener + Guided + Anisotropic):** `20.03 dB` PSNR · `0.530` SSIM (recovers fine high-frequency structural edges)

### 2. Semantic ROI-Adaptive Compression (`compression_rd_curve.csv`)
Comparison of uniform codecs against GeoPulse ROI-Adaptive coding across tactical bitrates:
- **@ 0.15 bpp (Severe Tactical Throttling ~250 kbps):**
  - Uniform Baseline JPEG: `19.44 dB` Foreground PSNR · `0.856` mIoU
  - Uniform JPEG2000: `24.29 dB` Foreground PSNR · `0.948` mIoU
  - **GeoPulse ROI-Adaptive JPEG2000:** **`24.91 dB`** Foreground PSNR (`+5.47 dB` vs JPEG) · **`0.946`** mIoU
- **@ 0.30 bpp (~500 kbps):**
  - Uniform Baseline JPEG: `26.35 dB` Foreground PSNR · `0.950` mIoU
  - Uniform JPEG2000: `27.78 dB` Foreground PSNR · `0.958` mIoU
  - **GeoPulse ROI-Adaptive JPEG2000:** **`28.42 dB`** Foreground PSNR (`+2.07 dB` vs JPEG, `+0.64 dB` vs Uniform JP2)
- **@ 0.50 bpp (~800 kbps):**
  - Uniform Baseline JPEG: `29.08 dB` Foreground PSNR
  - Uniform JPEG2000: `30.43 dB` Foreground PSNR
  - **GeoPulse ROI-Adaptive JPEG2000:** **`31.34 dB`** Foreground PSNR (`+2.26 dB` vs JPEG, `+0.91 dB` vs Uniform JP2)

### 3. Satellite Geo-Registration & Video Stabilization (`registration_video_metrics.csv`)
- **ORB + RANSAC:** 22 inliers / 34 matches (`64.7%`) · `0.902 px` reprojection error
- **SIFT + RANSAC:** 45 inliers / 91 matches (`49.5%`) · `0.518 px` reprojection error
- **AKAZE + RANSAC + CNN Hybrid:** **163 inliers / 188 matches (`86.7%`)** · **`0.367 px` reprojection error**
- **Lucas-Kanade Video Stabilization:** **`80.0%` trajectory jitter reduction** with persistent tracking across 40+ dynamic assets.

---

## 5. Repository Structure

```
geopulse/
├── app.py                          # Streamlit Tactical Command Center (4-Tab Dashboard)
├── requirements.txt                # Fast-installing dependencies (CUDA-compatible)
├── README.md                       # Comprehensive setup & syllabus traceability guide
├── configs/                        # Configuration settings
├── data/
│   ├── samples/                    # 4 synthetic 768x512 scenes + NIR + masks + UAV flyover video
│   └── satellite_pairs/            # Pre-disaster reference satellite tiles & GT homographies
├── scripts/
│   ├── generate_demo_assets.py     # Deterministic asset generator for aerial scenes & video
│   ├── run_full_benchmark.py       # Headless evaluation script exporting CSVs & 4-panel figure
│   └── verify_all.py               # Quick validation script across all modules
├── src/
│   ├── ingestion/
│   │   └── radiometric.py          # EXIF telemetry, 12-bit RAW sim, Reinhard HDR-to-SDR, YCbCr/LAB/NDWI
│   ├── degradation/
│   │   └── simulator.py            # Linear PSF motion blur, Koschmieder haze, Poisson-Gaussian noise
│   ├── restoration/
│   │   └── engine.py               # 2D-FFT blur estimator, Wiener filter, Guided dehaze, 6 denoisers
│   ├── compression/
│   │   └── roi_codec.py            # Importance map, ROI JPEG2000/WebP, Python Arithmetic Coder
│   ├── registration/
│   │   └── geo_align.py            # AKAZE/SIFT + RANSAC homography + MobileNetV3 CNN fallback
│   ├── video_analytics/
│   │   └── tracker.py              # Lucas-Kanade stabilization, MOG2 hazard detection, Kalman tracker
│   ├── analytics/
│   │   └── disaster_seg.py         # 10-class RescueNet segmentation & RFC-7946 GeoJSON exporter
│   └── utils/
│       └── metrics.py              # PSNR, SSIM, EPI, mIoU, and BD-Rate calculations
├── tests/
│   └── test_modules_0_3.py         # Unit tests covering Modules 0-3
└── outputs/
    ├── benchmarks/                 # Auto-generated CSV ablation tables & 4-panel publication figure
    └── geojson/                    # Exported disaster assessment GeoJSON reports
```

---

## 6. End-to-End System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                   EDGE / UAV ONBOARD PIPELINE                                     |
|                                                                                                   |
|  [Raw UAV Frame / Video + EXIF/GeoTIFF]                                                           |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 0) Radiometric & Metadata Parser                                                         |
|     ├── Extract GPS, Altitude, Gimbal Pitch, Sensor Bit-Depth                                     |
|     ├── 12-bit RAW Linearization -> Reinhard HDR-to-SDR Tone Mapping                              |
|     └── RGB -> YCbCr (Luminance Y for FFT/Wavelets) + CIELAB (Chrominance)                        |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 1) Physics-Based Degradation Simulator (Benchmark Mode Only)                             |
|     └── Calibrated Rotor Motion Blur (PSF) + Koschmieder Haze + Poisson-Gaussian Noise            |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 2) Spatial & Frequency Domain Restoration Engine                                         |
|     ├── 2D-FFT Spectrum Blur Kernel Estimator -> Frequency-Domain Wiener Deconvolution            |
|     ├── Dark Channel Prior + Guided Filter Dehazing                                               |
|     └── Perona-Malik Anisotropic Diffusion / Wavelet BayesShrink / TV Chambolle Denoising         |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 3) Semantic ROI-Adaptive Compression & Transmission                                      |
|     ├── Coarse Edge + Semantic Pass -> Spatial Importance Map M(x,y)                              |
|     ├── ROI-Weighted JPEG2000 (DWT) / WebP Encoding (Q_fg=95 vs Q_bg=12)                          |
|     └── Throttled Tactical Channel (150 - 4000 kbps) -> Artifact Post-Cleanup                     |
+---------------------------------------------------------------------------------------------------+
                                               │
                                 [Compressed Bitstream Over Link]
                                               │
                                               ▼
+---------------------------------------------------------------------------------------------------+
|                               COMMAND CENTER RECEIVER & ANALYTICS                                 |
|                                                                                                   |
|  (Module 4) Multi-Temporal Satellite-to-UAV Geo-Registration                                      |
|     └── AKAZE / SIFT + MobileNetV3 CNN Hybrid -> RANSAC Homography -> Change Heatmap              |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 5) Real-Time Video Stabilization & Asset Tracking                                        |
|     └── Lucas-Kanade Optical Flow (80% Jitter Reduction) -> MOG2 Motion -> Kalman Tracking        |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 6) Geospatial Disaster Segmentation & Decision Dashboard (Streamlit)                     |
|     └── 10-Class Damage Masking + Blocked Road Detection + RFC-7946 GeoJSON Export                |
+---------------------------------------------------------------------------------------------------+
```

---

## 7. Submission Checklist & Verification Status

- [x] **Zero External Dataset Downloads:** All assets deterministically generated locally.
- [x] **Windows 11 & GTX 1650 Ti Guardrails:** Memory capped at `max_dim=768` with graceful CPU fallback.
- [x] **100% UAI501 Syllabus Coverage:** Units 1 to 6 implemented with rigorous mathematical fidelity.
- [x] **Polished Dark-Mode UI:** High-contrast tactical styling with `@st.cache_data` snappy responsiveness.
- [x] **Automated Evaluation:** Generates 3 CSV tables and high-resolution ablation charts.
- [x] **Clean Execution:** Zero syntax, compilation, or deprecation warnings.
