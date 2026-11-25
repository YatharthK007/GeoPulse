
# **GeoPulse: Semantic-Aware, Bandwidth-Adaptive Aerial Disaster & Infrastructure Vision Pipeline**

|**Metadata Field**|**Specification**|
|---|---|
|**Product Name**|GeoPulse (Edge-to-Command Aerial Telemetry & Vision Engine)|
|**Academic Alignment**|Course `UAI501: Image Processing` (`L: 3, T: 0, P: 2, Cr: 4.0`) — End-Semester Lab Evaluation|
|**Target Hardware**|Windows 11 (x64) · Intel Core i5-10300H (2.50 GHz) · 16 GB DDR4 RAM · NVIDIA GeForce GTX 1650 Ti (4 GB VRAM)|
|**Document Version**|`v1.0 — Production Engineering & AI-Implementation Blueprint`|
|**Primary Stack**|Python 3.10+, OpenCV (`opencv-contrib-python`), SciPy, PyWavelets, PyTorch (CUDA), FFmpeg, Rasterio, Streamlit, Plotly|

## 1. Executive Summary & Core Vision

### 1.1 Problem Statement

During natural disasters (hurricanes, floods, earthquakes, landslides), Unmanned Aerial Vehicles (UAVs) and Earth-observation satellites serve as first-responder eyes on the ground. However, real-world aerial visual telemetry suffers from three compounding bottlenecks:

  

1. **Severe Physical & Sensor Degradation:** Rotor vibration induces directional motion blur; wildfire smoke and atmospheric moisture create non-uniform haze; extreme sunlight-to-shadow contrast clips dynamic range; and low-light dusk operations inject sensor noise.
    
      
    
2. **Extreme Uplink Bandwidth Scarcity:** Disaster zones frequently rely on degraded cellular or tactical radio links (250 kbps – 1.5 Mbps). Transmitting uncompressed or high-bitrate 4K UAV video is impossible, while applying standard uniform compression (low-quality JPEG or aggressive H.264 quantization) introduces severe blocking artifacts that destroy fine structural cracks, debris boundaries, and flooded road edges.
    
      
    
3. **Lack of Spatial-Temporal Alignment:** Raw shaky drone feeds cannot be directly overlaid onto pre-disaster satellite maps without geometric registration, nor can rescue coordinators track moving emergency assets across unstable frames.
    
      
    

### 1.2 Solution Overview

**GeoPulse** is an end-to-end **degradation-to-decision computer vision and signal-processing pipeline** that transforms degraded raw UAV imagery and video into bandwidth-efficient, radiometrically calibrated, stabilized, and geo-referenced disaster intelligence.

  

Unlike conventional student projects that rely solely on black-box deep learning classifiers or generative wrappers, GeoPulse places **classical mathematical image processing** (radiometric calibration, 2D-FFT frequency deconvolution, guided-filter dehazing, wavelet shrinkage, Region-of-Interest JPEG2000/CABAC rate-distortion optimization, AKAZE+RANSAC homography, and sparse optical flow) at the core of the pipeline, using lightweight neural networks strictly for semantic Region-of-Interest (ROI) guidance and multi-object association.

  

### 1.3 Differentiation & Strategic Value

- **Zero Overlap with Peer Projects:** Classmate topics center on standard medical segmentation/classification (skin cancer, glaucoma, retinal vessels, diabetic thermography), vintage CCD ISP, photometric stereo, underwater enhancement, solar cell defects, and GAN colorization. GeoPulse introduces **multi-modal geospatial telemetry + rate-distortion-aware video compression**.
    
      
    
- **100% Syllabus Coverage:** Every theoretical unit in `UAI501`—from `YCbCr`/`CIELAB` color spaces and `EXIF`/`GeoTIFF` metadata to `Perona–Malik` anisotropic diffusion, `JPEG2000`/`CABAC`, `SIFT`/`AKAZE` + `RANSAC`, `DeepSORT`, and geospatial disaster assessment—is implemented as an indispensable functional stage.
    
      
    
- **Resume & Industry Impact:** Demonstrates production systems engineering (VRAM budgeting, bitrate-vs-mIoU trade-offs, multi-sensor geometric registration) applicable to aerospace/defense, autonomous robotics, climate-tech, and edge-AI roles.
    
      
    

## 2. Target Hardware & Environment Constraints

All algorithms, batch sizes, tensor resolutions, and storage budgets are engineered specifically for the target workstation:

  

|**Component**|**Target Specification**|**Engineering Constraint & Design Decision**|
|---|---|---|
|**CPU**|Intel Core i5-10300H @ 2.50 GHz (4C/8T)|Vectorize all spatial/frequency filters in NumPy/SciPy/OpenCV C++ backends; avoid pure-Python nested pixel loops.|
|**GPU / VRAM**|NVIDIA GeForce GTX 1650 Ti (**4 GB GDDR6**)|Cap neural inference resolution at `640×480` or `768×512` using FP16/ONNX Runtime or lightweight PyTorch models (`YOLOv8n` / `MobileNetV3-UNet`) so peak VRAM stays **< 2.2 GB**.|
|**System RAM**|16.0 GB (15.8 GB usable)|Stream video frames via generators (`cv2.VideoCapture`) rather than loading full multi-minute video tensors into RAM.|
|**Disk Storage**|~91 GB free (386 GB / 477 GB used)|Curate a **~2.5 GB** representative dataset bundle (subset of RescueNet + VisDrone + satellite tiles) instead of downloading 22+ GB raw archives.|
|**OS & Runtime**|Windows 11 (64-bit, x64)|Use `ffmpeg-python` with a local static `ffmpeg.exe` binary and `Pillow`/`OpenCV` JPEG2000 codecs to avoid Windows C-compiler build failures.|

## 3. Complete `UAI501` Syllabus-to-Architecture Traceability Matrix

Every topic listed in the `UAI501` syllabus maps directly to a concrete module, mathematical formulation, and verifiable metric in GeoPulse:

  

|**Syllabus Unit & Topics image_6cdffd.png**|**GeoPulse Module**|**Exact Algorithms & Mathematical Implementation**|**Verification Metric / Visual Proof**|
|---|---|---|---|
|**Unit 1: Foundations & Advanced Image Representations**<br><br>  <br>  <br><br>• Multispectral & Satellite images<br><br>  <br>  <br><br>• EXIF, IPTC/XMP, Geo-Spatial metadata<br><br>  <br>  <br><br>• Bit-depth, RAW sensor data, quantization, dynamic range<br><br>  <br>  <br><br>• Gamma correction & radiometric transforms<br><br>  <br>  <br><br>• Color spaces (`YCbCr`, `LAB`, `LUV`)<br><br>  <br>  <br><br>• HDR fundamentals, Tone Mapping & HDR-to-SDR|`Module 0: Ingestion & Radiometric Calibration`|• Parse GPS lat/lon, altitude, gimbal pitch, and focal length from `EXIF`/`XMP` & `GeoTIFF` headers.<br><br>  <br>  <br><br>• Ingest 4-band RGB+NIR multispectral tiles to compute **NDWI** (Normalized Difference Water Index) for flood masking.<br><br>  <br>  <br><br>• Simulate 12-bit RAW linear sensor data (`0–4095`), apply **Reinhard / Drago HDR-to-SDR tone mapping** and inverse gamma correction.<br><br>  <br>  <br><br>• Convert RGB $\rightarrow$ `YCbCr` (to isolate luminance $Y$ for frequency filtering) and `CIELAB` (for perceptual chrominance preservation).|Metadata Telemetry Card; HDR vs. SDR histogram & dynamic range recovery plot; `Y/Cb/Cr` channel split view.|
|**Unit 2: Enhancement & Restoration (Spatial/Frequency)**<br><br>  <br>  <br><br>• Mean, median, adaptive median filters<br><br>  <br>  <br><br>• Fourier Transform (`DFT`, `FFT`), `Wavelet` transforms<br><br>  <br>  <br><br>• `Wiener` filter (spatial/frequency domain)<br><br>  <br>  <br><br>• `Non-Local Means (NLM)`, `Bilateral`, `Perona–Malik` anisotropic diffusion, `Total Variation (TV)` denoising<br><br>  <br>  <br><br>• `Guided filtering` & rolling guidance filter<br><br>  <br>  <br><br>• Motion blur estimation in frequency spectrum<br><br>  <br>  <br><br>• Wavelet shrinkage, low-light & haze restoration, decompression artifact cleanup|`Module 1 (Simulator)` & `Module 2 (Restoration Engine)`|• **Blur Estimation & Deconvolution:** Compute 2D-FFT log-magnitude spectrum $\log(1 + \vert{}\mathcal{F}(Y)\vert{})$, detect dark spectral zero-crossings via Radon/Hough orientation analysis to estimate PSF angle $\theta$ and length $L$, and restore via **Frequency-Domain Wiener Filter** $H^*(u,v)/(\vert{}H(u,v)\vert{}^2 + K)$.<br><br>  <br>  <br><br>• **Haze Removal:** Dark Channel Prior refined with **Guided Filter** (`cv2.ximgproc.guidedFilter`).<br><br>  <br>  <br><br>• **Edge-Preserving Denoising:** Benchmark **Adaptive Median**, **Bilateral**, **NLM**, **Perona–Malik Anisotropic Diffusion**, **TV Chambolle**, and **Wavelet BayesShrink**.<br><br>  <br>  <br><br>• **Artifact Cleanup:** Rolling guidance / bilateral post-filter to suppress block ringing.|Live **2D-FFT Spectrum** plot showing blur notch lines; Ablation Table comparing **PSNR (dB)**, **SSIM**, and **Edge Preservation Index (EPI)** across all filters.|
|**Unit 3: Image Compression**<br><br>  <br>  <br><br>• Lossless & lossy compression<br><br>  <br>  <br><br>• `Arithmetic Coding`<br><br>  <br>  <br><br>• `JPEG`, `JPEG2000`<br><br>  <br>  <br><br>• `CABAC` / `CAVLC` in `H.264/HEVC`<br><br>  <br>  <br><br>• `WebP` (lossy + lossless), `HEIF/HEIC`, `MPEG-2` (`FFmpeg`)|`Module 3: Semantic ROI-Adaptive Compression Engine`|• Compute an **ROI Importance Mask** $M(x,y) \in [0,1]$ combining structural edge energy + coarse semantic damage probability.<br><br>  <br>  <br><br>• **ROI-Adaptive Coding:** Preserve foreground disaster regions at high quality ($Q=90$) while compressing background water/vegetation ($Q=20$) using **JPEG2000** (DWT-based) and **WebP/HEIF**.<br><br>  <br>  <br><br>• **Codec & Entropy Benchmark:** Transcode via **FFmpeg** comparing **H.264 CAVLC vs. H.264 CABAC vs. HEVC (H.265) vs. MPEG-2**, plus a standalone Python **Arithmetic Coder** benchmark on quantized wavelet subbands.|**Rate-Distortion (BD-Rate) Curves** plotting Bitrate (`bpp` / `kbps`) vs. Downstream Damage Segmentation `mIoU` across codecs.|
|**Unit 4: Feature Detection & Image Matching**<br><br>  <br>  <br><br>• `SIFT`, `SURF`, `ORB`, `AKAZE`<br><br>  <br>  <br><br>• `RANSAC` for robust matching<br><br>  <br>  <br><br>• Deep feature extraction (`CNN embeddings`)|`Module 4: Cross-View Geo-Registration`|• Detect keypoints across oblique post-disaster UAV frames and nadir pre-disaster satellite tiles using **AKAZE** and **SIFT**.<br><br>  <br>  <br><br>• Filter outliers via Lowe's ratio test + **RANSAC** homography estimation (`cv2.findHomography`).<br><br>  <br>  <br><br>• Fallback/fusion with **CNN deep feature embeddings** (`MobileNetV3` intermediate feature map cosine matching) when severe rubble/flooding breaks local gradient descriptors.|Keypoint correspondence match lines (inliers vs. outliers), Homography Reprojection Error (px), and Pre/Post Wipe Overlay.|
|**Unit 5: Video Processing & Real-Time Analytics**<br><br>  <br>  <br><br>• Frame rates, temporal sampling, frame interpolation<br><br>  <br>  <br><br>• Motion estimation & `optical flow`<br><br>  <br>  <br><br>• Background subtraction (`MOG2`, `KNN`)<br><br>  <br>  <br><br>• Object tracking (`KCF`, `SORT`, `DeepSORT`)<br><br>  <br>  <br><br>• Real-time detection, temporal filtering, video stabilization|`Module 5: Stabilized Video Telemetry & Tracking`|• **Video Stabilization:** Track sparse features via **Lucas-Kanade Optical Flow**, estimate affine trajectory, smooth with sliding-window temporal Gaussian filter, and warp frames.<br><br>  <br>  <br><br>• **Dynamic Hazard Segmentation:** Separate surging floodwater/debris flow from static ground via **MOG2 / KNN** background subtraction.<br><br>  <br>  <br><br>• **Asset Tracking:** Run `YOLOv8n` + **DeepSORT** (Kalman filter + appearance embeddings) to track rescue boats, ambulances, and stranded persons.<br><br>  <br>  <br><br>• **Low-FPS Recovery:** Optical-flow-guided **frame interpolation** to reconstruct 30 FPS from 10 FPS throttled uplinks.|Trajectory jitter reduction plot (pre- vs. post-stabilization), live tracking IDs, and FPS / latency metrics.|
|**Unit 6: Cutting-Edge Domain Application**<br><br>  <br>  <br><br>• Geospatial, Land cover classification, Disaster detection & assessment|`Module 6: Disaster Assessment & GeoJSON Export`|• Pixel-level semantic segmentation of **10 RescueNet classes** (Building-No-Damage, Building-Minor, Building-Major, Building-Total-Destruction, Road-Clear, Road-Blocked, Water, Tree, Vehicle, Pool) + NDWI land-cover fusion.<br><br>  <br>  <br><br>• Project pixel damage masks through homography onto GPS coordinates to export an actionable **GeoJSON Damage Report**.|Color-coded Disaster Severity Map, Passable vs. Blocked Road Network Summary, and downloadable `.geojson` & `.csv` reports.|

## 4. End-to-End System Architecture & Data Flow

Plaintext

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
|     └── Injects calibrated Rotor Motion Blur (PSF) + Atmospheric Haze + Poisson-Gaussian Noise    |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 2) Spatial & Frequency Domain Restoration Engine                                         |
|     ├── 2D-FFT Spectrum Blur Kernel Estimator -> Frequency-Domain Wiener Deconvolution            |
|     ├── Dark Channel Prior + Guided Filter Dehazing                                               |
|     └── Perona-Malik Anisotropic Diffusion / Wavelet BayesShrink Denoising                        |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 3) Semantic ROI-Adaptive Compression & Transmission                                      |
|     ├── Coarse Edge + Lightweight Semantic Pass -> Spatial Importance Map M(x,y)                  |
|     ├── ROI-Weighted JPEG2000 (DWT) / WebP / FFmpeg HEVC-CABAC Encoding                           |
|     └── Simulated Bandwidth Channel (e.g., 300 kbps - 2 Mbps) -> Decompression Artifact Cleanup   |
+---------------------------------------------------------------------------------------------------+
                                               │
                                 [Compressed Bitstream Over Link]
                                               │
                                               ▼
+---------------------------------------------------------------------------------------------------+
|                               COMMAND CENTER RECEIVER & ANALYTICS                                 |
|                                                                                                   |
|  (Module 4) Multi-Temporal Satellite-to-UAV Geo-Registration                                      |
|     └── AKAZE / SIFT + CNN Embedding Fusion -> RANSAC Homography -> Pre/Post Change Overlay       |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 5) Real-Time Video Stabilization & Asset Tracking                                        |
|     └── Lucas-Kanade Optical Flow Stabilization -> MOG2 Flood Motion -> YOLOv8n + DeepSORT        |
|          │                                                                                        |
|          ▼                                                                                        |
|  (Module 6) Geospatial Disaster Segmentation & Decision Dashboard (Streamlit)                     |
|     └── Damage Severity Masking + Blocked Road Detection + GeoJSON & Ablation Report Export       |
+---------------------------------------------------------------------------------------------------+
```

## 5. Detailed Functional Specifications (Module-by-Module Blueprint)

### Module 0: Ingestion & Radiometric Calibration (`src/ingestion/radiometric.py`)

- **Inputs:** RGB/4-band UAV image (`.jpg`, `.png`, `.tif`) or video (`.mp4`), optional pre-disaster satellite GeoTIFF.
    
      
    
- **Functional Requirements:**
    
      
    1. `extract_metadata(filepath: str) -> dict`: Uses `ExifRead` and `Rasterio` to extract `GPSLatitude`, `GPSLongitude`, `RelativeAltitude`, `GimbalPitchDegree`, `BitDepth`, and affine geo-transform matrix. If EXIF tags are stripped in benchmark datasets, inject realistic synthetic telemetry headers and tag `metadata_source="synthetic_telemetry"`.
        
          
        
    2. `simulate_raw_and_tonemap(img_bgr: np.ndarray, gamma: float = 2.2, exposure_ev: float = 0.0) -> dict`:
        
          
        - Converts 8-bit sRGB to linear light $I_{\text{lin}} = (I_{\text{srgb}}/255)^{\gamma}$, quantizes to simulated 12-bit RAW (`uint16`, `0–4095`), applies synthetic dynamic-range expansion (simulating sun glint on water + dark building shadows), and recovers SDR via **Reinhard Global+Local Tone Mapping** (`cv2.createTonemapReinhard(gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.0)`).
            
              
            
    3. `convert_color_spaces(img_bgr: np.ndarray) -> dict`: Returns dictionary containing `YCbCr`, `LAB`, and `LUV` arrays so downstream restoration operates strictly on luminance $Y$ without introducing color-channel phase shifts.
        
          
        
    4. `compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray`: Computes $\text{NDWI} = \frac{G - \text{NIR}}{G + \text{NIR} + \epsilon}$ for multispectral water-boundary delineation.
        
          
        

### Module 1: Physics-Based Degradation Simulator (`src/degradation/simulator.py`)

- **Purpose:** Generates paired `(clean_gt, degraded_input)` frames with deterministic random seeds so restoration quality (`PSNR`, `SSIM`) and detection recovery (`mIoU`) can be measured objectively.
    
      
    
- **Functional Requirements:**
    
      
    1. `apply_motion_blur(img: np.ndarray, length: int = 15, angle: float = 35.0) -> Tuple[np.ndarray, np.ndarray]`: Generates a normalized linear Point Spread Function (PSF) kernel $h(x,y)$ rotated by `angle` and convolves via `cv2.filter2D`. Returns `(blurred_img, psf_kernel)`.
        
          
        
    2. `apply_atmospheric_haze(img: np.ndarray, beta: float = 1.2, airlight: float = 0.85) -> np.ndarray`: Applies Koschmieder's atmospheric scattering model $I(x) = J(x)t(x) + A(1 - t(x))$ where transmission $t(x) = e^{-\beta d(x)}$ uses a smooth synthetic depth/elevation gradient $d(x)$.
        
          
        
    3. `apply_sensor_noise(img: np.ndarray, shot_scale: float = 0.02, read_sigma: float = 8.0) -> np.ndarray`: Applies signal-dependent Poisson shot noise + additive Gaussian read noise.
        
          
        
    4. `degrade_pipeline(img: np.ndarray, severity: str = "severe") -> dict`: Presets `"mild"`, `"moderate"`, and `"severe"` combining blur + haze + noise + low-light gamma drop.
        
          
        

### Module 2: Spatial & Frequency Domain Restoration Engine (`src/restoration/engine.py`)

- **Functional Requirements:**
    
      
    1. `estimate_blur_fft(gray_y: np.ndarray) -> dict`:
        
          
        - Applies Hann windowing to suppress boundary spectral leakage, computes 2D-FFT $F(u,v) = \mathcal{F}\{Y_{\text{win}}\}$, and calculates log-magnitude spectrum $S(u,v) = \log(1 + \vert{}F(u,v)\vert{})$.
            
              
            
        - Detects dominant parallel dark interference fringes via Radon projection / Hough line transform on thresholded $S(u,v)$ to estimate blur angle $\hat{\theta}$ (perpendicular to spectral fringes) and blur length $\hat{L} = N / \Delta f$.
            
              
            
        - Returns `{"spectrum_vis": uint8_heatmap, "est_angle": float, "est_length": int}`.
            
              
            
    2. `wiener_deblur_freq(channel: np.ndarray, psf: np.ndarray, nsr: float = 0.015) -> np.ndarray`:
        
          
        - Pads `psf` to image dimensions $(H,W)$, shifts center via `np.fft.ifftshift`, and computes frequency-domain Wiener deconvolution:
            
              
            
            $$\hat{X}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + \text{NSR}} \cdot Y(u,v)$$
            
    3. `dehaze_guided_filter(img_bgr: np.ndarray, omega: float = 0.92, radius: int = 40, eps: float = 1e-3) -> np.ndarray`:
        
          
        - Computes Dark Channel $J^{\text{dark}}(x) = \min_{c \in \{R,G,B\}} \left(\min_{y \in \Omega(x)} J^c(y)\right)$, estimates atmospheric light $A$ from top $0.1\%$ brightest dark-channel pixels, refines raw transmission $\tilde{t}(x)$ via **Edge-Preserving Guided Filter**, and recovers scene radiance $J(x) = \frac{I(x) - A}{\max(t(x), 0.1)} + A$.
            
              
            
    4. `denoise_suite(img_bgr: np.ndarray, method: str = "anisotropic") -> np.ndarray`:
        
          
        - Supports selectable classical denoisers required by the syllabus:
            
              
            - `"adaptive_median"`: Variable-window impulse noise filter.
                
                  
                
            - `"bilateral"`: `cv2.bilateralFilter` spatial-range edge-preserving smoothing.
                
                  
                
            - `"nlm"`: `cv2.fastNlMeansDenoisingColored` Non-Local Means.
                
                  
                
            - `"anisotropic"`: **Perona–Malik Anisotropic Diffusion** ($c(\Vert{}\nabla I\Vert{}) = \exp(-(\Vert{}\nabla I\Vert{}/\kappa)^2)$) iterative PDE solver.
                
                  
                
            - `"tv"`: **Total Variation (TV) Chambolle** denoising (`skimage.restoration.denoise_tv_chambolle`).
                
                  
                
            - `"wavelet"`: **2D Discrete Wavelet Transform (`db4`, 3 levels)** with BayesShrink soft thresholding (`skimage.restoration.denoise_wavelet`).
                
                  
                
    5. `cleanup_compression_artifacts(img_bgr: np.ndarray) -> np.ndarray`: Applies rolling-guidance / bilateral smoothing tuned to 8×8 DCT block boundaries to remove JPEG/H.264 ringing.
        
          
        

### Module 3: Semantic ROI-Adaptive Compression Engine (`src/compression/roi_codec.py`)

- **Purpose:** Proves that allocating bits semantically preserves disaster detection accuracy at 60–75% lower bandwidth than uniform compression.
    
      
    
- **Functional Requirements:**
    
      
    1. `generate_roi_importance_map(img_bgr: np.ndarray, coarse_seg_probs: np.ndarray) -> np.ndarray`:
        
          
        - Fuses normalized Sobel/Laplacian structural gradient magnitude $E(x,y)$ with foreground disaster probability (buildings, roads, vehicles, debris) $P_{\text{fg}}(x,y)$ and Shannon prediction entropy $H(p) = -\sum p_i \log p_i$:
            
              
            
            $$M(x,y) = \text{clip}\big(0.4 \, E(x,y) + 0.4 \, P_{\text{fg}}(x,y) + 0.2 \, H(p(x,y)), \, 0, \, 1\big)$$
            
    2. `encode_roi_adaptive(img_bgr: np.ndarray, roi_mask: np.ndarray, target_bpp: float = 0.45, codec: str = "jpeg2000") -> dict`:
        
          
        - **Spatial Bit-Allocation Pre-Modulation:** Applies spatial-frequency pre-filtering (smoothing high-frequency background texture where $M(x,y) < 0.35$) and encodes foreground/background tiles or JPEG2000 wavelet bit-planes at differential quality levels (`Q_fg = 92`, `Q_bg = 18`) tuned to hit `target_bpp`.
            
              
            
        - Supports `"jpeg2000"` (via OpenJPEG/Pillow `.jp2`), `"webp"`, `"heif"` (`pillow-heif`), and baseline `"jpeg"` for direct comparison.
            
              
            
        - Returns `{"reconstructed_img": np.ndarray, "actual_bpp": float, "compression_ratio": float, "bit_allocation_heatmap": np.ndarray}`.
            
              
            
    3. `benchmark_video_codecs(video_path: str, target_kbps: int = 500) -> pd.DataFrame`:
        
          
        - Uses `ffmpeg-python` to transcode identical restored clips at `target_kbps` across **H.264 CAVLC** (`-coder 0`), **H.264 CABAC** (`-coder 1`), **HEVC/H.265** (`libx265`), and **MPEG-2** (`mpeg2video`), logging exact file size, bitrate, PSNR, SSIM, and downstream segmentation `mIoU`.
            
              
            
    4. `arithmetic_coding_demo(quantized_coeffs: np.ndarray) -> dict`: Implements a clean Python probability-model **Arithmetic Coder** on quantized DWT subbands, reporting theoretical Shannon entropy vs. achieved bits-per-symbol.
        
          
        

### Module 4: Multi-Temporal Satellite-to-UAV Geo-Registration (`src/registration/geo_align.py`)

- **Functional Requirements:**
    
      
    1. `register_cross_view(uav_img: np.ndarray, sat_img: np.ndarray, detector: str = "AKAZE", use_cnn_fallback: bool = True) -> dict`:
        
          
        - Extracts local keypoints & descriptors using **AKAZE** (default, fast and scale/rotation-invariant) or **SIFT** (`cv2.SIFT_create()`).
            
              
            
        - Matches descriptors via `BFMatcher` + Lowe's ratio test (`ratio = 0.75`), then estimates $3 \times 3$ homography $H$ via **RANSAC** (`cv2.findHomography(..., cv2.RANSAC, ransacReprojThreshold=5.0)`).
            
              
            
        - **CNN Deep Feature Hybrid Fallback:** If RANSAC inliers $< 12$ (due to severe flood/rubble appearance change), extracts dense spatial feature embeddings from `MobileNetV3-Small` (`layer 4`) and performs coarse-to-fine cosine similarity grid matching before refining homography.
            
              
            
        - Warps `uav_img` into `sat_img` coordinate frame (`cv2.warpPerspective`) and computes pixel-wise structural change map (`|Gray_post - Gray_pre|`).
            
              
            
        - Returns `{"warped_uav": np.ndarray, "match_vis": np.ndarray, "inlier_count": int, "reproj_error": float, "change_heatmap": np.ndarray}`.
            
              
            

### Module 5: Real-Time Video Stabilization, Motion & Tracking (`src/video_analytics/tracker.py`)

- **Functional Requirements:**
    
      
    1. `stabilize_video_stream(frames: list[np.ndarray], smoothing_radius: int = 15) -> tuple[list[np.ndarray], dict]`:
        
          
        - Detects Shi-Tomasi corners (`cv2.goodFeaturesToTrack`) and tracks frame-to-frame motion via **Lucas-Kanade Sparse Optical Flow** (`cv2.calcOpticalFlowPyrLK`).
            
              
            
        - Estimates rigid/affine transforms (`cv2.estimateAffinePartial2D`), accumulates cumulative trajectory $(x, y, \theta)$, smooths via moving-average temporal filter, and warps frames with a 1.05× crop border to eliminate black edges.
            
              
            
        - Returns stabilized frames + pre/post trajectory jitter arrays for plotting.
            
              
            
    2. `detect_dynamic_hazards_mog2(frames: list[np.ndarray]) -> list[np.ndarray]`: Uses `cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=True)` after camera-motion compensation to isolate active flood surges, landslides, or moving debris.
        
          
        
    3. `track_rescue_assets(frames: list[np.ndarray]) -> list[dict]`: Runs **YOLOv8n** (FP16 on GTX 1650 Ti) + **DeepSORT** (`deep-sort-realtime`) to assign persistent IDs to vehicles, boats, and people across stabilized frames.
        
          
        
    4. `interpolate_low_fps(frame_t0: np.ndarray, frame_t1: np.ndarray) -> np.ndarray`: Computes dense optical flow (`cv2.calcOpticalFlowFarneback`) to synthesize intermediate frame $t_{0.5}$ when uplink frame rate drops from 30 FPS to 15 FPS.
        
          
        

### Module 6: Geospatial Disaster Assessment & Export (`src/analytics/disaster_seg.py`)

- **Functional Requirements:**
    
      
    1. `segment_disaster_scene(img_bgr: np.ndarray) -> dict`:
        
          
        - Runs a lightweight semantic segmentation model (`SegFormer-B0` or `UNet-MobileNetV3` fine-tuned/loaded on RescueNet classes) at `640×480` on CUDA (`GTX 1650 Ti`).
            
              
            
        - Outputs pixel-level class map covering: `Building-Intact`, `Building-Minor-Damage`, `Building-Major-Damage`, `Building-Destroyed`, `Road-Clear`, `Road-Blocked`, `Flood-Water`, `Debris/Trees`, `Vehicle`.
            
              
            
        - Computes actionable KPIs: `% Structural Damage`, `Blocked Road Ratio`, `Flooded Area (m²)` (scaled via EXIF altitude & GSD).
            
              
            
    2. `export_geojson_report(seg_mask: np.ndarray, metadata: dict, output_path: str) -> str`: Vectorizes damaged building and blocked road contours (`cv2.findContours`), transforms pixel coordinates to `(lat, lon)` using the affine geo-transform, and writes an RFC-7946 compliant `.geojson` file ready for QGIS/emergency GIS ingestion.
        
          
        

## 6. User Interface & UX Specification (`GeoPulse Command Center`)

Built in **Streamlit** with custom dark-theme tactical CSS (`#0B0F19` background, `#111827` cards, `#00E5FF` cyan telemetry accents, `#FF3B30` hazard alerts), `streamlit-image-comparison` interactive sliders, and `Plotly` charts.

  

### Navigation & 4-Tab Layout

#### Sidebar: Mission Control & Pipeline Parameters

- **Input Mode Selector:** `Sample Disaster Preset (Hurricane / Flood / Urban Rubble)` | `Upload Custom UAV Image (.jpg/.png/.tif)` | `Upload UAV Video (.mp4)`.
    
      
    
- **Simulated Link & Weather Controls:**
    
      
    - Degradation Severity: `None (Clean)` | `Mild` | `Moderate` | `Severe (Blur + Haze + Noise + Low-Light)`
        
          
        
    - Uplink Bandwidth Throttle: Slider from `150 kbps` (Tactical Radio) to `4000 kbps` (4G/5G)
        
          
        
    - Restoration Filter Selector: `Auto (FFT-Wiener + Guided + Anisotropic)` | `Bilateral` | `Non-Local Means` | `Wavelet BayesShrink` | `Total Variation`
        
          
        
    - Compression Codec Selector: `ROI-Adaptive JPEG2000` | `Standard JPEG (Uniform)` | `WebP` | `HEIF`
        
          
        

#### Tab 1: Live Restoration & Semantic Triage (Primary Viva Demo View)

- **Top KPI Strip (4 Metric Cards):** `PSNR Gain (+dB)` · `SSIM Recovery` · `Bandwidth Saved (%)` · `Critical Damaged Structures Detected`
    
      
    
- **Center Interactive Wiper (`streamlit-image-comparison`):** Full-width draggable split-screen slider comparing **Raw Degraded Input** (left) vs. **Restored + Disaster Segmentation Overlay** (right).
    
      
    
- **Diagnostic X-Ray Row (3 Columns):**
    
      
    1. _2D-FFT Log-Magnitude Spectrum:_ Visualizes the spectral zero-crossing line with detected blur angle $\hat{\theta}$ and length $\hat{L}$ annotated.
        
          
        
    2. _Radiometric & Color Space Inspector:_ Toggle between `HDR Tone-Mapped`, `YCbCr Luminance`, `CIELAB`, and `NDWI Water Mask`.
        
          
        
    3. _Actionable Triage Summary:_ Bar chart of damaged vs. intact buildings and clear vs. blocked road segments.
        
          
        

#### Tab 2: Bandwidth & Compression Rate-Distortion Lab

- **Visual Side-by-Side Comparison at Target Bitrate:** Displays **Naive Uniform Compression** (blocky, smeared cracks) vs. **GeoPulse ROI-Adaptive Compression** (crisp buildings/roads, smooth background) at the exact same file size (`KB` / `bpp`).
    
      
    
- **Spatial Bit-Allocation Heatmap:** Jet/Viridis heatmap showing high bit-density concentrated on damaged buildings and roads.
    
      
    
- **Interactive Plotly Rate-Distortion Curve:** Plots `Bitrate (kbps)` vs. `Downstream Segmentation mIoU` across `JPEG`, `JPEG2000`, `WebP`, `H.264 CAVLC`, `H.264 CABAC`, and `HEVC`.
    
      
    

#### Tab 3: Multi-Temporal Satellite Geo-Registration

- **Keypoint Match Canvas:** Displays `AKAZE` / `SIFT` + `RANSAC` green inlier correspondence lines connecting oblique UAV frame to nadir pre-disaster satellite imagery (with live toggle to test `CNN Embedding Hybrid` mode).
    
      
    
- **Before vs. After Disaster Slider:** Interactive wiper comparing pre-disaster satellite baseline against perspective-warped post-disaster drone imagery + structural change heatmap.
    
      
    
- **One-Click GeoJSON Export:** Button to download `geopulse_damage_assessment.geojson`.
    
      
    

#### Tab 4: Stabilized Video Telemetry & Asset Tracking

- **Dual Synchronized Video Player:** Raw shaky/hazy drone clip vs. Optical-Flow Stabilized + Restored + `DeepSORT` tracked clip.
    
      
    
- **Camera Jitter Reduction Graph:** Plotly time-series comparing raw frame-to-frame `(dx, dy, dθ)` shake against smoothed trajectory.
    
      
    
- **Active Asset Counter:** Live count of tracked rescue vehicles/boats and MOG2 dynamic flood-flow area.
    
      
    

## 7. Repository Directory Structure & File Blueprint

Plaintext

```
geopulse/
├── app.py                          # Main Streamlit Command Center UI (4-Tab Dashboard)
├── requirements.txt                # Pinned Python dependencies (CUDA-compatible)
├── README.md                       # Setup, architecture overview & viva demo guide
├── configs/
│   └── default_config.yaml         # VRAM caps, resolution (640x480), codec bitrates, class palette
├── data/
│   ├── samples/                    # Pre-bundled 15 curated demo images + 2 UAV videos (<150 MB)
│   └── satellite_pairs/            # Pre/post-disaster aligned satellite + UAV tile pairs
├── scripts/
│   ├── download_subsets.py         # Automated downloader for curated RescueNet & VisDrone subsets
│   └── run_full_benchmark.py       # Headless script generating all ablation CSVs & report plots
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   └── radiometric.py          # EXIF/GeoTIFF parser, 12-bit RAW sim, HDR tone mapping, YCbCr/LAB/NDWI
│   ├── degradation/
│   │   └── simulator.py            # Physics-based PSF motion blur, Koschmieder haze, sensor noise
│   ├── restoration/
│   │   └── engine.py               # 2D-FFT blur estimator, Wiener filter, Guided dehaze, NLM/PM/TV/Wavelet
│   ├── compression/
│   │   └── roi_codec.py            # Semantic ROI mask, JPEG2000/WebP/HEIF coder, FFmpeg CABAC/CAVLC, Arithmetic coder
│   ├── registration/
│   │   └── geo_align.py            # AKAZE/SIFT + RANSAC homography + MobileNetV3 deep embedding fallback
│   ├── video_analytics/
│   │   └── tracker.py              # Lucas-Kanade optical flow stabilization, MOG2, YOLOv8n + DeepSORT, interpolation
│   ├── analytics/
│   │   └── disaster_seg.py         # Lightweight disaster segmentation + GeoJSON vector exporter
│   └── utils/
│       └── metrics.py              # PSNR, SSIM, BRISQUE, EPI, mIoU, and BD-Rate calculations
└── outputs/
    ├── benchmarks/                 # Auto-generated ablation tables (.csv) & publication charts (.png)
    └── geojson/                    # Exported disaster assessment GeoJSON files
```

## 8. Data Acquisition & Storage Budget Plan

To respect the **~91 GB available disk space** and enable rapid local iteration without waiting hours for 22 GB downloads:

  

|**Dataset**|**Role in GeoPulse**|**Curated Footprint**|**Acquisition Method**|
|---|---|---|---|
|**RescueNet (UAV Post-Disaster)**|High-res drone disaster scenes + 10-class pixel segmentation ground truth|**~600 MB** (250 stratified images + masks)|Download validation/sample split via Kaggle API / `scripts/download_subsets.py`|
|**VisDrone-MOT (`valset`)**|Real aerial UAV video sequences for optical-flow stabilization & DeepSORT tracking|**~1.48 GB** (7 aerial video sequences)|Direct download from official `VisDrone/VisDrone-Dataset` release|
|**xBD / Sentinel-2 Pairs**|Pre-disaster nadir satellite tiles + post-disaster aerial pairs + 4-band NIR/RGB|**~350 MB** (20 paired disaster tiles)|Bundled / fetched via `scripts/download_subsets.py`|
|**Pretrained Weights**|`YOLOv8n.pt` (6 MB) + `SegFormer-B0` / `MobileNetV3-UNet` (15 MB)|**~25 MB**|Auto-cached on first run via `ultralytics` / `torchvision`|
|**Total Disk Footprint**|**Complete end-to-end data + models + outputs**|**~2.45 GB**|Leaves >88 GB free on local SSD|

## 9. Quantitative Evaluation Framework & Auto-Report Generator

Even if a formal lab report is not explicitly announced yet, professors routinely ask for quantitative justification during end-semester viva evaluations. Running `python scripts/run_full_benchmark.py` will automatically execute an evaluation sweep over 100 test images and export **3 publication-ready figures + CSV tables** to `outputs/benchmarks/`:

  

1. **Table 1 — Spatial & Frequency Restoration Ablation (`restoration_ablation.csv`):**
    
      
    - Evaluates `Degraded Input` vs. `Adaptive Median` vs. `Bilateral` vs. `Non-Local Means` vs. `Wavelet BayesShrink` vs. `Perona–Malik Anisotropic` vs. `Full GeoPulse Pipeline (FFT-Wiener + Guided + Anisotropic)` across **PSNR (dB)**, **SSIM**, **Edge Preservation Index (EPI)**, and **Runtime (ms/frame)**.
        
          
        
2. **Table 2 — Compression Rate-Distortion & Codec Comparison (`compression_rd_curve.csv`):**
    
      
    - Compares `JPEG`, `WebP`, `HEIF`, `JPEG2000 (Uniform)`, and `GeoPulse ROI-Adaptive JPEG2000` at `0.15, 0.30, 0.50, 0.80 bpp`, plus `FFmpeg H.264 CAVLC vs. CABAC vs. HEVC` at `250, 500, 1000 kbps`, measuring **Bitrate**, **Foreground PSNR**, and **Downstream Disaster Segmentation mIoU**.
        
          
        
3. **Table 3 — Cross-View Registration & Video Stabilization Benchmark (`registration_video_metrics.csv`):**
    
      
    - Compares `ORB + RANSAC` vs. `SIFT + RANSAC` vs. `AKAZE + RANSAC + CNN Hybrid` on **Inlier Ratio**, **Homography Reprojection Error (px)**, and **Trajectory Jitter Reduction (%)**.
        
          
        

## 10. Focused Build Plan & AI Coding Assistant Handoff Protocol

### 10.1 Execution Schedule (High-Intensity Focused Sprint)

- **Block 1 (Hours 1–4) — Environment, Data & Core Classical IP (`Modules 0, 1, 2`):**
    
      
    - Set up Python virtual environment with PyTorch CUDA (for GTX 1650 Ti), OpenCV, SciPy, PyWavelets, Rasterio, and Streamlit.
        
          
        
    - Run `download_subsets.py` to populate `data/samples/`.
        
          
        
    - Implement and unit-test `radiometric.py`, `simulator.py`, and `engine.py` (verify 2D-FFT blur angle estimation and Guided Filter dehazing visually).
        
          
        
- **Block 2 (Hours 5–8) — ROI Compression, Registration & Video Analytics (`Modules 3, 4, 5, 6`):**
    
      
    - Implement `roi_codec.py` (ROI mask + JPEG2000 + FFmpeg CABAC/CAVLC wrapper + Arithmetic coder).
        
          
        
    - Implement `geo_align.py` (`AKAZE`/`SIFT` + `RANSAC` + `MobileNetV3` embedding fallback).
        
          
        
    - Implement `tracker.py` (Lucas-Kanade stabilization + MOG2 + `YOLOv8n` + `DeepSORT`) and `disaster_seg.py` (segmentation + GeoJSON export).
        
          
        
- **Block 3 (Hours 9–12) — Streamlit Command Center & Benchmark Export (`app.py` + `run_full_benchmark.py`):**
    
      
    - Build the 4-tab dark-mode Streamlit dashboard (`app.py`) with `streamlit-image-comparison` sliders and Plotly interactive graphs.
        
          
        
    - Execute `run_full_benchmark.py` to generate all CSV ablation tables and high-DPI charts for viva presentation.
        
          
        

### 10.2 How to Use This PRD with an AI Coding Assistant

When feeding this PRD into an AI assistant (e.g., Cursor, GitHub Copilot, or Gemini) to generate code step-by-step:

  

1. **Generate in strict module order:** Prompt the AI to implement one file at a time (`requirements.txt` $\rightarrow$ `src/ingestion/radiometric.py` $\rightarrow$ `src/degradation/simulator.py` $\rightarrow$ `src/restoration/engine.py` $\rightarrow$ `src/compression/roi_codec.py` $\rightarrow$ `src/registration/geo_align.py` $\rightarrow$ `src/video_analytics/tracker.py` $\rightarrow$ `src/analytics/disaster_seg.py` $\rightarrow$ `app.py` $\rightarrow$ `scripts/run_full_benchmark.py`).
    
      
    
2. **Enforce hardware guardrails:** Instruct the AI that every image/video processor must default to `max_dim=768` (or `640×480` for neural inference), release CUDA cache (`torch.cuda.empty_cache()`) between heavy operations, and include graceful CPU fallbacks so the app never crashes on `4 GB VRAM`.
    
      
    
3. **Require self-contained fallback data:** Ensure `scripts/download_subsets.py` also generates synthetic multi-band/EXIF-tagged test assets locally if external network requests fail during demo day.
    
      
    

Would you like to start generating the code immediately—beginning with **`requirements.txt` + `src/ingestion/radiometric.py` + `src/degradation/simulator.py` + `src/restoration/engine.py`**?