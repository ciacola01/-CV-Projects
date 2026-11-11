<center>
	<b>Recchia Luca - VR563083</b> <br>
	<a href="https://github.com/ciacola01/-CV-Projects.git">Repository GitHub</a>
</center>

---

# Objective and Data
The goal of the project is to build, apply and compare a classical image enhancement pipeline for images affected by non-uniform illumination, low contrast and noise. 
The images were taken from my library and they were captured during my travels over the past few years; so I degraded synthetically degraded them to simulate these conditions. 

The dataset consists of four images:
- `MonzaGP.jpeg`: personal photograph taken during the 2026 Italian GP
- `Oceanografic.jpeg`: personal photograph taken in Valencia
- `Barcelona.jpeg`: personal photograph taken in January at Park Güell
- `Champions.jpeg`: personal photograph taken at the Barcelona FC Museum 

**Synthetic Degradation**: To ensure reproducibility, we applied a fixed transformation to all images:
- Contrast reduction: $I' = \text{ContrastFactor} * I + \text{BrightnessOffset}$
- Gaussian noise: with $\text{mean}=0, \text{standard deviation}=10, \text{seed}=42$

The original images are the absolute references for the metric evaluation. 

---

# Methods and Parameters
The pipeline implementes the baseline and four enhancement techniques. The initial degradation parameters are set in `degrade.py` as `ALPHA = 0.5`(50% contrast reduction), `BETA = 30` (brightness offset), `SIGMA = 10.0` (Gaussian noise) and `SEED = 42` is just a fixed seed for reproducibility. 

**Baseline**: the degraded image without further processing with these formulas:
$$
\begin{matrix}
    I_{\text{degraded}} & = & \text{ContrastFactor} * I + \text{BrightnessOffset} \\
    N(x,y,c) & \sim & \mathcal{N}(0, \sigma^2)
\end{matrix}
$$

And then I return the degraded image clipped to an int with the relative cast `.astype(np.uint8)`.

**Linear Contrast Stretching**: This method linearly rescales the intensity levels to span the full range $[0, 255]$ using the formula $L' = \dfrac{L - L_\text{min}}{L_\text{max} - L_\text{min}} \times 255$. 
To avoid color shifts, the operation is applied only to the luminance channel $L$, preserving the chrominance information and avoid color shifts. 
*Expected trade-offs*: Recovers global brightness and contrast, but amplifies any existing noise. 

**Histogram Equalization on CIELab**: This non-linear operation redistributes pixel intensities to approximate a uniform histogram, allocating more levels to frequent values. 
Equalizing only the $L$ channel prevents severe color shifts. 
*Expected trade-offs*: excellent for local contrast, but strongly amplifies noise, producing a "grainy" appearance in uniform areas. 

**Histogram Equalization on HSV**: Similar to CIELab approach but operating on the Value ($V$) channel, separating Hue and Saturation. It tends to produce harsher, more saturated results and may shift the apparent hue in extreme regions. 
*Expected trade-offs*: it is similar to CIELab, but it looks more aggressive in saturated areas, like the blues in the Oceanografic or Champions trophy. 

**Bilateral Filter + Stretching**: A combination of denoising and contrast enhancement. 
The bilateral filter is applied first to remove noise before stretching, while preserving edges. The parameters are `d=7` (neighborhood diameter), `sigmaColor=75` (color tolerance)
and `sigmaSpace = 75` (spatial extent)
*Expected trade-off*: Excellent noise removal and edge-preservation, but risks over-smoothing fine textures. 

---
# Results: Visual Comparison and Metrics
Side-by-side comparisons were generated for each image, including the original, degraded baseline and all enhancement outputs.

**MonzaGP**
The original F1 car is obscured by a chain-link fence under strong sunlight, which creates deep shadows and harsh highlights. 
The degradation flattens the contrast hiding the details of the fence and of the F1 car. 
- *Linear Stretching*: improves global contrast and legibility, moreover the sky is a natural blue. 
- *Histogram Equalizations (CIELab and HSV)* are highly aggressive turning the sky into noisy mess. 
- *Bilateral + Stretch* cleans the sky and smooth the grass while keeping the fence sharp. 

![[MonzaGP_comparison.png]]

| Image        | Method                         | MSE       | PSNR (dB) |
| ------------ | ------------------------------ | --------- | --------- |
| MonzaGP.jpeg | Baseline                       | 1794.2195 | 15.59     |
| MonzaGP.jpeg | Linear Stretching              | 458.0962  | 21.52     |
| MonzaGP.jpeg | Histogram Equalization CIE Lab | 602.2585  | 20.33     |
| MonzaGP.jpeg | Histogram Equalization HSV     | 462.1481  | 21.48     |
| MonzaGP.jpeg | Bilateral + Stretch            | 261.7122  | 23.95     |

**Oceanografic**
The shark is in low contrast against a strong blue background. The degradation flattens the dynamic range making the shark blend into the background. 
- *Linear Stretching*: expands the blue intensity range, slightly improves the visibility of the shark's silhouette. 
- *Histogram Equalizations (CIELab and HSV)* reveal the shark's silhouette clearly, but CIELab produces a luminous, noisy image, while HSV clearly divides the colors.
- *Bilateral + Stretch* produces a clean image without noise but it is darker than the original one. 

![[Oceanografic_comparison.png]]

| Image        | Method                         | MSE       | PSNR (dB) |
| ------------ | ------------------------------ | --------- | --------- |
| Oceanografic.jpeg | Baseline                  | 1654.1890 |15.94|
| Oceanografic.jpeg | Linear Stretching              | 1259.5073|17.13|
| Oceanografic.jpeg | Histogram Equalization CIE Lab | 4625.5273|11.48|
| Oceanografic.jpeg | Histogram Equalization HSV     | 2260.4016|14.59|
| Oceanografic.jpeg | Bilateral + Stretch            | 1913.0543|15.31   |

**Barcelona**
The extreme dynamic range (bright sky to dark foreground) is reduced by the degradation.
- *Linear Stretching*: restore vibrancy and a natural look, though noise remains. 
- *Histogram Equalizations (CIELab and HSV)* perform poorly lacking a dominant peak to redistribute, they compress mid-tones, resulting in a muddy foreground ans overexposed sky, visually comparable to the baseline. 
- *Bilateral + Stretch* cleans the noise but zooming in reveals an over-smooth texture. 

![[Barcelona_comparison.png]]

| Image        | Method                         | MSE       | PSNR (dB) |
| ------------ | ------------------------------ | --------- | --------- |
| Barcelona.jpeg | Baseline                  | 4215.7310|11.88|
| Barcelona.jpeg | Linear Stretching              | 693.4457|19.72|
| Barcelona.jpeg | Histogram Equalization CIE Lab | 4183.7090|11.92|
| Barcelona.jpeg | Histogram Equalization HSV     | 5391.3218|10.81|
| Barcelona.jpeg | Bilateral + Stretch            | 437.5177|21.72|


**Champions**
The metallic trophy is characterized by bright highlights and deep shadows, against a dark background with light projections. 
- *Linear stretching*: improves the visibility of the trophy's inscriptions and the details of the reflections.
- *Histogram Equalizations (CIELab and HSV)* amplifies the noise and the HSV Histogram Equalization slightly alters the deep blue tones. 
- *Bilateral + Stretch* achieves the best result by removing background noise, though the the smoothing slightly flattens the trphy's inscriptions. 

![[Champions_comparison.png]]

| Image        | Method                         | MSE       | PSNR (dB) |
| ------------ | ------------------------------ | --------- | --------- |
| Champions.jpeg | Baseline                  | 2056.4563 | 15.00|
| Champions.jpeg | Linear Stretching              | 794.5435|19.13|
| Champions.jpeg | Histogram Equalization CIE Lab | 1669.1425|15.91|
| Champions.jpeg | Histogram Equalization HSV     | 460.0134|21.50|
| Champions.jpeg | Bilateral + Stretch            | 382.3455|22.31|

---

# Discussion

**Case 1: Oceanografic (The Shark)**
In this case the Histogram Equalization reaches the worst numerical result (PSNR=11.48dB and MSE=4625.5273), far below the Baseline. 
The underwater scene is dominated by the blue spectrum; Histogram Equalization aggressively spreads these intensities, changing the color distribution and amplifying the Gaussian noise. Qualitatively Histogram Equalization is the only method that clearly reveals the silhouette of the shark and the surface light patterns. Conversely, Bilateral + Stretch achieves a higher PSNR (15.31 dB) and a cleaner image, but over-smooths the underwater details. This demonstrates that a higher PSNR does not always correspond to better visual informativeness.

**Case 2: Barcelona (Panorama) - Why HE Fails on Broad Dynamic Ranges**
Histogram Equalization achieves a PSNR almost identical to the Baseline (11.92 dB vs. 11.88 dB). Since the image already contains a broad dynamic range, Histogram Equalizationlacks dominant intensity peaks to redistribute. Instead, it flattens the sky and darken the foliage. Linear Stretching (PSNR = 19.72 dB) performs significantly better, and Bilateral + Stretch achieves the best numerical (21.48 dB) and visual results, removing the noise from the trees while restoring a natural sky gradient.

**Qualitative Preference and Artifacts**
In MonzaGP, Linear Stretching offers a natural look, but Bilateral + Stretch is the best overall: it cleans the sky while keeping the fence details visible without color shifts.
In Champions, the Bilateral Filter creates a slightly "plastic" effect on the metallic reflections, but it successfully removes the heavy noise from the dark blue background. Linear Stretching preserves the reflections better but leaves visible grain.

**Observed Artifacts**
Overall Histogram Equalization causes severe noise amplification and risks unnatural color shifts, while Bilateral Filtering risks over-smoothing (loss of fine details) if sigmaColor is too high.

---

# Conclusion
While Bilateral Filter + Stretch consistently achieved the best PSNR and visually pleasing results, Histogram Equalization proved to be the most informative for low-contrast underwater scenes, despite its poor numerical metrics. This highlights the importance of combining quantitative metrics with qualitative assessment when evaluating image enhancement pipelines. 