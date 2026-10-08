// Image processing that runs in the browser on this machine: no model and no upload involved.

export type Filter = 'enhance' | 'grayscale' | 'night' | 'thermal' | 'edges' | 'sharpen';

export const FILTERS: { id: Filter; label: string; detail: string }[] = [
  { id: 'enhance', label: 'Auto-enhance', detail: 'Stretches contrast and lifts colour, as for a dull or hazy photo' },
  { id: 'sharpen', label: 'Sharpen', detail: 'Brings out fine detail and edges' },
  { id: 'grayscale', label: 'Grayscale', detail: 'Removes colour for printing or comparison' },
  { id: 'night', label: 'Night vision', detail: 'Green phosphor look of an image intensifier' },
  { id: 'thermal', label: 'Thermal', detail: 'False colour by brightness, like a thermal imager' },
  { id: 'edges', label: 'Edge detect', detail: 'Outlines of shapes and structures (Sobel)' },
];

/** The picture as a canvas, scaled down to at most `max` pixels on its longer side. */
export function toCanvas(image: HTMLImageElement, max: number): HTMLCanvasElement {
  const scale = Math.min(1, max / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.getContext('2d')!.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The file could not be read as a picture.'));
    image.src = url;
  });
}

const luma = (d: Uint8ClampedArray, i: number) => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];

/** Simple figures about a picture: average brightness and the dominant colour. */
export function measure(canvas: HTMLCanvasElement): { brightness: number; colour: string } {
  const d = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
  let light = 0, r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < d.length; i += 16) {
    light += luma(d, i);
    r += d[i]; g += d[i + 1]; b += d[i + 2];
    n++;
  }
  const hex = (v: number) => Math.round(v / n).toString(16).padStart(2, '0');
  return { brightness: Math.round((light / n / 255) * 100), colour: `#${hex(r)}${hex(g)}${hex(b)}` };
}

function convolve(src: ImageData, kernel: number[]): ImageData {
  const { width: w, height: h, data: s } = src;
  const out = new ImageData(w, h);
  const o = out.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const yy = Math.min(h - 1, Math.max(0, y + ky)), xx = Math.min(w - 1, Math.max(0, x + kx));
            sum += s[(yy * w + xx) * 4 + c] * kernel[(ky + 1) * 3 + kx + 1];
          }
        }
        o[i + c] = sum;
      }
      o[i + 3] = s[i + 3];
    }
  }
  return out;
}

/** Thermal palette: black, violet, red, orange, yellow, white. */
const RAMP = [[0, 0, 0], [70, 0, 120], [190, 20, 60], [245, 110, 0], [255, 220, 40], [255, 255, 255]];
function ramp(t: number): number[] {
  const at = Math.min(RAMP.length - 1.001, Math.max(0, t) * (RAMP.length - 1));
  const lo = RAMP[Math.floor(at)], hi = RAMP[Math.floor(at) + 1], f = at - Math.floor(at);
  return lo.map((v, k) => v + (hi[k] - v) * f);
}

/** The filtered picture on a new canvas. */
export function applyFilter(source: HTMLCanvasElement, filter: Filter): HTMLCanvasElement {
  const { width: w, height: h } = source;
  const input = source.getContext('2d')!.getImageData(0, 0, w, h);
  let output = new ImageData(new Uint8ClampedArray(input.data), w, h);
  const d = output.data;

  if (filter === 'enhance') {
    // Stretch each channel between its 1st and 99th percentile, then add a little saturation
    for (let c = 0; c < 3; c++) {
      const counts = new Array(256).fill(0);
      for (let i = c; i < d.length; i += 4) counts[d[i]]++;
      const total = w * h;
      let low = 0, high = 255, seen = 0;
      for (let v = 0; v < 256; v++) { seen += counts[v]; if (seen > total * 0.01) { low = v; break; } }
      seen = 0;
      for (let v = 255; v >= 0; v--) { seen += counts[v]; if (seen > total * 0.01) { high = v; break; } }
      const span = Math.max(1, high - low);
      for (let i = c; i < d.length; i += 4) d[i] = ((d[i] - low) * 255) / span;
    }
    for (let i = 0; i < d.length; i += 4) {
      const l = luma(d, i);
      for (let c = 0; c < 3; c++) d[i + c] = l + (d[i + c] - l) * 1.2;
    }
  } else if (filter === 'grayscale') {
    for (let i = 0; i < d.length; i += 4) d[i] = d[i + 1] = d[i + 2] = luma(d, i);
  } else if (filter === 'night') {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const dx = x / w - 0.5, dy = y / h - 0.5;
        const vignette = 1 - Math.min(1, (dx * dx + dy * dy) * 2.2);
        const l = Math.min(255, luma(d, i) * 1.5 + (Math.random() - 0.5) * 28) * vignette;
        d[i] = l * 0.15; d[i + 1] = l; d[i + 2] = l * 0.25;
      }
    }
  } else if (filter === 'thermal') {
    for (let i = 0; i < d.length; i += 4) {
      const [r, g, b] = ramp(luma(d, i) / 255);
      d[i] = r; d[i + 1] = g; d[i + 2] = b;
    }
  } else if (filter === 'sharpen') {
    output = convolve(input, [0, -1, 0, -1, 5, -1, 0, -1, 0]);
  } else if (filter === 'edges') {
    const s = input.data;
    const at = (x: number, y: number) => luma(s, (Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))) * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const gx = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1) + at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
        const gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
        const i = (y * w + x) * 4;
        const v = Math.min(255, Math.hypot(gx, gy));
        d[i] = v * 0.85; d[i + 1] = v; d[i + 2] = v * 0.6; d[i + 3] = 255;
      }
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d')!.putImageData(output, 0, 0);
  return canvas;
}
