// Regenerate raster icons from public/favicon.svg. Uses the same Playwright
// setup as verify-site-shell.cjs (PLAYWRIGHT_PACKAGE; installed Microsoft Edge).
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const publicRoot = path.resolve(__dirname, '../public');

async function main() {
  const svg = await fs.readFile(path.join(publicRoot, 'favicon.svg'), 'utf8');
  const sizes = [16, 32, 48];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const images = await page.evaluate(async ({ svg, sizes }) => {
      const image = new Image();
      image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
      await image.decode();
      return [...sizes, 180].map(size => {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        const context = canvas.getContext('2d');
        // iOS supplies its own corner mask; keep the touch icon fully opaque.
        if (size === 180) {
          context.fillStyle = '#0a0a0a';
          context.fillRect(0, 0, size, size);
        }
        context.drawImage(image, 0, 0, size, size);
        return canvas.toDataURL('image/png').split(',')[1];
      });
    }, { svg, sizes });

    const pngs = images.slice(0, sizes.length).map(image => Buffer.from(image, 'base64'));
    const directory = Buffer.alloc(6 + sizes.length * 16);
    directory.writeUInt16LE(1, 2); // ICO image type.
    directory.writeUInt16LE(sizes.length, 4);
    let offset = directory.length;
    sizes.forEach((size, index) => {
      const entry = 6 + index * 16;
      directory[entry] = directory[entry + 1] = size;
      directory.writeUInt16LE(1, entry + 4);
      directory.writeUInt16LE(32, entry + 6);
      directory.writeUInt32LE(pngs[index].length, entry + 8);
      directory.writeUInt32LE(offset, entry + 12);
      offset += pngs[index].length;
    });
    await fs.writeFile(path.join(publicRoot, 'favicon.ico'), Buffer.concat([directory, ...pngs]));
    await fs.writeFile(path.join(publicRoot, 'apple-touch-icon.png'), Buffer.from(images.at(-1), 'base64'));
    console.log('Generated favicon.ico (16, 32, 48 px) and apple-touch-icon.png (180 px).');
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
