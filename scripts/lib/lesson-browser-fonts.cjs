const fs = require('node:fs');

// Reuse the retained site-font fixture for deterministic offline browser review.
// This is test routing only; it does not change application font loading.
module.exports = async function installLessonFonts(surface) {
  const fonts = JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json', 'utf8'));
  await surface.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css' }));
  await surface.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()]
    ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf' }) : route.abort());
};
