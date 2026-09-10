// Keep this transformation stable: URLs and saved progress use these IDs.
export function slugify(str) {
  return str.toLowerCase().replace(/[()]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}
