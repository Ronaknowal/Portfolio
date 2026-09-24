const dateFormatter = new Intl.DateTimeFormat("en", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
export function formatArticleDate(date) { return dateFormatter.format(new Date(`${date}T00:00:00Z`)); }
